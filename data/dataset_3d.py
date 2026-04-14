import os, sys
import random
import numpy as np
import math

import h5py
from collections import defaultdict

import yaml
from easydict import EasyDict

import torch
import torch.utils.data as data

from utils.io import IO
from utils.build import DATASETS
from utils.logger import *
from utils.build import build_dataset_from_cfg
import json
import pickle
from PIL import Image


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    for b in range(batch_pc.shape[0]):
        dropout_ratio =  np.random.random()*max_dropout_ratio # 0~0.875
        drop_idx = np.where(np.random.random((batch_pc.shape[1]))<=dropout_ratio)[0]
        if len(drop_idx)>0:
            batch_pc[b,drop_idx,:] = batch_pc[b,0,:] # set to the first point
    return batch_pc


def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
    """
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, B)
    for batch_index in range(B):
        batch_data[batch_index,:,:] *= scales[batch_index]
    return batch_data


def shift_point_cloud(batch_data, shift_range=0.1):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, shifted batch of point clouds
    """
    B, N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B,3))
    for batch_index in range(B):
        batch_data[batch_index,:,:] += shifts[batch_index,:]
    return batch_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data


def rotate_perturbation_point_cloud(batch_data, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly perturb the point clouds by small rotations
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)
        Rx = np.array([[1,0,0],
                       [0,np.cos(angles[0]),-np.sin(angles[0])],
                       [0,np.sin(angles[0]),np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                       [0,1,0],
                       [-np.sin(angles[1]),0,np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                       [np.sin(angles[2]),np.cos(angles[2]),0],
                       [0,0,1]])
        R = np.dot(Rz, np.dot(Ry,Rx))
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), R)
    return rotated_data


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def load_scanobjectnn_data(root, sonn_type, partition, num_point):
    all_data = []
    all_label = []

    if partition == 'train':
        partition = 'training'
    if sonn_type == 'obj_only':
        h5_name = f'data/ScanObjectNN/main_split_nobg/{partition}_objectdataset.h5'
    elif sonn_type == 'obj_bg':
        h5_name = f'data/ScanObjectNN/main_split/{partition}_objectdataset.h5'
    elif sonn_type == 'hardest':
        h5_name = f'data/ScanObjectNN/main_split/{partition}_objectdataset_augmentedrot_scale75.h5'

    if num_point == 1024:
        h5_name = h5_name.replace('.h5', '_1024.h5')

    f = h5py.File(h5_name, mode="r")
    data = f['data'][:].astype('float32')
    label = f['label'][:].astype('int64')
    f.close()
    all_data.append(data)
    all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def read_mn_so_data(classnames, points, labels):
    items = []

    for i, pc in enumerate(points):
        label = int(labels[i])
        classname = classnames[label]

        item = {'pc': pc, 'label': label, 'classname': classname}
        items.append(item)
    
    return items


def generate_fewshot_dataset(data_source, num_shots=-1, repeat=True):
    if num_shots < 1:
        return data_source

    tracker = split_dataset_by_label(data_source)
    fewshot_dataset = []

    for items in tracker.values():
        if len(items) >= num_shots:
            sampled_items = random.sample(items, num_shots)
        else:
            if repeat:
                sampled_items = random.choices(items, k=num_shots)
            else:
                sampled_items = items
        fewshot_dataset.extend(sampled_items)

    return fewshot_dataset


def split_dataset_by_label(data_source):
    """Split a dataset, i.e. a list of Datum objects,
    into class-specific groups stored in a dictionary.

    Args:
        data_source (list): a list of Datum objects.
    """
    output = defaultdict(list)

    for item in data_source:
        label = item['label']
        output[label].append(item)

    return output


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
PROJ_DIR = os.path.dirname(BASE_DIR)


@DATASETS.register_module()
class ModelNet(data.Dataset):
    def __init__(self, config):
        self.root = config.DATA_PATH
        self.npoints = config.npoints
        self.num_category = config.NUM_CATEGORY
        self.subset = config.split
        assert (self.subset == 'train' or self.subset == 'test')
        self.num_learnable_prompt_tokens = config.num_learnable_prompt_tokens

        self.catfile = os.path.join(self.root, f'modelnet{self.num_category}_shape_names.txt')
        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        self.save_path = os.path.join(self.root, 'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, self.subset, self.npoints))
        print_log('Load processed data from %s...' % self.save_path, logger='ModelNet')

        with open(self.save_path, 'rb') as f:
            self.list_of_points, self.list_of_labels = pickle.load(f)

    def __len__(self):
        return len(self.list_of_labels)

    def _get_item(self, index):
        points, label = self.list_of_points[index], self.list_of_labels[index]

        if self.npoints < points.shape[0]:
            points = farthest_point_sample(points, self.npoints)

        points = points[:, 0:3]
        points[:, 0:3] = pc_normalize(points[:, 0:3])

        label = int(label)
        return points, label

    def __getitem__(self, index):
        pointcloud, label = self._get_item(index)

        if self.subset == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)

        label_name = self.cat[label]
        return pointcloud, label, label_name


@DATASETS.register_module()
class ModelNet_fs(ModelNet):
    def __init__(self, config):
        self.root = config.DATA_PATH
        self.npoints = config.npoints
        self.num_category = config.NUM_CATEGORY
        self.subset = config.split
        assert (self.subset == 'train' or self.subset == 'test')
        num_shots = config.nshots

        self.catfile = os.path.join(self.root, f'modelnet{self.num_category}_shape_names.txt')
        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        self.save_path = os.path.join(self.root, 'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, self.subset, self.npoints))
        print_log('Load processed data from %s...' % self.save_path, logger='ModelNet')

        with open(self.save_path, 'rb') as f:
            self.list_of_points, self.list_of_labels = pickle.load(f)

        if self.subset == 'train':
            train = read_mn_so_data(self.cat, self.list_of_points, self.list_of_labels)
            few_path = os.path.join("data/fewshot", f"modelnet_{num_shots}s_seed{config.args.seed}.pkl")
            if os.path.exists(few_path):
                print_log(f"Loading preprocessed few-shot data from {few_path}", logger='ModelNet40_fs')
                with open(few_path, "rb") as file:
                    self.data_source = pickle.load(file)
            else:
                print_log(f"Generating few-shot data to {few_path}", logger='ModelNet40_fs')
                self.data_source = generate_fewshot_dataset(train, num_shots=num_shots)
                with open(few_path, "wb") as file:
                    pickle.dump(self.data_source, file)
        else:
            self.data_source = read_mn_so_data(self.cat, self.list_of_points, self.list_of_labels)

    def __len__(self):
        return len(self.data_source)

    def _get_item(self, index):
        item = self.data_source[index]
        points, label, label_name = item['pc'], item['label'], item['classname']

        if self.npoints < points.shape[0]:
            points = farthest_point_sample(points, self.npoints)

        points = points[:, 0:3]
        points[:, 0:3] = pc_normalize(points[:, 0:3])

        return points, label, label_name

    def __getitem__(self, index):
        pointcloud, label, label_name = self._get_item(index)

        if self.subset == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)

        return pointcloud, label, label_name


@DATASETS.register_module()
class ScanObjectNN(data.Dataset):
    def __init__(self, config):
        self.root = config.DATA_PATH
        self.sonn_type = config.sonn_type
        self.partition = config.split
        self.data, self.label = load_scanobjectnn_data(self.root, self.sonn_type, self.partition, config.npoints)
        self.num_points = config.npoints

        self.shape_names_addr = os.path.join('data/ScanObjectNN', 'shape_names.txt')
        with open(self.shape_names_addr) as file:
            lines = file.readlines()
            self.shape_names = [line.rstrip() for line in lines]

    def __getitem__(self, item):
        pointcloud = self.data[item]
        label = self.label[item]
        label_name = self.shape_names[int(label)]
        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)

        return pointcloud, label, label_name

    def __len__(self):
        return self.data.shape[0]


@DATASETS.register_module()
class ScanObjectNN_fs(data.Dataset):
    def __init__(self, config):
        self.root = config.DATA_PATH
        self.sonn_type = config.sonn_type
        self.partition = config.split
        self.num_points = config.npoints
        num_shots = config.nshots

        shape_names_addr = os.path.join('data/ScanObjectNN', 'shape_names.txt')
        with open(shape_names_addr) as file:
            lines = file.readlines()
            shape_names = [line.rstrip() for line in lines]

        data, label = load_scanobjectnn_data(self.root, self.sonn_type, self.partition, self.num_points)
        data = read_mn_so_data(shape_names, data, label)

        if self.partition == 'train':
            print('\n============= Entering ScanObject_fs =============\n')
            few_path = os.path.join("data/fewshot", f"sonn_{self.sonn_type.replace('_', '')}_{num_shots}s_seed{config.args.seed}.pkl")
            if self.num_points == 2048:
                few_path = few_path.replace('.pkl', '_2k.pkl')

            if os.path.exists(few_path):
                print_log(f"Loading preprocessed few-shot data from {few_path}", logger='ScanObjectNN_fs')
                with open(few_path, "rb") as file:
                    self.data_source = pickle.load(file)
            else:
                print_log(f"Generating few-shot data to {few_path}", logger='ScanObjectNN_fs')
                self.data_source = generate_fewshot_dataset(data, num_shots=num_shots)
                with open(few_path, "wb") as file:
                    pickle.dump(self.data_source, file)
        else:
            self.data_source = data

    def __len__(self):
        return len(self.data_source)
    
    def __getitem__(self, idx):
        pointcloud = self.data_source[idx]['pc']
        label = self.data_source[idx]['label']
        label_name = self.data_source[idx]['classname']

        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)

        return pointcloud, label, label_name


@DATASETS.register_module()
class Objaverse_Lvis_Colored(data.Dataset):
    def __init__(self, config):
        self.npoints = config.args.npoints

        self.lvis_list_addr = 'data/objaverse-lvis/lvis.json'
        self.lvis_metadata_addr = 'data/objaverse-lvis/objaverse_lvis_metadata.json'

        with open(self.lvis_list_addr, 'r') as f:
            self.npy_file_map = json.load(f)

        with open("data/lvis_testset.txt", 'r') as f:
            lines = f.readlines()

        if config.get('whole'):
            self.npy_file_map = {a.split(',')[2]: a.split(',')[-1][1:-1] for a in lines}
        else:
            datum_dict = {i: [] for i in range(1156)}
            for line in lines:
                datum_dict[int(line.split(',')[0])].append(line)
            splited = {'train': [], 'test': []}
            for d in datum_dict.values():
                split = math.ceil(len(d) * 0.8)
                splited['train'].extend(d[:split])
                splited['test'].extend(d[split:])

            self.npy_file_map = {d.split(',')[2]: d.split(',')[-1][1:-1] for d in splited[config.subset]}

        self.file_list = list(self.npy_file_map.keys())

        with open(self.lvis_metadata_addr, 'r') as f:
            self.lvis_metadata = json.load(f)

        self.prompt_template_addr = 'data/templates.json'
        with open(self.prompt_template_addr) as f:
            self.templates = json.load(f)[config.dataset_prompt]

        self.sample_points_num = self.npoints

        print_log(f'Objaverse lvis {len(self.file_list)} instances were loaded', logger='objaverse_lvis')

        self.permutation = np.arange(self.npoints)

        # =================================================
        self.augment = False
        if self.augment:
            print("using augmented point clouds.")

        self.objaverse_lvis_path = 'data/objaverse-lvis'
        self.objaverse_lvis_path = os.path.join(self.objaverse_lvis_path, f"{self.npoints}")

    def pc_norm(self, pc):
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    def random_sample(self, pc, num):
        np.random.shuffle(self.permutation)
        pc = pc[self.permutation[:num]]
        return pc

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        pc_addr = os.path.join(self.objaverse_lvis_path, self.npy_file_map[sample])
        data = np.load(pc_addr, allow_pickle=True)
        dict_data = data.item()
        xyz_data = dict_data['xyz']
        # rgb_data = dict_data['rgb']

        data = self.pc_norm(xyz_data)

        if len(self.file_list) > 30000:
            np.random.shuffle(data)
        if self.augment:
            data = random_point_dropout(data[None, ...])
            data = random_scale_point_cloud(data)
            data = shift_point_cloud(data)
            data = rotate_perturbation_point_cloud(data)
            data = rotate_point_cloud(data)
            data = data.squeeze()
        else:
            data = translate_pointcloud(data)

        data = torch.from_numpy(data).float()
        data = data.contiguous()

        name = self.lvis_metadata["value_to_key_mapping"][sample]
        label = self.lvis_metadata["key_to_id"][name]

        return data, label, name

    def __len__(self):
        return len(self.file_list)


import collections.abc as container_abcs
int_classes = int
from torch._six import string_classes

import re
default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")
np_str_obj_array_pattern = re.compile(r'[SaUO]')


def merge_new_config(config, new_config):
    for key, val in new_config.items():
        if not isinstance(val, dict):
            if key == '_base_':
                with open(new_config['_base_'], 'r') as f:
                    try:
                        val = yaml.load(f, Loader=yaml.FullLoader)
                    except:
                        val = yaml.load(f)
                config[key] = EasyDict()
                merge_new_config(config[key], val)
            else:
                config[key] = val
                continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)
    return config


def cfg_from_yaml_file(cfg_file):
    # e.g., cfg_file -> ./data/ShapeNet-55.yaml
    config = EasyDict()
    with open(cfg_file, 'r') as f:
        new_config = yaml.load(f, Loader=yaml.FullLoader)
    merge_new_config(config=config, new_config=new_config)
    return config


class Dataset_3D():
    def __init__(self, args, tokenizer, dataset_type, train_transform=None):
        if dataset_type not in ['train', 'val', 'test']:
            raise ValueError("not supported dataset type.")
        
        self.dataset_name = args.dataset_name
        self.tokenizer = tokenizer
        self.train_transform = train_transform
        self.dataset_prompt = args.dataset_prompt
        self.dataset_type = dataset_type

        with open(os.path.join(PROJ_DIR, 'data/dataset_catalog.json'), 'r') as f:
            self.dataset_catalog = json.load(f)
            self.dataset_config_dir = self.dataset_catalog[self.dataset_name]['config']
        self.build_3d_dataset(args, self.dataset_config_dir)

    def build_3d_dataset(self, args, dataset_config_dir):
        config = cfg_from_yaml_file(dataset_config_dir)
        config.sonn_type = args.sonn_type
        config.tokenizer = self.tokenizer
        config.train_transform = self.train_transform
        config.dataset_prompt = self.dataset_prompt
        config.split = self.dataset_type
        config.args = args
        config.use_height = args.use_height
        config.npoints = args.npoints
        config.nshots = args.nshots
        config.template_init = args.template_init
        config.num_learnable_prompt_tokens = args.num_learnable_prompt_tokens
        config.subsample = args.dataset_subsample
        config_others = EasyDict({'subset': config.split, 'whole': False})
        self.dataset = build_dataset_from_cfg(config, config_others)
