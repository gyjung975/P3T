from collections import OrderedDict
import os
import math
import time

import torch
import torch.nn.parallel
from torch.utils.data.distributed import DistributedSampler
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import collections

from data.dataset_3d import *

from utils.utils import set_random_seed, get_dataset, accuracy, AverageMeter, ProgressMeter
import models.ULIP_models as models
from utils.tokenizer import SimpleTokenizer
from utils import utils
from tqdm import tqdm


def main(args):
    ckpt = torch.load(args.test_ckpt_addr, map_location='cpu')
    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v

    print("=> creating model: {}".format(args.model))

    model = getattr(models, args.model)(args=args)
    model.cuda()
    model.load_state_dict(state_dict, strict=False)
    print("=> loaded pretrained checkpoint '{}'".format(args.test_ckpt_addr))

    tokenizer = SimpleTokenizer()

    test_dataset = get_dataset(None, tokenizer, args, 'train')
    # test_dataset = get_dataset(None, tokenizer, args, 'test')

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True, sampler=None, drop_last=False
    )

    test_zeroshot_3d_core(test_loader, model, args)


@torch.no_grad()
def test_zeroshot_3d_core(test_loader, model, args):
    model.eval()
    per_class_feat, per_class_feat_norm = collections.defaultdict(list), collections.defaultdict(list)
    for i, (pc, target, target_name) in enumerate(tqdm(test_loader)):
        pc = pc.cuda(args.gpu, non_blocking=True)

        pc_features = utils.get_model(model).encode_pc(pc)
        pc_features = pc_features / pc_features.norm(dim=-1, keepdim=True)
        for j, name in enumerate(target_name):
            per_class_feat_norm[name].append(pc_features[j])

    prefix = f'sonn_{args.sonn_type.replace("_", "")}' if 'scanobjectnn' in args.dataset_name \
        else 'lvis' if 'objaverse_lvis_colored' in args.dataset_name else args.dataset_name
    proto_path = f'data/prototype/{prefix}-ULIP2-PointBERT.pt'
    if not os.path.exists(proto_path):
        torch.save(per_class_feat_norm, proto_path)


if __name__ == '__main__':
    from parser import args

    main(args)
