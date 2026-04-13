from collections import OrderedDict
import os
import math
import time
import wandb

import torch
import torch.nn.parallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.nn import CrossEntropyLoss
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import collections

from data.dataset_3d import *

from utils.utils import set_random_seed, get_dataset, accuracy, AverageMeter, ProgressMeter
import models.ULIP_models as models
from utils.tokenizer import SimpleTokenizer
from utils import utils
import torch.nn.functional as F


def main(args):
    utils.init_distributed_mode(args)

    if utils.is_main_process() and args.wandb:
        wandb.init(project='P3T', name=args.exp_name.replace('sonn_', '').replace('-32v-middle', ''))
        wandb.config.update(args)

    print(args)

    seed = args.seed + utils.get_rank()
    set_random_seed(seed)

    print("=> creating model: {}".format(args.model))
    model = getattr(models, args.model)(args)
    model.cuda(args.gpu)

    if args.distributed:
        model = DDP(model, device_ids=[args.gpu], find_unused_parameters=False)

    criterion = CrossEntropyLoss(label_smoothing=args.label_smoothing).cuda(args.gpu)

    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=args.betas,
                                    eps=args.eps, weight_decay=args.wd)

    cudnn.benchmark = True

    print(f"=> creating a {args.nshots}-Shot dataset")
    tokenizer = SimpleTokenizer()

    train_dataset = get_dataset(None, tokenizer, args, 'train')
    val_dataset = get_dataset(None, tokenizer, args, 'test')
    print(f'------ len(train_dataset) [{args.dataset_name}]:', len(train_dataset))
    print(f'------ len(val_dataset) [{args.dataset_name}]:', len(val_dataset))

    if args.distributed:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=False)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=False)

    lr_scheduler = utils.cosine_scheduler(args.lr, args.lr_end, args.epochs,
        len(train_loader) // args.update_freq, warmup_epochs=args.warmup_epochs, start_warmup_value=args.lr_start)

    proto_data = f"sonn_{args.sonn_type.replace('_', '')}" if 'scanobjectnn' in args.dataset_name \
        else 'lvis' if args.dataset_name == 'objaverse_lvis_colored' else args.dataset_name
    proto_path = f"data/prototype/{proto_data.replace('_fs', '')}-{args.nshots}s{args.seed}-ULIP2-PointBERT.pt"
    prototype = torch.load(proto_path, map_location=torch.device('cpu'))
    print(f"------ Loaded prototype from {proto_path}")
    prototype_mean = {}
    for k in prototype.keys():
        feat = torch.stack(prototype[k], dim=0).mean(0, keepdim=True)
        prototype_mean[k] = feat

    with open(os.path.join("./data", 'templates.json')) as f:
        templates = json.load(f)[args.dataset_prompt]
    classnames = [name.replace("_", " ") for name in args.classnames]
    if args.dataset_subsample != 'all':
        classnames = classnames[:int(np.ceil(len(classnames)/2))]
    with torch.no_grad():
        fixed_embeddings = []
        for l in classnames:
            fixed_prompts = [t.format(l) for t in templates]
            fixed_tokenized_prompts = tokenizer(fixed_prompts).cuda(args.gpu, non_blocking=True)
            fixed_features = utils.get_model(model).encode_text_ulip(fixed_tokenized_prompts)
            # fixed_features = model.encode_text_ulip(fixed_tokenized_prompts)
            fixed_features = fixed_features / fixed_features.norm(dim=-1, keepdim=True)
            fixed_features = fixed_features.mean(dim=0)
            fixed_features = fixed_features / fixed_features.norm(dim=-1, keepdim=True)
            fixed_embeddings.append(fixed_features)
        fixed_embeddings = torch.stack(fixed_embeddings, dim=0)

    print("=> beginning finetuning")

    best_acc1 = 0
    best_epoch = -1
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_stats = train(train_loader, model, prototype_mean, fixed_embeddings, criterion, optimizer, epoch, lr_scheduler, args)

        val_stats = validate(val_loader, model, criterion, args)
        acc1 = val_stats["acc1"]
        print(val_stats)

        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if args.distributed:
            state_dict_prompt = model.module.prompt_learner.state_dict()
        else:
            state_dict_prompt = model.prompt_learner.state_dict()

        if is_best:
            best_epoch = epoch
            print("=> saving best checkpoint")
            saved_data = {'epoch': epoch + 1, 'state_dict': state_dict_prompt,
                          'optimizer': optimizer.state_dict(),
                          'best_acc1': best_acc1,  'args': args,
                          'all_state_dict': model.state_dict()}
            utils.save_on_master(saved_data, is_best, os.path.join(args.output_dir, args.proj_name, args.exp_name))

        if (epoch + 1) % 50 == 0 or epoch + 1 == args.epochs:
            e = 'last' if epoch + 1 == args.epochs else epoch + 1
            print(f"=> saving {e} checkpoint")
            saved_data = {'epoch': e, 'state_dict': state_dict_prompt,
                          'optimizer': optimizer.state_dict(),
                          'best_acc1': best_acc1, 'args': args,
                          'all_state_dict': model.state_dict()}
            if utils.get_rank() == 0:
                torch.save(saved_data, f'{os.path.join(args.output_dir, args.proj_name, args.exp_name)}/checkpoint_{e}.pt')

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in val_stats.items()},
                     'epoch': epoch,
                     'best_acc1': best_acc1,
                     'best_epoch': best_epoch}

        if utils.is_main_process():
            if args.wandb:
                wandb.log(log_stats)

    if utils.is_main_process():
        if args.wandb:
            wandb.finish()


def train(train_loader, model, prototype, fixed_embeddings, criterion, optimizer, epoch, lr_scheduler, args):
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.2f')
    mem = AverageMeter('Mem (GB)', ':6.1f')
    metric_names = ['loss', 'acc', 'loss_proto', 'loss_size', 'loss_center', 'loss_con']
    iters_per_epoch = len(train_loader) // args.update_freq
    metrics = OrderedDict([(name, AverageMeter(name, ':.2e')) for name in metric_names])
    progress = ProgressMeter(
        iters_per_epoch,
        [batch_time, data_time, mem, *metrics.values()],
        prefix="Epoch: [{}]".format(epoch))

    model.train()

    end = time.time()
    for data_iter, inputs in enumerate(train_loader):
        data_time.update(time.time() - end)

        optimizer.zero_grad()
        optim_iter = data_iter // args.update_freq

        it = iters_per_epoch * epoch + optim_iter
        for _, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = lr_scheduler[it]

        pc = inputs[0]
        pc = pc.to(args.gpu)
        label = inputs[1].long().to(args.gpu)

        pred, prompt_dict = model(pc)

        target_name = inputs[2]
        proto = torch.cat([prototype[name] for name in target_name])
        pc_embed, text_embed = prompt_dict['pc_embed'], prompt_dict['text_embed']

        if args.con_type == 'mse':
            loss_proto = F.mse_loss(pc_embed, proto.to(pc_embed.device))
            loss_con = F.mse_loss(text_embed, fixed_embeddings)
        elif args.con_type == 'l1':
            loss_proto = F.l1_loss(pc_embed, proto.to(pc_embed.device), reduction='mean')
            loss_con = F.l1_loss(text_embed, fixed_embeddings, reduction='mean')
        elif args.con_type == 'cos':
            cos = torch.nn.CosineSimilarity(dim=1, eps=1e-7)
            pc_score = cos(pc_embed, proto.to(pc_embed.device))
            text_score = cos(text_embed, fixed_embeddings)
            loss_proto = 1.0 - torch.mean(pc_score)
            loss_con = 1.0 - torch.mean(text_score)

        neighbor, center = prompt_dict['neighbor'], prompt_dict['center']
        neighbor_prom, center_prom = prompt_dict['prompt_neighbor'], prompt_dict['prompt_center']

        dist_patch, dist_patch_prom = torch.cdist(neighbor, neighbor), torch.cdist(neighbor_prom, neighbor_prom)    # (b, g, m, m)
        size_prom = torch.max(dist_patch_prom.view(dist_patch_prom.shape[0], dist_patch_prom.shape[1], -1), dim=-1).values
        G = torch.max(dist_patch.view(dist_patch.shape[0], -1), dim=-1)[0]      # (b,)
        loss_size = (size_prom - G.unsqueeze(1)).clamp(min=0).mean()

        mu_pc, mu_ori, mu_prom = torch.mean(pc, dim=1), center, center_prom
        dist_center, dist_center_prom = (mu_ori - mu_pc.unsqueeze(1)).norm(dim=-1), (mu_prom - mu_pc.unsqueeze(1)).norm(dim=-1)
        H = torch.max(dist_center, dim=-1).values    # (b,)
        loss_center = (dist_center_prom - H.unsqueeze(1)).clamp(min=0).mean()

        loss_prompt = args.w_proto * loss_proto + args.w_size * loss_size + args.w_center * loss_center + args.w_con * loss_con

        loss = criterion(pred, label)
        loss_total = loss + loss_prompt

        loss_total.backward()
        optimizer.step()

        pred_idx = pred.argmax(dim=1)
        correct = torch.eq(pred_idx, label).sum()
        acc = correct / len(label)

        if not math.isfinite(loss_total.item()):
            print("Loss is {}, stopping training".format(loss_total.item()))
            sys.exit(1)

        if (data_iter + 1) % args.update_freq != 0:
            continue

        utils.get_model(model).logit_scale.data.clamp_(0, 4.6052)
        logit_scale = utils.get_model(model).logit_scale.exp().item()

        metrics['loss'].update(loss.item())
        metrics['acc'].update(acc.item())
        metrics['loss_proto'].update(loss_proto.item())
        metrics['loss_size'].update(loss_size.item())
        metrics['loss_center'].update(loss_center.item())
        metrics['loss_con'].update(loss_con.item())

        batch_time.update(time.time() - end)
        end = time.time()

        mem.update(torch.cuda.max_memory_allocated() // 1e9)

        if optim_iter % args.print_freq == 0:
            progress.display(optim_iter)

    progress.synchronize()
    return {**{k: v.avg for k, v in metrics.items()},
            'lr': optimizer.param_groups[0]['lr'],
            'logit_scale': logit_scale}


def validate(test_loader, model, criterion, args):
    print('------ Evaluation')
    batch_time = AverageMeter('Time', ':6.3f')
    val_top1 = AverageMeter('Acc@1', ':6.2f')
    val_top5 = AverageMeter('Acc@5', ':6.2f')
    val_loss = AverageMeter('Loss', ':6.2f')
    progress = ProgressMeter(
        len(test_loader),
        [batch_time, val_top1, val_top5, val_loss],
        prefix='Test: ')

    model.eval()

    with torch.no_grad():
        end = time.time()
        per_class_stats = collections.defaultdict(int)
        per_class_correct_top1 = collections.defaultdict(int)
        per_class_correct_top5 = collections.defaultdict(int)

        for i, inputs in enumerate(test_loader):
            pc = inputs[0]
            target = inputs[1]
            target_name = inputs[2]

            for name in target_name:
                per_class_stats[name] += 1

            pc = pc.cuda(args.gpu)
            target = target.long().cuda(args.gpu)

            pred, _ = model(pc)
            loss = criterion(pred, target)

            (acc1, acc5), correct = accuracy(pred, target, topk=(1, 5))

            val_top1.update(acc1.item(), pc.size(0))
            val_top5.update(acc5.item(), pc.size(0))
            val_loss.update(loss.item(), pc.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            top1_accurate = correct[:1].squeeze(0)
            top5_accurate = correct[:5].float().sum(0, keepdim=True).squeeze(0)
            for idx, name in enumerate(target_name):
                if top1_accurate[idx].item():
                    per_class_correct_top1[name] += 1
                if top5_accurate[idx].item():
                    per_class_correct_top5[name] += 1

            if i % args.print_freq == 0:
                progress.display(i)

        top1_accuracy_per_class = {}
        top5_accuracy_per_class = {}
        for name in per_class_stats.keys():
            top1_accuracy_per_class[name] = per_class_correct_top1[name] / per_class_stats[name]
            top5_accuracy_per_class[name] = per_class_correct_top5[name] / per_class_stats[name]

        top1_accuracy_per_class = collections.OrderedDict(top1_accuracy_per_class)
        top5_accuracy_per_class = collections.OrderedDict(top5_accuracy_per_class)
        print(','.join(top1_accuracy_per_class.keys()))
        print(','.join([str(value) for value in top1_accuracy_per_class.values()]))
        print(','.join([str(value) for value in top5_accuracy_per_class.values()]))

    progress.synchronize()
    print(f'Test * Acc@1 {val_top1.avg:.3f} Acc@5 {val_top5.avg:.3f} Loss {val_loss.avg:.3f}')
    return {'acc1': val_top1.avg, 'acc5': val_top5.avg, 'loss': val_loss.avg}


if __name__ == '__main__':
    from parser import args

    main(args)
