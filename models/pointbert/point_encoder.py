import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_

from models.pointbert.dvae import Group
from models.pointbert.dvae import Encoder
from models.pointbert.logger import print_log
from models.pointbert.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message

from knn_cuda import KNN
from models.backbone import feature_extractor
from data.dataset_3d import cfg_from_yaml_file
import random


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    """ Transformer Encoder without hierarchical structure
    """

    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])

    def forward(self, x, pos):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        return x


class PointTransformer_RAW(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.args = kwargs["args"]

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.drop_path_rate = config.drop_path_rate
        self.cls_dim = config.cls_dim
        self.num_heads = config.num_heads

        self.group_size = config.group_size
        self.num_group = config.num_group

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        self.encoder_dims = config.encoder_dims
        self.encoder = Encoder(encoder_channel=self.encoder_dims)
        self.reduce_dim = nn.Linear(self.encoder_dims, self.trans_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads
        )

        self.norm = nn.LayerNorm(self.trans_dim)
        if not self.args.evaluate_3d:
            self.load_model_from_ckpt("../NAS/ULIP/data/initialize_models/point_bert_pretrained.pt")

    def load_model_from_ckpt(self, bert_ckpt_path):
        ckpt = torch.load(bert_ckpt_path)
        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
        for k in list(base_ckpt.keys()):
            if k.startswith('transformer_q') and not k.startswith('transformer_q.cls_head'):
                base_ckpt[k[len('transformer_q.'):]] = base_ckpt[k]
            elif k.startswith('base_model'):
                base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
            del base_ckpt[k]

        incompatible = self.load_state_dict(base_ckpt, strict=False)

        if incompatible.missing_keys:
            print_log('missing_keys', logger='Transformer')
            print_log(
                get_missing_parameters_message(incompatible.missing_keys),
                logger='Transformer'
            )
        if incompatible.unexpected_keys:
            print_log('unexpected_keys', logger='Transformer')
            print_log(
                get_unexpected_parameters_message(incompatible.unexpected_keys),
                logger='Transformer'
            )

        print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')

    def forward(self, pts):
        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)
        group_input_tokens = self.reduce_dim(group_input_tokens)

        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)
        pos = self.pos_embed(center)

        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)

        x = self.blocks(x, pos)
        x = self.norm(x)

        concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)
        return concat_f


class Prompter(nn.Module):
    def __init__(self, num_group, group_size, trans_dim, num_heads, dpr, args):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.trans_dim = trans_dim
        self.num_heads = num_heads

        backbone_config_addr = f'models/backbone/PointBERT_{args.npoints}point.yaml'
        backbone_config = cfg_from_yaml_file(backbone_config_addr)
        self.feature_extractor = getattr(feature_extractor, "ULIP_PointBERT")(backbone_config, args=args)

        class PromptArgs:
            def __init__(self):
                self.k = 20
                self.emb_dims = 1024
                self.dropout = 0.5
                self.leaky_relu = 1
        prompt_args = PromptArgs()
        self.prompt_encoder = DGCNNView(prompt_args, self.trans_dim)

        self.num_prompt = round(self.num_group * args.prompt_ratio)
        self.k = round(self.num_group * 0.3)
        self.knn = KNN(k=self.k, transpose_mode=True)

        class OffsetArgs:
            def __init__(self):
                self.k = 20
                self.emb_dims = 1024
                self.dropout = 0.5
                self.leaky_relu = 1
        offset_args = OffsetArgs()
        self.offset_generator = DGCNNViewLight(offset_args, self.trans_dim)

        self.center_offset = nn.Sequential(
            nn.Conv1d(self.trans_dim, 128, 1),
            nn.GELU(),
            nn.Conv1d(128, 3, 1)
        )
        self.group_offset = nn.Sequential(
            nn.Conv1d(self.trans_dim, 256, 1),
            nn.GELU(),
            nn.Conv1d(256, 256, 1),
            nn.GELU(),
            nn.Conv1d(256, 3 * self.group_size, 1)
        )

        for l in self.center_offset:
            if isinstance(l, nn.Conv1d) and l.out_channels == 3:
                nn.init.zeros_(l.weight)
                nn.init.zeros_(l.bias)
        for l in self.group_offset:
            if isinstance(l, nn.Conv1d) and l.out_channels == 3*self.group_size:
                nn.init.zeros_(l.weight)
                nn.init.zeros_(l.bias)

    def forward(self, neighborhood, center):
        batch_size = center.size(0)

        with torch.no_grad():
            feat = self.feature_extractor(neighborhood, center).detach()

        pos = None
        prompt_token, prompt_feat = self.prompt_encoder(feat.transpose(1, 2), pos)

        global_feat, max_indices = torch.max(feat, dim=1)
        vul_mask = torch.zeros(batch_size, self.num_group, dtype=torch.bool, device=center.device)
        indices = set(range(self.num_group))

        for b, v in enumerate(max_indices):
            unique, count = torch.unique(v, return_counts=True)
            vul_idx = list(indices - set(unique.tolist()))
            random.shuffle(vul_idx)
            vul_idx = vul_idx[:self.num_prompt]
            if len(vul_idx) < self.num_prompt:
                extra = torch.argsort(count)[:self.num_prompt - len(vul_idx)]
                extra_idx = unique[extra].tolist()
                vul_idx.extend(extra_idx)
            vul_mask[b, vul_idx] = True

        vul_center = center[vul_mask].view(batch_size, self.num_prompt, 3)
        vul_neighbor = neighborhood[vul_mask].view(batch_size, self.num_prompt, self.group_size, 3)

        _, idx = self.knn(center, vul_center)
        idx_base = torch.arange(0, batch_size, device=center.device).view(-1, 1, 1) * self.num_group
        idx = idx + idx_base
        idx = idx.view(-1)

        feature = prompt_feat.contiguous().view(-1, self.trans_dim)[idx, :]
        feature = feature.view(batch_size, self.num_prompt, self.k, self.trans_dim)
        offset_feat = torch.max(feature, dim=2)[0]

        x_full = torch.cat([prompt_feat, offset_feat], dim=1)
        _, offset_feat = self.offset_generator(x_full.transpose(1, 2), None)
        offset_feat = offset_feat[:, -self.num_prompt:]

        prompt_center_offset = self.center_offset(offset_feat.transpose(1, 2)).transpose(1, 2)
        prompt_group_offset = self.group_offset(offset_feat.transpose(1, 2)).transpose(1, 2).view(batch_size, -1, self.group_size, 3)

        prompt_center = vul_center + prompt_center_offset
        prompt_neighbor = vul_neighbor + prompt_group_offset

        prompt_dict = {'prompt_token': prompt_token,
                       'prompt_neighbor': prompt_neighbor, 'prompt_center': prompt_center,
                       'neighbor': neighborhood, 'center': center}
        return prompt_dict


class PointTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.args = kwargs["args"]

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.drop_path_rate = config.drop_path_rate
        self.cls_dim = config.cls_dim
        self.num_heads = config.num_heads

        self.group_size = config.group_size
        self.num_group = config.num_group

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        self.encoder_dims = config.encoder_dims
        self.encoder = Encoder(encoder_channel=self.encoder_dims)
        self.reduce_dim = nn.Linear(self.encoder_dims, self.trans_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        self.prompter = Prompter(num_group=self.num_group, group_size=self.group_size, trans_dim=self.trans_dim,
                                 num_heads=self.num_heads, dpr=dpr, args=self.args)

        self.prompt_token_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))
        trunc_normal_(self.prompt_token_pos, std=.02)

    def load_model_from_ckpt(self, bert_ckpt_path):
        ckpt = torch.load(bert_ckpt_path, map_location=torch.device('cpu'))
        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
        for k in list(base_ckpt.keys()):
            if k.startswith('transformer_q') and not k.startswith('transformer_q.cls_head'):
                base_ckpt[k[len('transformer_q.'):]] = base_ckpt[k]
            elif k.startswith('base_model'):
                base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
            del base_ckpt[k]

        incompatible = self.load_state_dict(base_ckpt, strict=False)

        if incompatible.missing_keys:
            print_log('missing_keys', logger='Transformer')
            print_log(
                get_missing_parameters_message(incompatible.missing_keys),
                logger='Transformer'
            )
        if incompatible.unexpected_keys:
            print_log('unexpected_keys', logger='Transformer')
            print_log(
                get_unexpected_parameters_message(incompatible.unexpected_keys),
                logger='Transformer'
            )

        print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')

    def forward(self, pts):
        neighborhood, center = self.group_divider(pts)

        prompt_dict = self.prompter(neighborhood, center)
        prompt_neighbor, prompt_center = prompt_dict['prompt_neighbor'], prompt_dict['prompt_center']

        neighborhood = torch.cat([neighborhood, prompt_neighbor], dim=1)
        center = torch.cat([center, prompt_center], dim=1)

        group_input_tokens = self.encoder(neighborhood)
        group_input_tokens = self.reduce_dim(group_input_tokens)

        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)
        prompt_token_pos = self.prompt_token_pos.expand(group_input_tokens.size(0), -1, -1)
        pos = self.pos_embed(center)

        prompt_token = prompt_dict['prompt_token']
        x = torch.cat((cls_tokens, prompt_token, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, prompt_token_pos, pos), dim=1)

        x = self.blocks(x, pos)
        x = self.norm(x)

        concat_f = torch.cat([x[:, 0], x[:, 2:].max(1)[0]], dim=-1)
        return concat_f, prompt_dict


class TransFormerLayerView(nn.Module):
    def __init__(self, trans_dim=768, depth=4, num_heads=12, dpr=0.,):
        super(TransFormerLayerView, self).__init__()

        self.blocks = TransformerEncoder(
            embed_dim=trans_dim,
            depth=depth,
            drop_path_rate=dpr,
            num_heads=num_heads,
        )

    def forward(self, x, pos):
        x = self.blocks(x.transpose(1, 2), pos)

        global_feature = torch.max(x, dim=1, keepdim=True)[0]

        return global_feature, x


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)       # (bs', #group, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature


class DGCNNView(nn.Module):
    def __init__(self, args, dim=512):
        super(DGCNNView, self).__init__()
        self.args = args
        self.k = args.k
        self.leaky_relu = bool(args.leaky_relu)
        self.dim = dim
        self.bn1 = nn.BatchNorm2d(self.dim)
        self.bn2 = nn.BatchNorm2d(self.dim)
        self.bn3 = nn.BatchNorm2d(self.dim)
        self.bn5 = nn.BatchNorm1d(self.dim)

        if self.leaky_relu:
            act_mod = nn.LeakyReLU
            act_mod_args = {'negative_slope': 0.2}
        else:
            act_mod = nn.ReLU
            act_mod_args = {}

        self.conv1 = nn.Sequential(nn.Conv2d(self.dim * 2, self.dim, kernel_size=1, bias=False),
                                   self.bn1,
                                   act_mod(**act_mod_args))
        self.conv2 = nn.Sequential(nn.Conv2d(self.dim * 2, self.dim, kernel_size=1, bias=False),
                                   self.bn2,
                                   act_mod(**act_mod_args))
        self.conv3 = nn.Sequential(nn.Conv2d(self.dim * 2, self.dim, kernel_size=1, bias=False),
                                   self.bn3,
                                   act_mod(**act_mod_args))
        self.conv5 = nn.Sequential(nn.Conv1d(self.dim * 3, self.dim, kernel_size=1, bias=False),
                                   self.bn5,
                                   act_mod(**act_mod_args))

        self.knn = KNN(k=self.k, transpose_mode=True)

    def forward(self, x, pos):
        batch_size = x.size(0)
        # x = x + pos.transpose(1, 2)
        _, idx = self.knn(x.transpose(1, 2), x.transpose(1, 2))
        x = get_graph_feature(x, k=self.k, idx=idx)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1).unsqueeze(1)

        return x1, x.transpose(1, 2)


class DGCNNViewLight(nn.Module):
    def __init__(self, args, dim=512):
        super(DGCNNViewLight, self).__init__()
        self.args = args
        self.k = args.k
        self.leaky_relu = bool(args.leaky_relu)
        self.dim = dim
        self.bn1 = nn.BatchNorm2d(self.dim)
        self.bn5 = nn.BatchNorm1d(self.dim)

        if self.leaky_relu:
            act_mod = nn.LeakyReLU
            act_mod_args = {'negative_slope': 0.2}
        else:
            act_mod = nn.ReLU
            act_mod_args = {}

        self.conv1 = nn.Sequential(nn.Conv2d(self.dim*2, self.dim, kernel_size=1, bias=False),
                                   self.bn1,
                                   act_mod(**act_mod_args))

        self.conv5 = nn.Sequential(nn.Conv1d(self.dim, self.dim, kernel_size=1, bias=False),
                                   self.bn5,
                                   act_mod(**act_mod_args))

        self.knn = KNN(k=self.k, transpose_mode=True)

    def forward(self, x, pos):
        batch_size = x.size(0)
        # x = x + pos.transpose(1, 2)
        _, idx = self.knn(x.transpose(1, 2), x.transpose(1, 2))
        x = get_graph_feature(x, k=self.k, idx=idx)
        x = self.conv1(x)
        x = x.max(dim=-1, keepdim=False)[0]

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1).unsqueeze(1)

        return x1, x.transpose(1, 2)


class DGCNNViewLight2(nn.Module):
    def __init__(self, args, dim=512):
        super(DGCNNViewLight2, self).__init__()
        self.args = args
        self.k = args.k
        self.leaky_relu = bool(args.leaky_relu)
        self.dim = dim
        self.bn1 = nn.BatchNorm2d(self.dim)
        self.bn2 = nn.BatchNorm2d(self.dim)
        self.bn5 = nn.BatchNorm1d(self.dim)

        if self.leaky_relu:
            act_mod = nn.LeakyReLU
            act_mod_args = {'negative_slope': 0.2}
        else:
            act_mod = nn.ReLU
            act_mod_args = {}

        self.conv1 = nn.Sequential(nn.Conv2d(self.dim*2, self.dim, kernel_size=1, bias=False),
                                   self.bn1,
                                   act_mod(**act_mod_args))
        self.conv2 = nn.Sequential(nn.Conv2d(self.dim * 2, self.dim, kernel_size=1, bias=False),
                                   self.bn2,
                                   act_mod(**act_mod_args))
        self.conv5 = nn.Sequential(nn.Conv1d(self.dim * 2, self.dim, kernel_size=1, bias=False),
                                   self.bn5,
                                   act_mod(**act_mod_args))

        self.knn = KNN(k=self.k, transpose_mode=True)

    def forward(self, x, pos):
        batch_size = x.size(0)
        # x = x + pos.transpose(1, 2)
        _, idx = self.knn(x.transpose(1, 2), x.transpose(1, 2))
        x = get_graph_feature(x, k=self.k, idx=idx)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]


        x = torch.cat((x1, x2), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1).unsqueeze(1)

        return x1, x.transpose(1, 2)


class DGCNNViewMLP(nn.Module):
    def __init__(self, args, dim=512):
        super(DGCNNViewMLP, self).__init__()
        self.args = args
        self.k = args.k
        self.leaky_relu = bool(args.leaky_relu)
        self.dim = dim
        self.bn5 = nn.BatchNorm1d(self.dim)

        if self.leaky_relu:
            act_mod = nn.LeakyReLU
            act_mod_args = {'negative_slope': 0.2}
        else:
            act_mod = nn.ReLU
            act_mod_args = {}

        self.conv5 = nn.Sequential(nn.Conv1d(self.dim, self.dim, kernel_size=1, bias=False),
                                   self.bn5,
                                   act_mod(**act_mod_args))

        self.bn5 = nn.BatchNorm1d(self.dim)

    def forward(self, x, pos):
        batch_size = x.size(0)
        # x = x + pos.transpose(1, 2)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1).unsqueeze(1)

        return x1, x.transpose(1, 2)
