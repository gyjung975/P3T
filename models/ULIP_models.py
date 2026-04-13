from collections import OrderedDict

import os
import numpy as np
import torch
from torch import nn
from data.dataset_3d import *

from torch.nn.parameter import Parameter
from easydict import EasyDict

from utils.tokenizer import SimpleTokenizer


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class PromptLearner(nn.Module):
    def __init__(self, kwargs):
        super().__init__()
        self.class_name_position = kwargs.class_name_position
        self.classnames = kwargs.classnames
        self.num_learnable_prompt_tokens = kwargs.num_learnable_prompt_tokens
        self.transformer_width = kwargs.transformer_width
        self.device = kwargs.device

        self.num_learnable_prompt_tokens = kwargs.num_learnable_prompt_tokens
        prompt_prefix = " ".join(["X"] * self.num_learnable_prompt_tokens)

        self.learnable_tokens = nn.Parameter(torch.empty(self.num_learnable_prompt_tokens, self.transformer_width))

        classnames = [name.replace("_", " ") for name in self.classnames]
        tokenizer = SimpleTokenizer()
        self.name_lengths = [len(tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        self.tokenized_prompts = torch.stack([tokenizer(p) for p in prompts]).cuda(device=self.device)   # (40, 77); tokenizer(p): (77,)

    def forward(self, token_embedding):
        embedding = token_embedding(self.tokenized_prompts)

        num_classes = embedding.shape[0]
        if self.learnable_tokens.dim() == 2:
            learnable_tokens = self.learnable_tokens.unsqueeze(0).repeat(num_classes, 1, 1)

        prefix = embedding[:, :1, :]   # [SOS]
        suffix = embedding[:, 1+self.num_learnable_prompt_tokens:, :]

        if self.class_name_position == "front":
            prompts = []
            for i in range(num_classes):
                shape_name_len = self.name_lengths[i]

                prefix_i = prefix[i : i+1, :1, :]
                class_i = suffix[i : i+1, :shape_name_len, :]
                learnable_tokens_i = learnable_tokens[i : i+1, :, :]
                suffix_i = suffix[i : i+1, shape_name_len:, :]

                prompt = torch.cat([prefix_i, class_i, learnable_tokens_i, suffix_i], dim=1)
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_name_position == "middle":
            prompts = []
            half_len = self.num_learnable_prompt_tokens // 2
            for i in range(num_classes):
                shape_name_len = self.name_lengths[i]

                prefix_i = prefix[i : i+1, :, :]
                learnable_tokens_i_half1 = learnable_tokens[i : i+1, :half_len, :]
                class_i = suffix[i : i+1, :shape_name_len, :]
                learnable_tokens_i_half2 = learnable_tokens[i : i+1, half_len:, :]
                suffix_i = suffix[i : i+1, shape_name_len:, :]

                prompt = torch.cat([prefix_i, learnable_tokens_i_half1, class_i,
                                    learnable_tokens_i_half2, suffix_i], dim=1)
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_name_position == "end":
            prompts = torch.cat([prefix, learnable_tokens, suffix], dim=1)

        else:
            raise ValueError(f'`class_name_position`: {self.class_name_position} not in supported modes ["front", "middle", "end"]')

        # [num_classes, context_length, transformer_width]
        return prompts


class ULIP_WITH_IMAGE_RAW(nn.Module):
    def __init__(self, point_encoder, **kwargs):
        super().__init__()
        kwargs = EasyDict(kwargs)
        self.context_length = kwargs.context_length

        self.transformer = Transformer(
            width=kwargs.transformer_width,
            layers=kwargs.transformer_layers,
            heads=kwargs.transformer_heads,
            attn_mask=self.build_attention_mask(),
        )

        self.vocab_size = kwargs.vocab_size
        self.token_embedding = nn.Embedding(kwargs.vocab_size, kwargs.transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, kwargs.transformer_width))
        self.ln_final = LayerNorm(kwargs.transformer_width)

        self.text_projection = nn.Parameter(torch.empty(kwargs.transformer_width, kwargs.embed_dim))
        self.pc_projection = nn.Parameter(torch.empty(kwargs.pc_feat_dims, kwargs.embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.point_encoder = point_encoder
        self.initialize_parameters()

    def encode_text(self, text):
        x = self.token_embedding(text)
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)

        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)
        nn.init.normal_(self.pc_projection, std=512 ** -0.5)

    def encode_pc(self, pc):
        pc_feat = self.point_encoder(pc)
        pc_embed = pc_feat @ self.pc_projection
        return pc_embed

    def forward(self, pc, text, image=None):

        text_embed_all = []
        for i in range(text.shape[0]):
            text_for_one_sample = text[i]
            text_embed = self.encode_text(text_for_one_sample)
            text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
            text_embed = text_embed.mean(dim=0)
            text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
            text_embed_all.append(text_embed)

        text_embed_all = torch.stack(text_embed_all)
        pc_embed = self.encode_pc(pc)
        if image is not None:
            image_embed = self.encode_image(image)
            return {'text_embed': text_embed_all,
                    'pc_embed': pc_embed,
                    'image_embed': image_embed,
                    'logit_scale': self.logit_scale.exp()}

        else:
            return {'text_embed': text_embed_all,
                    'pc_embed': pc_embed,
                    'logit_scale': self.logit_scale.exp()}


class ULIP_WITH_IMAGE(nn.Module):
    def __init__(self, point_encoder, **kwargs):
        super().__init__()
        kwargs = EasyDict(kwargs)
        self.task = kwargs.task
        self.context_length = kwargs.context_length
        self.device = kwargs.device

        self.transformer = Transformer(
            width=kwargs.transformer_width,
            layers=kwargs.transformer_layers,
            heads=kwargs.transformer_heads,
            attn_mask=self.build_attention_mask(),
        )

        self.vocab_size = kwargs.vocab_size
        self.token_embedding = nn.Embedding(kwargs.vocab_size, kwargs.transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, kwargs.transformer_width))
        self.ln_final = LayerNorm(kwargs.transformer_width)

        self.text_projection = nn.Parameter(torch.empty(kwargs.transformer_width, kwargs.embed_dim))
        self.pc_projection = nn.Parameter(torch.empty(kwargs.pc_feat_dims, kwargs.embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.point_encoder = point_encoder

        self.prompt_learner = PromptLearner(kwargs)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

        self.initialize_parameters()

    def encode_text_ulip(self, text):
        x = self.token_embedding(text)
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)

        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def encode_text(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.unsqueeze(dim=0).repeat(prompts.shape[0], 1, 1)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x

    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        nn.init.normal_(self.prompt_learner.learnable_tokens, std=0.02)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)
        nn.init.normal_(self.pc_projection, std=512 ** -0.5)

    def encode_pc(self, pc):
        pc_feat, prompt_dict = self.point_encoder(pc)
        pc_embed = pc_feat @ self.pc_projection
        return pc_embed, prompt_dict

    def forward(self, pc, subsample=None):
        pc_embed, prompt_dict = self.encode_pc(pc)

        prompts = self.prompt_learner(self.token_embedding)
        tokenized_prompts = self.tokenized_prompts

        text_embed = self.encode_text(prompts, tokenized_prompts)
        text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * pc_embed @ text_embed.t()

        prompt_dict['pc_embed'], prompt_dict['text_embed'] = pc_embed, text_embed
        return logits, prompt_dict


def ULIP_PointBERT_RAW(args):
    from models.pointbert.point_encoder import PointTransformer_RAW
    config_addr = f'./models/pointbert/PointTransformer_{args.npoints}point.yaml'
    config = cfg_from_yaml_file(config_addr)
    point_encoder = PointTransformer_RAW(config.model, args=args)
    pc_feat_dims = 768
    # =====================================================================

    model = ULIP_WITH_IMAGE_RAW(embed_dim=512, point_encoder=point_encoder, context_length=77,
                                vocab_size=49408, transformer_width=512, transformer_heads=8, transformer_layers=12,
                                pc_feat_dims=pc_feat_dims)

    if not args.evaluate_3d:
        pretrain_slip_model = torch.load('./data/initialize_models/slip_base_100ep.pt', map_location=torch.device('cpu'))
        pretrain_slip_model_params = pretrain_slip_model['state_dict']
        pretrain_slip_model_params = {param_name.replace('module.', ''): param for param_name, param in
                                      pretrain_slip_model_params.items()}

        learnable_layers = []
        for name, param in model.named_parameters():
            if name not in pretrain_slip_model_params:
                learnable_layers.append(name)
                continue

            if isinstance(pretrain_slip_model_params[name], Parameter):
                param_new = pretrain_slip_model_params[name].data
            else:
                param_new = pretrain_slip_model_params[name]

            param.requires_grad = False
            print('load {} and freeze'.format(name))
            param.data.copy_(param_new)

        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('\n====================\nLearnable layers\n\t', '\n\t'.join(learnable_layers),
              '\n\n\tNumber of learnable params:', params, '\n====================\n')

    return model


def ULIP_PointBERT(args):
    from models.pointbert.point_encoder import PointTransformer
    config_addr = f'./models/pointbert/PointTransformer_{args.npoints}point.yaml'
    config = cfg_from_yaml_file(config_addr)
    point_encoder = PointTransformer(config.model, args=args)
    pc_feat_dims = 768
    # =====================================================================

    model = ULIP_WITH_IMAGE(embed_dim=512, point_encoder=point_encoder, context_length=77,
                            vocab_size=49408, classnames=args.classnames, class_name_position=args.class_name_position,
                            num_learnable_prompt_tokens=args.num_learnable_prompt_tokens, transformer_width=512, transformer_heads=8, transformer_layers=12,
                            pc_feat_dims=pc_feat_dims, device=args.gpu, task=args.task)

    if not args.evaluate_3d:
        if args.ulip2:
            pretrain_point_model = torch.load('./data/pretrained_models/pointbert_ulip2.pt', map_location=torch.device('cpu'))
        else:
            pretrain_point_model = torch.load('./data/pretrained_models/pointbert_ulip.pt', map_location=torch.device('cpu'))
        pretrain_point_model_params = pretrain_point_model['state_dict']
        pretrain_point_model_params = {param_name.replace('module.', ''): param for param_name, param in
                                       pretrain_point_model_params.items()}

        learnable_layers = []
        for name, param in model.named_parameters():
            if 'feature_extractor' in name:
                param.requires_grad = False
                print('load {} and freeze'.format(name))
                continue

            if name == 'point_encoder.prompt_token_pos' or 'point_encoder.prompter' in name:
                learnable_layers.append(name)
                continue

            if name == 'prompt_learner.learnable_tokens' or 'point_encoder.cls_head' in name:
                learnable_layers.append(name)
                continue

            if name in pretrain_point_model_params:
                if isinstance(pretrain_point_model_params[name], Parameter):
                    param_new = pretrain_point_model_params[name].data
                else:
                    param_new = pretrain_point_model_params[name]
            else:
                raise ValueError(f'=======Cannot find {name} in the pre-trained model=======')

            param.requires_grad = False
            print('load {} and freeze'.format(name))
            param.data.copy_(param_new)

        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('\n====================\nLearnable layers\n\t', '\n\t'.join(learnable_layers),
              '\n\n\tNumber of learnable params:', params, '\n====================\n')

    return model
