# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import math
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange
import timm.models.vision_transformer
from modules.moss import MOSSBlock

logger = logging.getLogger(__name__)


def conv_3xnxn_std(inp, oup, kernel_size=3, stride=3, groups=1):
    return nn.Conv3d(inp, oup, (3, kernel_size, kernel_size), (1, stride, stride), (1, 0, 0), groups=groups)

def bn_3d(dim):
    return nn.BatchNorm3d(dim)

def conv_3x1x1(inp, oup, groups=1):
    return nn.Conv3d(inp, oup, (3, 1, 1), (1, 1, 1), (1, 0, 0), groups=groups)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class CMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        

    def forward(self, x):
        h = int(x.shape[0] ** 0.5)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class AttnCBlock(nn.Module):
    def __init__(self,
                 dim,
                 side_dim,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 kernel_size=5,
                 T=8
                 ):
        super().__init__()
        self.bn_1 = bn_3d(dim)
        self.conv = nn.Sequential(*[
            nn.Conv3d(dim, side_dim, (1, 1, 1), (1, 1, 1), (0, 0, 0), groups=1),
            nn.Conv3d(side_dim, side_dim, (3, 1, 1), (1, 1, 1), (1, 0, 0), groups=side_dim),
            nn.Conv3d(side_dim, dim, (1, 1, 1), (1, 1, 1), (0, 0, 0), groups=1),
        ])
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.bn_2 = bn_3d(dim)
        mlp_hidden_dim = int(dim* mlp_ratio)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.attn = nn.MultiheadAttention(dim, dim//64, dropout=0.)
        self.ln_1 = LayerNorm(dim)

        self.T = T
        side_attn_std = dim ** -0.5
        side_fc_std = (2 * dim) ** -0.5
        side_proj_std = (dim ** -0.5) * ((2 * 12) ** -0.5)
        for name, p in self.named_parameters():
            if 'mlp.fc1.weight' in name:
                nn.init.normal_(p, std=side_fc_std)
            elif 'mlp.fc2.weight' in name:
                nn.init.normal_(p, std=side_proj_std)
            elif 'pw_conv1.weight' in name:
                nn.init.normal_(p, std=0.02)
            elif 'pw_conv2.weight' in name:
                nn.init.normal_(p, std=0.02)
            elif 'dw_conv1.weight' in name:
                nn.init.normal_(p, std=side_attn_std)
            elif 'attn.in_proj_weight' in name:
                nn.init.normal_(p, std = side_attn_std)
            elif 'attn.out_proj.weight' in name:
                nn.init.normal_(p, std = side_proj_std)

        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
        if isinstance(m, nn.BatchNorm3d):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def attention(self, x: torch.Tensor):
        # x: 50 bT c
        self.attn_mask = None # self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x[1:, :, :], x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def shift_token(self, x_token): # [1, bt, d]
        random_num = np.random.uniform()
        c = x_token.shape[-1]
        fold = c // 2
        x_token = rearrange(x_token, 'n (b t) d -> n b t d', t=self.T)
        out = torch.zeros_like(x_token)
        out[:, :, :-1, :fold] = x_token[:, :, 1:, :fold]
        out[:, :, 1:, fold:] = x_token[:, :, :-1, fold:]
        out = rearrange(out, 'n b t d -> n (b t) d')
        return out

    def forward(self, x, x_token=None, side_position_embeddings=None, use_ckpt=False):
        n, bt, d = x.size()
        h = int(x.shape[0] ** 0.5)
        x = rearrange(x, '(h w) (b t) d -> b d t h w', h=h, t=self.T)
        if use_ckpt:
            conv_out = checkpoint.checkpoint(self.conv, self.bn_1(x))
            x = x + self.drop_path(conv_out)
        else:
            x = x + self.drop_path(self.conv(self.bn_1(x)))
        x = rearrange(x, 'b d t h w -> (h w) (b t) d', h=h, t=self.T)

        ## shift class token
        x_token = self.shift_token(x_token)
        xt = torch.cat([x_token, x], dim=0)
        xt = xt.permute(1, 0, 2)
        xt[:, 1:, :] = xt[:, 1:, :] + side_position_embeddings
        xt = xt.permute(1, 0, 2)
        if use_ckpt:
            attn_out = checkpoint.checkpoint(self.attention, self.ln_1(xt))
            x = x + self.drop_path(attn_out)
        else:
            xt = self.drop_path(self.attention(self.ln_1(xt)))
            x = x + xt

        x_ = x
        x = rearrange(x, '(h w) (b t) d -> b d t h w', h=h, t=self.T)
        x = self.bn_2(x)
        x = rearrange(x, 'b d t h w -> (h w) (b t) d', h=h, t=self.T)
        if use_ckpt:
            mlp_out = checkpoint.checkpoint(self.mlp, x)
            x = x_ + self.drop_path(mlp_out)
        else:
            x = x_ + self.drop_path(self.mlp(x))
        return x


class SideNetwork(nn.Module):
    def __init__(self,
                 width: int,
                 layers: int,
                 heads: int,
                 attn_mask: torch.Tensor = None,
                 dropout=None,
                 side_dim=384,
                 T=8,
                 patch_num=49,
                 drop_layers: list = [],
                 corr_dim: int = 128,
                 corr_func: str = "cosine",
                 corr_layer_index: list = [],
                 corr_window: list = [5, 9, 9],
                 corr_ext_chnls: list = [4, 16, 64, 64],
                 corr_int_chnls: list = [64, 64, 128],
                 corr_num_encoders: int = 2,
                 num_checkpoints: int = 0
                 ):
        super().__init__()
        if dropout is None:
            dropout = [0.0 for i in range(layers)] 
        logger.info('dropout used:{}'.format(dropout))

        self.width = width
        self.layers = layers
        self.T = T
        self.num_checkpoints = num_checkpoints
        self.resblocks = []
        self.adaptation = []
        self.lns_pre = []
        self.drop_layers = drop_layers
        self.side_layers = [l for l in range(self.layers) if l not in self.drop_layers]
        self.corr_dim = corr_dim
        self.side_dim = side_dim
        self.temporal_ratio = 1
        for i in range(len(self.side_layers)):
            self.resblocks.append(AttnCBlock(self.side_dim,
                                             int(self.side_dim * self.temporal_ratio),
                                             kernel_size=1,
                                             T=self.T,
                                             drop_path=dropout[self.side_layers[i]]))
            self.adaptation.append(nn.Linear(width, self.side_dim))
            self.lns_pre.append(LayerNorm(width))
        self.resblocks = nn.ModuleList(self.resblocks)
        self.adaptation = nn.ModuleList(self.adaptation)
        self.lns_pre = nn.ModuleList(self.lns_pre)
        side_scale = self.side_dim ** -0.5
        self.side_spatial_position_embeddings = nn.Parameter(side_scale * torch.randn((patch_num, self.side_dim)))
        # SELFY block
        self.corr_layer_index = corr_layer_index
        self.moss_layers = []
        for i in self.corr_layer_index:
            self.moss_layers.append(MOSSBlock(
                                d_in=width,
                                d_hid=self.corr_dim,
                                d_out=self.side_dim,
                                num_segments=T,
                                window=corr_window,
                                ext_chnls=corr_ext_chnls,
                                int_chnls=corr_int_chnls,
                                corr_func=corr_func,
                                n_encoders=corr_num_encoders
                                ))
        self.moss_layers = nn.ModuleList(self.moss_layers)
        # init weights
        nn.init.normal_(self.side_spatial_position_embeddings, std=0.01)

    def forward(self, x: torch.tensor, x_img: torch.tensor):
        l = 0
        for i_vid, i_img in enumerate(self.side_layers):
            xs2xt = self.adaptation[i_vid](self.lns_pre[i_vid](x_img[i_img]))
            x_token = xs2xt[:1, :, :]
            xs2xt = xs2xt[1:, :, :]
            x = 0.5 * x + 0.5 * xs2xt
            # SELFY block
            if i_img in self.corr_layer_index:
                x_corr = self.moss_layers[l](x_img[i_img][1:])
                x = x + x_corr
                l += 1
            # resblock
            x = self.resblocks[i_vid](x, x_token,
                                      self.side_spatial_position_embeddings,
                                      use_ckpt=False)
        return x


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self,
                 img_size=224, patch_size=16, in_chans=3,embed_dim=768,
                 num_classes=0, depth=12, num_heads=12, mlp_ratio=4,
                 qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 dropout=0.0, T=1, side_dim=384, drop_layers=[], corr_dim=128,
                 corr_layer_index=[], corr_func="cosine", corr_window=[5, 9, 9],
                 corr_ext_chnls=[4, 16, 64, 64], corr_int_chnls=[64, 64, 128],
                 corr_num_encoders=2,
                 **kwargs):
        super(VisionTransformer, self).__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            num_classes=num_classes,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            **kwargs)

        del self.norm  # remove the original norm
        # Side Network
        self.T = T
        side_layers = [l for l in range(depth) if l not in drop_layers]
        dpr = [x.item() for x in torch.linspace(0, dropout, len(side_layers))]  # stochastic depth decay rule
        self.side_network = SideNetwork(width=embed_dim,
                                        layers=depth,
                                        heads=num_heads,
                                        dropout=dpr,
                                        side_dim=side_dim,
                                        T=T,
                                        patch_num=(img_size // patch_size) ** 2,
                                        drop_layers=drop_layers,
                                        corr_dim=corr_dim,
                                        corr_func=corr_func,
                                        corr_layer_index=corr_layer_index,
                                        corr_window=corr_window,
                                        corr_ext_chnls=corr_ext_chnls,
                                        corr_int_chnls=corr_int_chnls,
                                        corr_num_encoders=corr_num_encoders)
        self.side_post_bn = bn_3d(side_dim)
        self.side_conv1 = conv_3xnxn_std(3, side_dim, kernel_size=patch_size, stride=patch_size)
        self.side_pre_bn3d = nn.BatchNorm3d(side_dim)
        nn.init.ones_(self.side_pre_bn3d.weight)
        nn.init.zeros_(self.side_pre_bn3d.bias)
        nn.init.ones_(self.side_post_bn.weight)
        nn.init.zeros_(self.side_post_bn.bias)

    def forward_features(self, x):
        x_side = rearrange(x, '(b t) c h w -> b c t h w', t=self.T)
        # Image Network
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        act_img = []
        for blk in self.blocks:
            x = blk(x)
            act_img.append(x)
        act_img = torch.stack(act_img, dim=0)
        act_img = act_img.permute(0, 2, 1, 3)
        # Side Network
        x_side = self.side_pre_bn3d(self.side_conv1(x_side))
        x_side = rearrange(x_side, 'b c t h w -> (b t) (h w) c')
        x_side = x_side.permute(1, 0, 2)
        x_side = self.side_network(x_side, act_img)
        x_side = x_side.permute(1, 0, 2)
        h = int(x_side.shape[1] ** 0.5)
        x_side = rearrange(x_side, '(b t) (h w) d -> b d t h w', t=self.T, h=h)
        x_side = self.side_post_bn(x_side)
        x_side = x_side.flatten(2).mean(-1)
        return x_side

    def forward(self, x):
        x = self.forward_features(x)
        return x


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model