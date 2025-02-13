# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Mostly copy-paste from timm library.
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
import math
import logging
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from .utils import trunc_normal_
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


class AfterReconstruction(nn.Identity):
    def __init__(self, inplanes):
        super().__init__()
        self.inplanes = inplanes

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
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
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


class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(self,
                 img_size=[224],
                 patch_size=16,
                 in_chans=3,
                 num_classes=0,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 T=8,
                 drop_layers=[],
                 side_dim=320,
                 corr_layer_index=[],
                 corr_dim=320,
                 corr_func=None,
                 corr_window=None,
                 corr_ext_chnls=None,
                 corr_int_chnls=None,
                 corr_num_encoders=2,
                 **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Side Network
        self.T = T
        self.side_network = SideNetwork(width=embed_dim,
                                        layers=depth,
                                        heads=num_heads,
                                        dropout=dpr,
                                        side_dim=side_dim,
                                        T=T,
                                        patch_num=num_patches,
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
        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)

    def forward(self, x):
        x_side = rearrange(x, '(b t) c w h -> b c t w h', t=self.T)
        # Image network
        x = self.prepare_tokens(x) # BT, N, C
        act_img = []
        for blk in self.blocks:
            x = blk(x)
            act_img.append(x)
        act_img = torch.stack(act_img, dim=0)
        act_img = act_img.permute(0, 2, 1, 3)
        x = self.norm(x)
        # Side network
        x_side = self.side_pre_bn3d(self.side_conv1(x_side))
        x_side = rearrange(x_side, 'b c t w h -> (b t) (w h) c')
        x_side = x_side.permute(1, 0, 2)
        x_side = self.side_network(x_side, act_img)
        x_side = x_side.permute(1, 0, 2)
        h = int(x_side.shape[1] ** 0.5)
        x_side = rearrange(x_side, '(b t) (w h) d -> b d t w h', t=self.T, h=h)
        x_side = self.side_post_bn(x_side)
        x_side = x_side.flatten(2).mean(-1)
        return x_side

    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output


def vit_tiny(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_small(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x