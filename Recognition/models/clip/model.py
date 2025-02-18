from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from random import sample
from modules.moss import MOSSBlock

from utils.logger import setup_logger, get_logger

logger = get_logger(__name__)

def conv_3xnxn_std(inp, oup, kernel_size=3, stride=3, groups=1):
    return nn.Conv3d(inp, oup, (3, kernel_size, kernel_size), (1, stride, stride), (1, 0, 0), groups=groups)

def bn_3d(dim):
    return nn.BatchNorm3d(dim)

def conv_3x1x1(inp, oup, groups=1):
    return nn.Conv3d(inp, oup, (3, 1, 1), (1, 1, 1), (1, 0, 0), groups=groups)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x



def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
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


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, dropout = 0.):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.ln_1 = LayerNorm(d_model)
        
        self.drop_path = DropPath(dropout) if dropout > 0. else nn.Identity()
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

        self.control_point1 = AfterReconstruction(d_model)
        self.control_point2 = AfterReconstruction(d_model)
        self.control_atm = AfterReconstruction(d_model)

    def attention(self, x: torch.Tensor):
        # x: 50 bT c
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor, use_ckpt=False):
        x = self.control_atm(x)
        x = self.control_point1(x)
        # MHSA
        if use_ckpt:
            attn_out = checkpoint.checkpoint(self.attention, self.ln_1.float()(x))
            x = x + self.drop_path(attn_out)
        else:
            x = x + self.drop_path(self.attention(self.ln_1.float()(x)))

        x = self.control_point2(x)
        # FFN
        if use_ckpt:
            mlp_out = checkpoint.checkpoint(self.mlp, self.ln_2.float()(x))
            x = x + self.drop_path(mlp_out)
        else:
            x = x + self.drop_path(self.mlp(self.ln_2.float()(x)))
        return x
     
class Transformer(nn.Module):
    def __init__(self,
                 width: int,
                 layers: int,
                 heads: int,
                 attn_mask: torch.Tensor = None,
                 dropout=None,
                 ):
        super().__init__()
        if dropout is None:
            dropout = [0.0 for i in range(layers)] 
        logger.info('dropout used:{}'.format(dropout))
        self.width = width
        self.layers = layers
        self.resblocks = []
        for i in range(layers):
            self.resblocks.append(ResidualAttentionBlock(width, heads, attn_mask, dropout=dropout[i]))
        self.resblocks = nn.ModuleList(self.resblocks)
        
    def forward(self, x: torch.tensor):
        act = []
        for i in range(len(self.resblocks)):
            x = self.resblocks[i](x)
            act.append(x)
        act = torch.stack(act, dim=0)
        return x, act


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
            # MOSS block
            if i_img in self.corr_layer_index:
                x_corr = self.moss_layers[l](x_img[i_img][1:])
                x = x + x_corr
                l += 1
            # resblock
            use_checkpoint = self.num_checkpoints > 0 and i_vid >= len(self.resblocks) - self.num_checkpoints
            x = self.resblocks[i_vid](x, x_token,
                                      self.side_spatial_position_embeddings,
                                      use_ckpt=use_checkpoint)
        return x


class VisualTransformer(nn.Module):
    def __init__(self,
                 input_resolution: int,
                 patch_size: int,
                 width: int,
                 layers: int,
                 heads: int,
                 output_dim: int,
                 dropout = None,
                 joint=False,
                 emb_dropout: float = 0.,
                 T: int = 8,
                 drop_layers: list = [],
                 side_dim: int = 384,
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
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.dropout = nn.Dropout(emb_dropout)
        self.ln_pre = LayerNorm(width)
        self.emb_dropout = emb_dropout
        self.joint = joint
        self.T = T
        if joint:
            logger.info('=====using space-time attention====')
            self.time_embedding = nn.Parameter(scale * torch.randn(T, width))  # pos emb
        if emb_dropout > 0:
            logger.info('emb_dropout:{}'.format(emb_dropout))
        # if max(dropout) > 0:
        #     logger.info('dropout:{}'.format(dropout))
        self.side_dim = side_dim
        ## Attention Blocks
        self.transformer = Transformer(width,
                                       layers,
                                       heads,
                                       dropout=None
                                       )
        self.side_network = SideNetwork(width,
                                    layers,
                                    heads,
                                    dropout=dropout,
                                    side_dim=side_dim,
                                    T=T,
                                    patch_num=(input_resolution // patch_size) ** 2,
                                    drop_layers=drop_layers,
                                    corr_dim=corr_dim,
                                    corr_func=corr_func,
                                    corr_layer_index=corr_layer_index,
                                    corr_window=corr_window,
                                    corr_ext_chnls=corr_ext_chnls,
                                    corr_int_chnls=corr_int_chnls,
                                    corr_num_encoders=corr_num_encoders,
                                    num_checkpoints=num_checkpoints
                                    )
        self.side_post_bn = bn_3d(self.side_dim)
        self.side_conv1 = conv_3xnxn_std(3, self.side_dim, kernel_size=patch_size, stride=patch_size)
        self.side_pre_bn3d = nn.BatchNorm3d(self.side_dim)
        nn.init.ones_(self.side_pre_bn3d.weight)
        nn.init.zeros_(self.side_pre_bn3d.bias)
        nn.init.ones_(self.side_post_bn.weight)
        nn.init.zeros_(self.side_post_bn.bias)

    def forward(self, x: torch.Tensor):
        x_side = rearrange(x, '(b t) c h w -> b c t h w', t=self.T)

        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        
        x_side = self.side_pre_bn3d(self.side_conv1(x_side))
        x_side = rearrange(x_side, 'b c t h w -> (b t) (h w) c')

        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        
        if self.joint:
            B = x.shape[0] // self.T
            cls_tokens = x[:B, 0, :].unsqueeze(1)  # only one cls_token
            x = x[:,1:]
            x = rearrange(x, '(b t) n c -> (b n) t c',b=B,t=self.T)
            x = x + self.time_embedding.to(x.dtype)   # temporal pos emb
            x = rearrange(x, '(b n) t c -> b (n t) c',b=B,t=self.T)
            x = torch.cat((cls_tokens, x), dim=1)

        if self.emb_dropout > 0:
            x = self.dropout(x)
        
        x = self.ln_pre.float()(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x_side = x_side.permute(1, 0, 2)

        _, act_img = self.transformer(x)
        x_side = self.side_network(x_side, act_img)
        x_side = x_side.permute(1, 0, 2)

        h = int(x_side.shape[1] ** 0.5)
        x_side = rearrange(x_side, '(b t) (h w) d -> b d t h w', t=self.T, h=h)
        x_side = self.side_post_bn(x_side)
        x_side = x_side.flatten(2).mean(-1)

        return x_side


class CLIP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.side_dim = config.network.side_dim
        self.T = config.data.num_segments
        self.context_length = config.network.context_length
        if config.network.dropout > 0.:
            dpr = [x.item() for x in torch.linspace(0, config.network.dropout, config.network.vision_layers)]  # stochastic depth decay rule
        else:
            dpr = None

        if isinstance(config.network.vision_layers, (tuple, list)):
            vision_heads = config.network.vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=config.network.vision_layers,
                output_dim=config.network.embed_dim,
                heads=vision_heads,
                input_resolution=config.network.image_resolution,
                width=config.network.vision_width
            )
            if config.network.tm:
                logger.info('=========using Temporal Shift Module==========')
                from modules.temporal_modeling import make_temporal_shift
                make_temporal_shift(self.visual, self.T)

        else:
            vision_heads = config.network.vision_width // 64
            self.visual = VisualTransformer(
                input_resolution=config.network.image_resolution,
                patch_size=config.network.vision_patch_size,
                width=config.network.vision_width,
                layers=config.network.vision_layers,
                heads=vision_heads,
                output_dim=config.network.embed_dim,
                joint=config.network.joint,dropout=dpr,
                emb_dropout=config.network.emb_dropout,
                T=self.T,
                drop_layers=config.network.drop_layers,
                side_dim=config.network.side_dim,
                corr_dim=config.network.corr_dim,
                corr_layer_index=config.network.corr_layer_index,
                corr_window=config.network.corr_window,
                corr_ext_chnls=config.network.corr_ext_chnls,
                corr_int_chnls=config.network.corr_int_chnls,
                corr_func=config.network.corr_func,
                corr_num_encoders=config.network.corr_num_encoders,
                num_checkpoints=config.network.num_checkpoints
            )
            if config.network.tm == 'tsm':
                logger.info('=========using Temporal Shift Module==========')
                from modules.temporal_modeling import make_temporal_shift_vit
                make_temporal_shift_vit(self.visual, self.T)
            elif config.network.tm == 'tokenshift':
                logger.info('=========using TokenShift =========={} layers'.format(config.network.vision_layers))
                from modules.temporal_modeling import make_tokenshift
                make_tokenshift(
                    self.visual, self.T, n_div=4,
                    locations_list=[x for x in range(config.network.vision_layers)]
                )
            elif config.network.tm == "tokent1d":
                logger.info('=========using TokenT1D ==========')
                from modules.temporal_modeling import make_tokenT1D
                make_tokenT1D(
                    self.visual, self.T, n_div=4,
                    locations_list=[x for x in range(config.network.vision_layers)]
                )                
            elif config.network.tm == 'dividedST':
                logger.info('=========using DividedST ==========')
                from modules.temporal_modeling import make_DividedST
                make_DividedST(
                    self.visual, self.T, vision_heads, config.network.emb_dropout, None,
                    locations_list=[8,9,10,11]
                )

            elif config.network.tm == 'localuni':
                logger.info('=========using LocalUni ==========')
                from modules.temporal_modeling import make_LocalUni
                if config.network.vision_layers == 12:
                    start = int(config.network.vision_layers * 1/3)
                else:
                    start = int(config.network.vision_layers * 1/3)
                make_LocalUni(
                    self.visual, self.T, vision_heads, config.network.emb_dropout, None,
                    locations_list=[x for x in range(start, config.network.vision_layers)]
                )                
            elif config.network.tm == 't1d':
                logger.info('=========using T1D ==========')
                from modules.temporal_modeling import make_T1D4VIT
                if config.network.vision_layers == 12:
                    start = int(config.network.vision_layers * 1/3)
                else:
                    start = int(config.network.vision_layers * 1/3)
                make_T1D4VIT(
                    self.visual, self.T,
                    locations_list=[x for x in range(start, config.network.vision_layers)]
                )    

            elif config.network.tm == 'atm':
                logger.info('=========using ATM ==========')
                from modules.ATM import make_ATM
                if config.network.vision_layers == 12:
                    start = 10 # int(vision_layers * 1/3)
                else:
                    start = 22 # int(vision_layers * 1/3)
                make_ATM(
                    self.visual, self.T,
                    locations_list=[x for x in range(start, config.network.vision_layers)]
                )  

        self.transformer = Transformer(
            width=config.network.transformer_width,
            layers=config.network.transformer_layers,
            heads=config.network.transformer_heads,
            attn_mask=self.build_attention_mask(),
            dropout=dpr
        )

        self.vocab_size = config.network.vocab_size
        self.token_embedding = nn.Embedding(config.network.vocab_size, config.network.transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, config.network.transformer_width))
        self.ln_final = LayerNorm(config.network.transformer_width)
        
        self.dropout = nn.Dropout(config.network.emb_dropout)
        self.emb_dropout = config.network.emb_dropout
        
        self.text_projection = nn.Parameter(torch.empty(config.network.transformer_width, config.network.embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)
                        
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        side_fc_std = (2 * self.side_dim) ** -0.5

        # for block in self.visual.side_network.adaptation:
        #     nn.init.normal_(block.weight, std=side_fc_std)

        # for block in self.visual.side_network.lns_pre:
        #     nn.init.zeros_(block.bias)
        #     nn.init.ones_(block.weight)

        # for block in self.visual.side_network.side_corr_linears:
        #     nn.init.normal_(block.weight, std=side_fc_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, images):
        return self.visual(images.type(self.dtype))


    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        if self.emb_dropout > 0:
            x = self.dropout(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x    


    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        return image_features, text_features, self.logit_scale.exp()


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict,
                config
                ):
    """
    Build model from state_dict and config

    Args:
        state_dict (dict): model state dictionary
        config (dict): model configuration

    Returns:
        (nn.Module): model
    """
    
    vit = "visual.proj" in state_dict

    if vit:
        config.network.vision_width = state_dict["visual.conv1.weight"].shape[0]
        config.network.vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        config.network.vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        config.network.grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        config.network.image_resolution = config.network.vision_patch_size * config.network.grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        config.network.vision_layers = tuple(counts)        
        config.network.vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        config.network.output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        config.network.vision_patch_size = None
        assert config.network.output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        config.network.image_resolution = config.network.output_width * 32

    config.network.embed_dim = state_dict["text_projection"].shape[1]
    config.network.context_length = state_dict["positional_embedding"].shape[0]
    config.network.vocab_size = state_dict["token_embedding.weight"].shape[0]
    config.network.transformer_width = state_dict["ln_final.weight"].shape[0]
    config.network.transformer_heads = config.network.transformer_width // 64
    config.network.transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))
    
    model = CLIP(config)

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]
    # tm is set to False by default.
    if config.network.tm == True or config.network.tm in ["tsm", "tokenshift"]:
        # old dict for new model, rename some keys
        model_dict = model.state_dict()
        replace_dict = []
        for k, v in model_dict.items():
            if k not in state_dict and k.replace('.net', '') in state_dict:
                replace_dict.append((k.replace('.net', ''), k))
        for k, k_new in replace_dict:
            state_dict[k_new] = state_dict.pop(k)


    convert_weights(model)
    if config.network.init:
        logger.info('loading clip pretrained model!')
        if config.network.joint_st and config.network.tm != "dividedST":  #or emb_dropout>0 or dropout>0
            model.load_state_dict(state_dict,strict=False)
        else:
            if config.network.tm == "tokent1d":
                model.load_state_dict(state_dict, strict=False)
            elif config.network.tm == "localuni":
                model.load_state_dict(state_dict, strict=False)
            elif config.network.tm == "t1d":
                model.load_state_dict(state_dict, strict=False)                
            elif config.network.tm == "dividedST":
                # model.load_state_dict(state_dict, strict=False)
                model_dict = model.state_dict()
                new_state_dict = state_dict.copy()
                for key in state_dict:
                    if 'visual.transformer.resblocks' in key and 'attn' in key:
                        new_key1 = key.replace('attn','control_point1.temporal_attn')
                        new_key2 = key.replace('attn','control_point2.temporal_attn')
                        if new_key1 in model_dict:
                            new_state_dict[new_key1] = state_dict[key]
                        if new_key2 in model_dict:
                                new_state_dict[new_key2] = state_dict[key]                            
                    if 'visual.transformer.resblocks' in key and 'ln' in key:
                        new_key1 = key.replace('ln_1', 'control_point1.temporal_ln')
                        new_key2 = key.replace('ln_1', 'control_point2.temporal_ln')
                        if new_key1 in model_dict:
                            new_state_dict[new_key1] = state_dict[key]
                        if new_key2 in model_dict:
                            new_state_dict[new_key2] = state_dict[key]
                model.load_state_dict(new_state_dict)
            else:
                model.load_state_dict(state_dict, strict=False)
    else:
        logger.info('not using full clip pretrained model, only visual!')
        
        for k in list(state_dict.keys()):
            if not k.find("visual")>-1: 
                state_dict.pop(k)

        model.load_state_dict(state_dict,strict=False)

    return model.eval()
