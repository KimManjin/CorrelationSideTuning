import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class STSSTransformation(nn.Module):
    def __init__(self,
                 num_segments,
                 window=(5,9,9),
                 corr_func="cosine"):
        super(STSSTransformation, self).__init__()
        self.num_segments = num_segments
        self.window = window
        assert window[1] == window[2]
        self.corr_func = corr_func
        if self.corr_func == "cosine":
            self.pad_value = 0 # -1 #TODO: check which one is better
        elif self.corr_func == "dotproduct_softmax":
            self.pad_value = -float("Inf")
        else:
            self.pad_value = 0
        


    def _convert_global_to_local(self, corr_g):
        """
        Convert absolute correlation to relative correlation.
        
        Args:
        - corr_g (torch.Tensor): Input global correlation tensor of shape (b, h, w, h, w)
        - h (int): Height of the feature map
        - w (int): Width of the feature map
        
        Returns:
        - torch.Tensor: Relative correlation tensor of shape (b, h, w, window, window)
        """
        max_d = self.window[1] // 2

        # Convert global correlation to local correlation
        corr_l = [F.pad(torch.diagonal(corr_g, offset=i, dim1=1, dim2=3),
                        (abs(i) if i<0 else 0, abs(i) if i>=0 else 0),
                        value=self.pad_value) \
                        for i in range(-max_d, max_d+1)]
        corr_l = torch.stack(corr_l, dim=-1) # B, W1, W2, H1, H2 -> U

        corr_l = [F.pad(torch.diagonal(corr_l, offset=i, dim1=1, dim2=2),
                        (abs(i) if i<0 else 0, abs(i) if i>=0 else 0),
                        value=self.pad_value) \
                        for i in range(-max_d, max_d+1)]
        corr_l = torch.stack(corr_l, dim=-1) # B, H1, H2 -> U, W1, W2 -> V
        corr_l = corr_l.transpose(2, 3).contiguous() # B, H1, W1, H2 -> U, W2 -> V
        
        return corr_l # B, H1, W1, U, V
    

    def _correlation(self, feat1, feat2):
        if self.corr_func == "cosine":
            feat1 = F.normalize(feat1, p=2, dim=1, eps=1e-7) # btl, c, h, w
            feat2 = F.normalize(feat2, p=2, dim=1, eps=1e-7) # btl, c, h, w
        elif self.corr_func in ["dotproduct", "dotproduct_softmax"]:
            scale = feat1.size(1) ** -0.5
            feat1 = feat1 * scale
            
        corr = torch.einsum('bchw,bcuv->bhwuv', feat1, feat2)
        corr = self._convert_global_to_local(corr)

        if self.corr_func == "dotproduct_softmax":
            corr_shape = corr.shape
            corr = rearrange(corr, 'b h w u v -> b h w (u v)')
            corr = F.softmax(corr, dim=-1)
            corr = corr.reshape(corr_shape)

        return corr


    def forward(self, x):
        # resize spatial resolution to 14x14

        if self.window[0] > 1:
            x = rearrange(x, '(b t) c h w -> b t c h w', t=self.num_segments)
            x_src = repeat(x, 'b t c h w -> (b t l) c h w', l=self.window[0])
            x_tgt = F.pad(x, (0,0,0,0,0,0,self.window[0]//2,self.window[0]//2), 'constant', 0).unfold(1,self.window[0],1)
            x_tgt = rearrange(x_tgt, 'b t c h w l -> (b t l) c h w')     
        else:
            x_src = x
            x = rearrange(x, '(b t) c h w -> b t c h w', t=self.num_segments)
            x_tgt = torch.cat((x[:, 0].unsqueeze(1), x[:, :-1]), 1)
            x_tgt = rearrange(x_tgt, 'b t c h w -> (b t) c h w')
            
        stss = self._correlation(x_src, x_tgt)   
        stss = rearrange(stss, '(b t l) h w u v -> b t h w 1 l u v', t=self.num_segments, l=self.window[0])
        return stss

    
class STSSExtraction(nn.Module):
    def __init__(self, num_segments, window=(5,9,9), chnls=(4,16,64,64),):
        super(STSSExtraction, self).__init__()
        self.num_segments = num_segments
        self.window = window
        self.chnls = chnls
        
        self.conv0 = nn.Sequential(
            nn.Conv3d(self.window[1]*self.window[2], chnls[0], kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0), bias=False),
            nn.BatchNorm3d(chnls[0]),
            nn.GELU())
        
        
        # self.conv0 = nn.Sequential(
        #     nn.Conv3d(1, chnls[0], kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=False),
        #     nn.BatchNorm3d(chnls[0]),
        #     nn.GELU())
        
        # self.conv1 = nn.Sequential(
        #     nn.Conv3d(chnls[0], chnls[1], kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1), bias=False),
        #     nn.BatchNorm3d(chnls[1]),
        #     nn.GELU())
        
        # self.conv2 = nn.Sequential(
        #     nn.Conv3d(chnls[1], chnls[2], kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1), bias=False),
        #     nn.BatchNorm3d(chnls[2]),
        #     nn.GELU())
        
        # self.conv3 = nn.Sequential(
        #     nn.Conv3d(chnls[2], chnls[3], kernel_size=(1,3,3), stride=(1,1,1), padding=(0,0,0), bias=False),
        #     nn.BatchNorm3d(chnls[3]),
        #     nn.GELU())    
        
    def forward(self, x):
        b,t,h,w,_,l,u,v = x.size()
        x = rearrange(x, 'b t h w 1 l u v -> (b l) (u v) t h w', t=t, h=h, w=w)
        x = self.conv0(x)
        return x
        # x = rearrange(x, 'b t h w 1 l u v -> (b t h w) 1 l u v')
        # x = self.conv0(x)
        # x = self.conv1(x)
        # x = self.conv2(x)
        # x = self.conv3(x)
        # x = rearrange(x, '(b t h w) c l 1 1 -> (b l) c t h w', t=t, h=h, w=w)
        
        # return x
    
    
class STSSIntegration(nn.Module):
    def __init__(self,
                 d_in,
                 num_segments,
                 window=(5,9,9),
                 chnls=(64,64,64)):
        super(STSSIntegration, self).__init__()
        self.num_segments = num_segments
        self.window = window
        self.chnls = chnls
        
        self.conv0 = nn.Sequential(
            nn.Conv3d(d_in, chnls[0], kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=False),
            nn.BatchNorm3d(chnls[0]),
            nn.GELU())
        
        self.conv1 = nn.Sequential(
            nn.Conv3d(chnls[0], chnls[1], kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=False),
            nn.BatchNorm3d(chnls[1]),
            nn.GELU())
        
        self.conv2_fuse = nn.Sequential(
            Rearrange('(b l) c t h w -> b (l c) t h w', l=self.window[0]),
            nn.Conv3d(chnls[1]*self.window[0], chnls[2], kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=False),
            nn.BatchNorm3d(chnls[2]),
            nn.GELU()
        )
        

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2_fuse(x)
        
        return x
    
    
    
class SELFYBlock(nn.Module):
    def __init__(self,
                 d_in,
                 d_hid,
                 d_out,
                 num_segments=8,
                 window=(5,9,9),
                 ext_chnls=(4,16,64,64),
                 int_chnls=(64,64,64,64),
                 corr_func="cosine"
                ):
        """
        SELFY block.

        Args:
        - d_in (int): Input feature dimension
        - d_hid (int): Hidden feature dimension
        - d_out (int): Output feature dimension
        - num_segments (int): Number of temporal segments
        - window (tuple): Window size for spatio-temporal self-attention
        - ext_chnls (tuple): Number of channels for each layer in the extraction module
        - int_chnls (tuple): Number of channels for each layer in the integration module
        - corr_func (str): Correlation function to use

        Returns:
        - torch.Tensor: Output tensor of shape (b, d_in, num_segments, h, w)
        """
        super(SELFYBlock, self).__init__()
        self.window = window
        self.ln_pre = LayerNorm(d_in)
        self.in_proj = nn.Linear(d_in, d_hid)
        self.stss_transformation = STSSTransformation(
            num_segments=num_segments,
            window=window,
            corr_func=corr_func
        )
        
        self.stss_extraction = STSSExtraction(
            num_segments=num_segments,
            window = window,
            chnls = ext_chnls
        )
        
        self.stss_integration = STSSIntegration(
            ext_chnls[-1],
            num_segments=num_segments,
            window = window,
            chnls = int_chnls
        )
        self.out_proj = nn.Linear(int_chnls[-1], d_out)
        
    def forward(self, x):
        # identity = x
        # x shape: (H x W + 1, B x T, C)
        x = self.in_proj(self.ln_pre(x))
        H = W = int(sqrt(x.shape[0]))
        x = rearrange(x, '(h w) bt c -> bt c h w', h=H, w=W)
        # stss transformation -> stss extraction -> stss integration
        x = self.stss_transformation(x)
        x = self.stss_extraction(x)
        x = self.stss_integration(x)
        # Output projection
        x = self.out_proj(rearrange(x, 'b c t h w -> (h w) (b t) c '))
        return x