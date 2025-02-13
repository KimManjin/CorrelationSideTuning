import logging
import torch
import torch.nn as nn

from .models_vit import VisionTransformer

logger = logging.getLogger(__name__)

def build_model_from_checkpoints(config, pretrained=None):
    # load pretrained weights
    assert pretrained is not None, "pretrained weight is required"
    state_dict = torch.load(pretrained, map_location="cpu")["model"]
    # get model config
    config.network.vision_width = state_dict["patch_embed.proj.weight"].shape[0]
    config.network.vision_layers = len([k for k in state_dict.keys() if k.startswith("blocks.") and k.endswith(".attn.proj.weight")])
    config.network.vision_patch_size = state_dict["patch_embed.proj.weight"].shape[-1]
    # build model
    model = VisionTransformer(
        img_size=config.data.input_size,
        patch_size=config.network.vision_patch_size,
        in_chans=3,
        num_classes=0, # only use backbone
        embed_dim=config.network.vision_width,
        depth=config.network.vision_layers,
        num_heads=config.network.vision_width // 64,
        mlp_ratio=4.0,
        qkv_bias=True,
        T=config.data.num_segments,
        side_dim=config.network.side_dim,
        drop_layers=config.network.drop_layers,
        corr_layer_index=config.network.corr_layer_index,
        corr_dim=config.network.corr_dim,
        corr_func=config.network.corr_func,
        corr_window=config.network.corr_window,
        corr_ext_chnls=config.network.corr_ext_chnls,
        corr_int_chnls=config.network.corr_int_chnls,
        corr_num_encoders=config.network.corr_num_encoders,
    )
    msg = model.load_state_dict(state_dict, strict=False)
    
    logger.info("Pretrained weights found at {} and loaded with msg: {}".format(pretrained, msg))
    model.eval()
    return model
