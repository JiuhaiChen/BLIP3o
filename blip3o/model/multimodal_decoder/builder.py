from diffusers import AutoencoderDC, SanaTransformer2DModel
import torch


def build_sana(vision_tower_cfg, **kwargs):
    sana = SanaTransformer2DModel.from_pretrained(vision_tower_cfg.diffusion_name_or_path, subfolder="transformer", torch_dtype=torch.bfloat16)
    return sana


def build_vae(vision_tower_cfg, **kwargs):
    vae = AutoencoderDC.from_pretrained(vision_tower_cfg.diffusion_name_or_path, subfolder="vae", torch_dtype=torch.bfloat16)
    return vae


