from diffusers import DiffusionPipeline, AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import DDIMScheduler
import numpy as np
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, AutoModel
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
import torch
import pdb
import copy
import sys
import argparse
import os
import json
from tqdm import tqdm
import shortuuid
from blip3o.constants import *
from blip3o.conversation import conv_templates, SeparatorStyle
from blip3o.model.builder import load_pretrained_model
from blip3o.utils import disable_torch_init
from blip3o.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
import math
import requests
from blip3o.conversation import conv_templates, SeparatorStyle
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import base64
from io import BytesIO
from qwen_vl_utils import process_vision_info

import re, random

model_path = sys.argv[1]
diffusion_path = model_path + "/diffusion-decoder"



processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")


device_1 = 0


disable_torch_init()
model_path = os.path.expanduser(model_path)
model_name = get_model_name_from_path(model_path)
tokenizer, multi_model, context_len = load_pretrained_model(model_path, None, model_name)
multi_model.model.gen_vision_tower.vision_tower_pretrained = # path to EVA CLIP encoder
multi_model.model.gen_vision_tower.load_model()
multi_model.model.gen_vision_tower.to(f'cuda:{device_1}')




pipe = DiffusionPipeline.from_pretrained(
   diffusion_path,
   custom_pipeline="pipeline_reconstruct",
   torch_dtype=torch.bfloat16,
   use_safetensors=True,
   variant="bf16",
   multimodal_encoder=multi_model,
   tokenizer=tokenizer,
   safety_checker=None
)


pipe.vae.to(f'cuda:{device_1}')
pipe.unet.to(f'cuda:{device_1}')



def create_image_grid(images, rows, cols):
    """Creates a grid of images and returns a single PIL Image."""

    assert len(images) == rows * cols

    width, height = images[0].size
    grid_width = width * cols
    grid_height = height * rows

    grid_image = Image.new('RGB', (grid_width, grid_height))

    for i, image in enumerate(images):
        x = (i % cols) * width
        y = (i // cols) * height
        grid_image.paste(image, (x, y))

    return grid_image


def add_template(prompt):
   conv = conv_templates['qwen'].copy()
   conv.append_message(conv.roles[0], prompt[0])
   conv.append_message(conv.roles[1], None)
   prompt = conv.get_prompt()
   return [prompt]



def set_global_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



image = Image.open('dog.jpg').convert('RGB') # input your image path here
gen_img = pipe([image], guidance_scale=3.0)
gen_img.image.save(f"reconstruct.png")

