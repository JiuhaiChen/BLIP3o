import argparse
import json
import os
import numpy as np
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid
from torchvision.transforms import ToTensor
import cv2
from diffusers import DiffusionPipeline, AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import DDIMScheduler
import shortuuid
from blip3o.constants import *
from blip3o.conversation import conv_templates, SeparatorStyle
from blip3o.model.builder import load_pretrained_model
from blip3o.utils import disable_torch_init
from blip3o.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
import math
import requests
import random
from blip3o.conversation import conv_templates, SeparatorStyle

def set_global_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def add_template(prompt):
    conv = conv_templates['qwen'].copy()
    conv.append_message(conv.roles[0], prompt[0])
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    return [prompt]

torch.set_grad_enabled(False)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Huggingface model name"
    )
    parser.add_argument(
        "--prompt_template",
        type=str,
        default="qwen",
        help="Template format"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        help="dir to write results to",
        default="outputs"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=4,
        help="number of samples",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        nargs="?",
        const="ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face",
        default=None,
        help="negative prompt for guidance"
    )
    parser.add_argument(
        "--H",
        type=int,
        default=None,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=None,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=3.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="how many samples can be produced simultaneously",
    )
    parser.add_argument(
        "--skip_grid",
        action="store_true",
        help="skip saving grid",
    )
    parser.add_argument("--index", type=int, default=0, help="Chunk index to process (0-indexed)")
    parser.add_argument("--n_chunks", type=int, default=1, help="Total number of chunks")
    opt = parser.parse_args()
    return opt

def main(opt):
    model_name = opt.model
    diffusion_path = model_name + "/diffusion-decoder"

    outdir = f"{model_name}/geneval_{opt.prompt_template}"
    os.makedirs(outdir, exist_ok=True)
    prompt_template = opt.prompt_template
    disable_torch_init()
    tokenizer, multi_model, context_len = load_pretrained_model(model_name)

    pipe = DiffusionPipeline.from_pretrained(
        diffusion_path,
        custom_pipeline="pipeline_llava_gen",
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        variant="bf16",
        multimodal_encoder=multi_model,
        tokenizer=tokenizer,
        safety_checker=None
    )
    
    device_id = 0
    pipe.vae.to(f'cuda:{device_id}')
    pipe.unet.to(f'cuda:{device_id}')

    # Load all prompts
    with open('geneval_prompt.jsonl') as fp:
        metadatas = [json.loads(line) for line in fp]

    # Split the data into chunks: each instance will process every n_chunks-th entry
    metadatas = metadatas[opt.index::opt.n_chunks]
    print(f"Processing chunk {opt.index} out of {opt.n_chunks} total chunks, {len(metadatas)} samples assigned.")

    for index, metadata in enumerate(metadatas):
        set_global_seed(seed=42)
        outpath = os.path.join(outdir, f"{metadata['index']}")
        os.makedirs(outpath, exist_ok=True)
        prompt = metadata['prompt']

        prompt = [f"Please generate image based on the following caption: {prompt}"]
        if "qwen" in prompt_template:
            prompt = add_template(prompt)
        print(f"Prompt ({index: >3}/{len(metadatas)}): '{prompt}'")

        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)
        with open(os.path.join(outpath, "metadata.jsonl"), "w") as fp:
            json.dump(metadata, fp)

        sample_count = 0
        batch_size = opt.batch_size
        n_rows = opt.batch_size
        with torch.no_grad():
            all_samples = list()
            for n in trange((opt.n_samples + batch_size - 1) // batch_size, desc="Sampling"):
                
                gen_img = pipe(prompt, guidance_scale=3.0).image

                samples = [gen_img]
                for sample in samples:
                    sample.save(os.path.join(sample_path, f"{sample_count:05}.png"))
                    sample_count += 1
                if not opt.skip_grid:
                    all_samples.append(torch.stack([ToTensor()(sample) for sample in samples], 0))

            if not opt.skip_grid:
                grid = torch.stack(all_samples, 0)
                grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                grid = make_grid(grid, nrow=n_rows)
                grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                grid = Image.fromarray(grid.astype(np.uint8))
                grid.save(os.path.join(outpath, f'grid.png'))
                del grid
        del all_samples

    print("Done.")

if __name__ == "__main__":
    opt = parse_args()
    main(opt)






