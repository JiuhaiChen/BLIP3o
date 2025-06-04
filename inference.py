import torch
from PIL import Image
from train_image_reconstruction import (
    DiffusionDecoderPipeline,
    DiffusionDecoderConfig,
)


checkpoint_path = "/your/model/path"



pipe = DiffusionDecoderPipeline.from_pretrained(checkpoint_path)
pipe.to(device="cuda:1").to(dtype=torch.float32)


## images you try to reconstruct
image_paths = [
    "your/image/1.png",
    "your/image/2.png",
    "your/image/3.png",
    "your/image/4.png",
]


source_images = [Image.open(p).convert("RGB") for p in image_paths]


with torch.no_grad():
    outputs = pipe(
        x_source=source_images,
        guidance_scale=3.0,
        num_inference_steps=30,
        num_images_per_prompt=1,      
        enable_progress_bar=True,     
    )


for idx, img in enumerate(outputs):
    img.save(f"./inference_out_{idx}.png")

