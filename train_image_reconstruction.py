from dataclasses import dataclass
import io
import math
import PIL
from PIL import PngImagePlugin
import torch
from torch import nn
import glob
import os
from datasets import load_dataset, Image, concatenate_datasets
from torchvision.transforms import v2

import transformers
from diffusers import AutoencoderDC, FlowMatchEulerDiscreteScheduler, SanaTransformer2DModel
from transformers import Trainer, PreTrainedModel, PretrainedConfig, SiglipImageProcessor, SiglipVisionModel
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from diffusers.models.normalization import RMSNorm
from typing import Optional, Union, List
import numpy as np
from tqdm import tqdm


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


PIL.Image.MAX_IMAGE_PIXELS = None
PngImagePlugin.MAX_TEXT_CHUNK = 100 * (1024**2)


def numpy_to_pil(images: np.ndarray):
    """
    Convert a NumPy array of shape (batch, height, width, channels) to a list of PIL Images.
    """
    pil_images = []
    for img in images:
        img_uint8 = (img * 255).round().astype("uint8")
        if img_uint8.shape[2] == 1:
            img_uint8 = img_uint8[..., 0]
        pil_images.append(PIL.Image.fromarray(img_uint8))
    return pil_images


def randn_tensor(shape, generator=None, device=None, dtype=None):
    """
    Generate a tensor of random normal noise.
    """
    if isinstance(generator, list):
        generator = generator[0]
    if generator is None:
        return torch.randn(shape, device=device, dtype=dtype)
    return torch.randn(shape, generator=generator, device=device, dtype=dtype)


class ProcessorWrapper:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, tensor):
        return self.processor(tensor, return_tensors="pt")["pixel_values"].squeeze(0)


@dataclass
class ModelArguments:
    _gradient_checkpointing: bool = True
    encoder_id: str = "google/siglip2-so400m-patch16-512"
    vae_downsample_f: int = 32
    vae_id: str = "Efficient-Large-Model/Sana_1600M_512px_diffusers"
    noise_scheduler_id: str = "Efficient-Large-Model/Sana_1600M_512px_diffusers"
    scheduler_id: str = "Efficient-Large-Model/Sana_1600M_512px_diffusers"
    diffusion_id: str = "Efficient-Large-Model/Sana_1600M_512px_diffusers"
    target_image_size: int = 512
    num_pooled_tokens: int = 64


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: str = "/fsx/home/jiuhai.chen/BLIP3o/"
    eval_strategy: str = "no"
    per_device_train_batch_size: int = 64
    optim: str = "adamw_torch"
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 0.5
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 5000
    logging_dir: str = "log"
    logging_steps: int = 5
    save_steps: int = 1000
    save_total_limit: int = 1
    restore_callback_states_from_checkpoint: bool = True
    seed: int = 42
    data_seed: int = 42
    bf16: bool = True
    tf32: bool = True
    dataloader_num_workers: int = 4
    dataloader_persistent_workers: bool = False
    dataloader_pin_memory: bool = True
    dataloader_drop_last: bool = True
    remove_unused_columns: bool = False
    run_name: str = "test"
    report_to: str = "none"
    ddp_find_unused_parameters: bool = False
    resume_from_checkpoint: str = None


class DiffusionDecoderConfig(PretrainedConfig):
    model_type = "diffusion_decoder"

    def __init__(
        self,
        _gradient_checkpointing: bool = True,
        encoder_id: str = "google/siglip2-so400m-patch16-512",
        vae_downsample_f: int = 32,
        vae_id: str = "Efficient-Large-Model/Sana_1600M_512px_diffusers",
        noise_scheduler_id: str = "Efficient-Large-Model/Sana_1600M_512px_diffusers",
        scheduler_id: str = "Efficient-Large-Model/Sana_1600M_512px_diffusers",
        diffusion_id: str = "Efficient-Large-Model/Sana_1600M_512px_diffusers",
        target_image_size: int = 512,
        num_pooled_tokens: int = 64,
        input_size: int = 64,
        in_channels: int = 32,
        **kwargs  
    ):

        super().__init__(**kwargs)

        self._gradient_checkpointing = _gradient_checkpointing
        self.encoder_id = encoder_id
        self.vae_downsample_f = vae_downsample_f
        self.vae_id = vae_id
        self.noise_scheduler_id = noise_scheduler_id
        self.scheduler_id = scheduler_id
        self.diffusion_id = diffusion_id
        self.target_image_size = target_image_size
        self.num_pooled_tokens = num_pooled_tokens
        self.input_size = input_size
        self.in_channels = in_channels


class DiffusionDecoder(PreTrainedModel):
    config_class = DiffusionDecoderConfig

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.config = config
        self.processor = SiglipImageProcessor.from_pretrained(config.encoder_id)
        self.source_image_size = min(self.processor.size["height"], self.processor.size["width"])
        self.source_transform = v2.Compose(
            [
                v2.Resize(self.source_image_size),
                v2.CenterCrop(self.source_image_size),
                ProcessorWrapper(self.processor),
            ]
        )

        self.processor = SiglipImageProcessor.from_pretrained(config.encoder_id)
        self.encoder = SiglipVisionModel.from_pretrained(config.encoder_id, attn_implementation="sdpa")
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            config.noise_scheduler_id, subfolder="scheduler"
        )
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(config.scheduler_id, subfolder="scheduler")

        self.vae = AutoencoderDC.from_pretrained(config.vae_id, subfolder="vae")
        self.transformer = SanaTransformer2DModel.from_pretrained(
            config.diffusion_id, subfolder="transformer", torch_dtype=torch.bfloat16
        )
        self.connector_in_dim = self.encoder.config.hidden_size
        self.connector_out_dim = self.transformer.config.caption_channels
        norm = RMSNorm(self.connector_out_dim, eps=1e-5, elementwise_affine=True)
        with torch.no_grad():
            norm.weight.fill_(math.sqrt(5.5))
        self.connector = nn.Sequential(
            nn.Linear(self.connector_in_dim, self.connector_out_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.connector_out_dim, self.connector_out_dim),
            norm,
        )

        self.vae.eval()
        self.vae.requires_grad_(False)
        self.encoder.eval()
        self.encoder.requires_grad_(False)
        if self.config._gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()

    def get_sigmas(self, timesteps, device, n_dim=4, dtype=torch.float32):
        sigmas = self.noise_scheduler.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = self.noise_scheduler.timesteps.to(device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def forward(self, x_target, x_source=None, **kwargs):
        latents = self.vae.encode(x_target).latent
        if "shift_factor" in self.vae.config and self.vae.config.shift_factor is not None:
            latents = latents - self.vae.config.shift_factor
        latents = latents * self.vae.config.scaling_factor
        bsz = latents.shape[0]

        x_source = self.encoder(x_source, output_attentions=False, output_hidden_states=True).hidden_states[-2]
        ## pooling the SigLIP feature: [bsz, 32, 32, 1152] to [bsz, 8, 8, 1152]
        size = int(x_source.size(1) ** 0.5)
        x_source = x_source.view(-1, size, size, x_source.size(-1))
        x_source = (
            nn.functional.adaptive_avg_pool2d(
                x_source.permute(0, 3, 1, 2), int(self.config.num_pooled_tokens**0.5)
            )
            .permute(0, 2, 3, 1)
            .view(x_source.size(0), -1, x_source.size(-1))
        )

        noise = torch.randn_like(latents, device=latents.device)

        weighting_scheme = "uniform"
        u = compute_density_for_timestep_sampling(
            weighting_scheme=weighting_scheme,
            batch_size=bsz,
            logit_mean=0.0,
            logit_std=1.0,
            mode_scale=1.29,
        )
        indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
        timesteps = self.noise_scheduler.timesteps[indices].to(device=latents.device)

        sigmas = self.get_sigmas(timesteps, latents.device, n_dim=latents.ndim, dtype=latents.dtype)
        noisy_latents = (1.0 - sigmas) * latents + sigmas * noise

        model_pred = self.transformer(
            hidden_states=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=self.connector(x_source),
            encoder_attention_mask=None,
        ).sample

        target = noise - latents
        weighting = compute_loss_weighting_for_sd3(weighting_scheme=weighting_scheme, sigmas=sigmas)
        loss = torch.mean(
            (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
            1,
        )
        loss = loss.mean()

        return {"loss": loss}


class DiffusionDecoderPipeline(DiffusionDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def decode_latents(self, latents):
        latents = latents / self.vae.config.scaling_factor
        if "shift_factor" in self.vae.config and self.vae.config.shift_factor is not None:
            latents = latents + self.vae.config.shift_factor
        samples = self.vae.decode(latents).sample
        samples = (samples / 2 + 0.5).clamp(0, 1)
        samples = samples.cpu().permute(0, 2, 3, 1).float().numpy()
        samples = numpy_to_pil(samples)
        return samples

    def __call__(
        self,
        x_source=None,
        guidance_scale: float = 3.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 30,
        num_images_per_prompt: int = 1,
        enable_progress_bar=False,
        **kwargs,
    ):
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype


        if not isinstance(x_source[0], torch.Tensor):
            x_source = [self.source_transform(img).to(device=device, dtype=dtype) for img in x_source]
        bsz = len(x_source)


        x_source_null = [
            self.source_transform(PIL.Image.new("RGB", (img.shape[1], img.shape[2])))
            .to(device=device, dtype=dtype)
            for img in x_source
        ]
        x_source = torch.stack(x_source_null + x_source, dim=0)


        latent_size = self.transformer.config.sample_size
        latent_channels = self.vae.config.latent_channels
        latents = randn_tensor(
            shape=(bsz * num_images_per_prompt, latent_channels, latent_size, latent_size),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )


        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        self.scheduler.set_timesteps(num_inference_steps, sigmas=sigmas)


        x_source_input = self.encoder(
            x_source, output_attentions=False, output_hidden_states=True
        ).hidden_states[-2]
        size = int(x_source_input.size(1) ** 0.5)
        x_source_input = x_source_input.view(-1, size, size, x_source_input.size(-1))
        x_source_input = (
            nn.functional.adaptive_avg_pool2d(
                x_source_input.permute(0, 3, 1, 2), int(self.config.num_pooled_tokens**0.5)
            )
            .permute(0, 2, 3, 1)
            .view(x_source_input.size(0), -1, x_source_input.size(-1))
        )
        x_source_input = x_source_input.to(device=device, dtype=dtype)


        for t in tqdm(self.scheduler.timesteps, desc="Sampling images", disable=not enable_progress_bar):
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = latent_model_input.to(device=device, dtype=dtype)

            if hasattr(self.scheduler, "scale_model_input"):
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

           

            model_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=t.unsqueeze(0).expand(latent_model_input.shape[0]).to(latents.device),
                encoder_hidden_states=self.connector(x_source_input),
                encoder_attention_mask=None,
            ).sample


            model_pred_uncond, model_pred = model_pred.chunk(2)
            model_pred = model_pred_uncond + guidance_scale * (model_pred - model_pred_uncond)


            latents = self.scheduler.step(model_pred, t, latents).prev_sample


        samples = self.decode_latents(latents.to(self.vae.dtype))
        return samples


if __name__ == "__main__":
    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()
    model = DiffusionDecoder(config=DiffusionDecoderConfig(**model_args.__dict__))


    list_data_dict = []
    data_files = glob.glob(os.path.join('/your/data/folder/', "*.tar"))
    
    train_dataset = load_dataset("webdataset", data_files=data_files, split="train", num_proc=128)    
    train_dataset = train_dataset.rename_column("jpg", "image")
    train_dataset = train_dataset.remove_columns(
        [col for col in train_dataset.column_names if col not in (["image"])]
    )
    list_data_dict.append(train_dataset)

    # for _ in range(100000):
    #     list_data_dict.append(train_dataset)


    if len(list_data_dict) > 1:
        list_data_dict = concatenate_datasets(list_data_dict)
    else:
        list_data_dict = list_data_dict[0]
    list_data_dict = list_data_dict.shuffle(seed=42)


    target_transform = v2.Compose(
        [
            v2.Resize(model_args.target_image_size),
            v2.CenterCrop(model_args.target_image_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.5], [0.5]),
        ]
    )
    source_transform = model.source_transform


    def i2i_process_fn(batch):
        images = batch["image"]
        for i in range(len(images)):
            try:
                images[i] = PIL.Image.open(
                    io.BytesIO(images[i]["bytes"]) if images[i]["bytes"] is not None else images[i]["path"]
                ).convert("RGB")
            except:
                images[i] = None
        batch["x_target"] = [
            target_transform(image) if image is not None else None for image in images
        ]
        rand_probs = torch.rand((len(images), 1))
        null_image_mask = rand_probs <= 0.1
        images = [
            PIL.Image.new("RGB", (image.width, image.height))
            if (image is not None and null_image_mask[i])
            else image
            for i, image in enumerate(images)
        ]
        batch["x_source"] = [[image] if image is not None else None for image in images]
        keys_to_delete = [key for key in list(batch.keys()) if key not in (["x_target", "x_source"])]
        for key in keys_to_delete:
            del batch[key]
        return batch

    list_data_dict = list_data_dict.cast_column("image", Image(decode=False))
    list_data_dict.set_transform(i2i_process_fn)
    list_data_dict = list_data_dict.shuffle(seed=training_args.data_seed)




    def collate_fn(batch):
        none_idx = [i for i, example in enumerate(batch) if example["x_target"] is None]
        if len(none_idx) > 0:
            batch = [example for i, example in enumerate(batch) if i not in none_idx]
        return_dict = {"x_target": torch.stack([example["x_target"] for example in batch])}
        x_source = [example["x_source"] if "x_source" in example else None for example in batch]
        return_dict["x_source"] = torch.stack(
            [source_transform(image[0]) for image in x_source], dim=0
        )
        return return_dict




    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=list_data_dict,
        eval_dataset=None,
        data_collator=collate_fn,
    )
    train_output = trainer.train()
