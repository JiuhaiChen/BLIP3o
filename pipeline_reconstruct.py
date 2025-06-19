from dataclasses import dataclass
from typing import List, Optional

from PIL import Image
import numpy as np
import torch
from torchvision import transforms as TF
from tqdm import tqdm

from diffusers import DiffusionPipeline
from diffusers.utils import BaseOutput

from diffusers import UNet2DConditionModel, EulerDiscreteScheduler, AutoencoderKL
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import CLIPImageProcessor
from transformers import AutoModelForCausalLM, AutoTokenizer

EVA_IMAGE_SIZE = 448
OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)
DEFAULT_IMG_PLACEHOLDER = "[<IMG_PLH>]"

@dataclass
class EmuVisualGenerationPipelineOutput(BaseOutput):
    image: Image.Image
    nsfw_content_detected: Optional[bool]


class EmuVisualGenerationPipeline(DiffusionPipeline):

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        multimodal_encoder: AutoModelForCausalLM,
        scheduler: EulerDiscreteScheduler,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        feature_extractor: CLIPImageProcessor,
        safety_checker: StableDiffusionSafetyChecker,
        eva_size=EVA_IMAGE_SIZE,
        eva_mean=OPENAI_DATASET_MEAN,
        eva_std=OPENAI_DATASET_STD,
    ):
        super().__init__()
        self.register_modules(
            tokenizer=tokenizer,
            multimodal_encoder=multimodal_encoder,
            scheduler=scheduler,
            unet=unet,
            vae=vae,
            feature_extractor=feature_extractor,
            safety_checker=safety_checker,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        self.transform = TF.Compose([
            TF.Resize((eva_size, eva_size), interpolation=TF.InterpolationMode.BICUBIC),
            TF.ToTensor(),
            TF.Normalize(mean=eva_mean, std=eva_std),
        ])

        self.negative_prompt = {}

    def device(self, module):
        return next(module.parameters()).device

    def dtype(self, module):
        return next(module.parameters()).dtype

    @torch.no_grad()
    def __call__(
        self,
        inputs: List[Image.Image | str] | str | Image.Image,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.,
        crop_info: List[int] = [0, 0],
        original_size: List[int] = [1024, 1024],
    ):
        if not isinstance(inputs, list):
            inputs = [inputs]

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        device = self.device(self.unet)
        dtype = self.dtype(self.unet)

        do_classifier_free_guidance = guidance_scale > 1.0

        # 1. Encode input prompt
        prompt_embeds = self._prepare_and_encode_inputs(
            inputs,
            do_classifier_free_guidance,
        ).to(dtype).to(device)
        batch_size = prompt_embeds.shape[0] // 2 if do_classifier_free_guidance else prompt_embeds.shape[0]

        unet_added_conditions = {}
        time_ids = torch.LongTensor(original_size + crop_info + [height, width]).to(device)
        if do_classifier_free_guidance:
            unet_added_conditions["time_ids"] = torch.cat([time_ids, time_ids], dim=0)
        else:
            unet_added_conditions["time_ids"] = time_ids
        unet_added_conditions["text_embeds"] = torch.mean(prompt_embeds, dim=1)

        # 2. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 3. Prepare latent variables
        shape = (
            batch_size,
            self.unet.config.in_channels,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        latents = torch.randn(shape, device=device, dtype=dtype)
        latents = latents * self.scheduler.init_noise_sigma

        # 4. Denoising loop
        for t in tqdm(timesteps):
            # expand the latents if we are doing classifier free guidance
            # 2B x 4 x H x W
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=unet_added_conditions,
            ).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # 5. Post-processing
        images = self.decode_latents(latents)

        # 6. Run safety checker
        images, has_nsfw_concept = self.run_safety_checker(images)

        # 7. Convert to PIL
        images = self.numpy_to_pil(images)
        return EmuVisualGenerationPipelineOutput(
            image=images[0],
            nsfw_content_detected=None if has_nsfw_concept is None else has_nsfw_concept[0],
        )

    def _prepare_and_encode_inputs(
        self,
        inputs: List[str | Image.Image],
        do_classifier_free_guidance: bool = False,
        placeholder: str = DEFAULT_IMG_PLACEHOLDER,
    ):
        device = self.device(self.multimodal_encoder.model.gen_vision_tower)
        dtype = self.dtype(self.multimodal_encoder.model.gen_vision_tower)

        has_image, has_text = False, False
        text_prompt, image_prompt = "", []
        for x in inputs:
            if isinstance(x, str):
                has_text = True
                text_prompt += x
            else:
                has_image = True
                text_prompt += placeholder
                image_prompt.append(self.transform(x))

        if len(image_prompt) == 0:
            image_prompt = None
        else:
            image_prompt = torch.stack(image_prompt)
            image_prompt = image_prompt.type(dtype).to(device)

        if has_image and not has_text:
            prompt = self.multimodal_encoder.encode_image(images=image_prompt)
            if do_classifier_free_guidance:
                key = "[NULL_IMAGE]"
                if key not in self.negative_prompt:
                    negative_image = torch.zeros_like(image_prompt)
                    self.negative_prompt[key] = self.multimodal_encoder.encode_image(images=negative_image)
                prompt = torch.cat([prompt, self.negative_prompt[key]], dim=0)
                # breakpoint()
        else:
            prompt = self.multimodal_encoder.generate_image(text=[text_prompt], image=image_prompt, tokenizer=self.tokenizer)
            if do_classifier_free_guidance:
                key = ""
                if key not in self.negative_prompt:
                    self.negative_prompt[key] = self.multimodal_encoder.generate_image(text=[""], tokenizer=self.tokenizer)
                prompt = torch.cat([prompt, self.negative_prompt[key]], dim=0)

        return prompt

    def decode_latents(self, latents: torch.Tensor) -> np.ndarray:
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    def numpy_to_pil(self, images: np.ndarray) -> List[Image.Image]:
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    def run_safety_checker(self, images: np.ndarray):
        if self.safety_checker is not None:
            device = self.device(self.safety_checker)
            dtype = self.dtype(self.safety_checker)
            safety_checker_input = self.feature_extractor(self.numpy_to_pil(images), return_tensors="pt").to(device)
            images, has_nsfw_concept = self.safety_checker(
                images=images, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        else:
            has_nsfw_concept = None
        return images, has_nsfw_concept
