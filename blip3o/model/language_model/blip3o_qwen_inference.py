from typing import List, Optional, Tuple, Union, Dict
import torch
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from blip3o.model.blip3o_arch import blip3oMetaModel, blip3oMetaForCausalLM
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLModel, Qwen2_5_VLForConditionalGeneration
from blip3o.constants import UND_IMAGE_TOKEN_IDX


from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import numpy_to_pil
import numpy as np
from diffusers.models import AutoencoderKL
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler


class blip3oQwenConfig(Qwen2_5_VLConfig):
    model_type = "blip3o_qwen_inference"


class blip3oQwenModel(blip3oMetaModel, Qwen2_5_VLModel):
    config_class = blip3oQwenConfig

    def __init__(self, config: Qwen2_5_VLConfig):
        super(blip3oQwenModel, self).__init__(config)


class blip3oQwenForInferenceLM(Qwen2_5_VLForConditionalGeneration, blip3oMetaForCausalLM):
    config_class = blip3oQwenConfig

    def __init__(self, config):
        Qwen2_5_VLForConditionalGeneration.__init__(self, config)
        config.model_type = "blip3o_qwen_inference"

        self.model = blip3oQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model


    @torch.no_grad()
    def generate_image(
        self,
        text: List[str],
        tokenizer: AutoTokenizer,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        max_var: Optional[float] = None,
        # placeholder: str = DEFAULT_IMG_PLACEHOLDER,
    ):  
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained("Alpha-VLLM/Lumina-Next-SFT-diffusers", subfolder="scheduler")


        N_QUERY = self.get_n_query()            
        inputs = tokenizer(text, padding="longest", return_tensors="pt")
        device = self.get_model().device
        attention_mask = inputs.attention_mask.to(device)
        input_ids = inputs.input_ids.to(device)  # B x N
        input_ids = torch.cat([input_ids, torch.tensor([[151665]]).to(device)], dim=1)
        # breakpoint()


        text_embeds = self.get_model().embed_tokens(input_ids)
        latent_queries = self.get_model().latent_queries.repeat(text_embeds.shape[0], 1, 1)


        if pixel_values is not None:
            und_image_idx = (input_ids == UND_IMAGE_TOKEN_IDX)
            pixel_values = pixel_values.type(self.visual.dtype)
            und_image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
            text_embeds[und_image_idx] = und_image_embeds.to(text_embeds.device)[:und_image_idx.sum(), :]


        text_embeds = torch.cat([text_embeds, latent_queries], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(latent_queries[:, :, 0])], dim=1)


        outputs = self.model(
            inputs_embeds=text_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states[-1][:,-N_QUERY:,:]
        img_hidden_states = hidden_states 
        output_img = self.sample_images(img_hidden_states, scheduler)
        output_img = output_img.view(1, 1792, -1).permute(0,2,1).contiguous()

        return output_img

    def sample_images(
        self,
        img_hidden_states,
        scheduler,
        guidance_scale: float = 3.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 30,
        num_images_per_prompt: int = 1,
        return_tensor=False,
        **kwargs,
    ):
        
        device = img_hidden_states.device
        dtype = img_hidden_states.dtype


        img_hidden_states_null = torch.zeros_like(img_hidden_states, device=device, dtype=dtype)
        img_hidden_states_input = torch.cat([img_hidden_states_null, img_hidden_states], 0)

        batch_size = img_hidden_states.shape[0]
        latent_size = self.get_model().dit.config.input_size
        latent_channels = self.get_model().dit.config.in_channels

        latents = randn_tensor(
            shape=(batch_size * num_images_per_prompt, latent_channels, latent_size, latent_size),
            generator=generator,
            device=device,
            dtype=dtype,
        )

        # set step values
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        scheduler.set_timesteps(num_inference_steps, sigmas=sigmas)

        # Repeat z_latents and conditions for each image per prompt
        img_hidden_states_input = img_hidden_states_input.repeat_interleave(num_images_per_prompt, dim=0)

        for t in scheduler.timesteps:
            latent_model_input = latents.repeat(2, 1, 1, 1)
            if hasattr(scheduler, "scale_model_input"):
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            # predict noise model_output
            noise_pred = self.get_model().dit(
                x=latent_model_input,
                timestep=t.unsqueeze(0).expand(latent_model_input.shape[0]).to(latent_model_input.device, torch.long),
                z_latents=img_hidden_states_input,
            )

            # perform guidance
            noise_pred_uncond, noise_pred = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)

            # compute previous image: x_t -> x_t-1
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        # samples = self.decode_latents(latents, return_tensor=return_tensor)
        # breakpoint()
        return latents

    def decode_latents(self, latents, normalize=True, return_tensor=False):
        if isinstance(self.get_model().vae, AutoencoderKL):
            latents = latents / self.get_model().vae.config.scaling_factor
            if self.get_model().vae.config.shift_factor is not None:
                latents = latents + self.get_model().vae.config.shift_factor
            latents = latents.to(dtype=torch.float32)
            samples = self.get_model().vae.decode(latents).sample
        else:
            samples = self.get_model().vae.decode(latents)
        if normalize:
            samples = (samples / 2 + 0.5).clamp(0, 1)
        else:
            samples = samples.clamp(-1, 1)
        if return_tensor:
            return samples
        samples = samples.cpu().permute(0, 2, 3, 1).float().numpy()
        samples = numpy_to_pil(samples)
        return samples

    def prepare_and_encode_inputs(
        self,
        inputs: List[str | Image.Image],
        tokenizer: AutoTokenizer,
        do_classifier_free_guidance: bool = False,
    ):
        # pdb.set_trace()
        device = self.get_model().device
        dtype = self.get_model().dtype

        has_image, has_text = False, False
        text_prompt, image_prompt = "", []
        img_processor = self.get_vision_tower().image_processor
        negative_prompt = {}

        for x in inputs:
            if isinstance(x, str):
                has_text = True
                text_prompt += x
            else:
                has_image = True
                text_prompt += DEFAULT_IMAGE_TOKEN
                image_prompt.append(img_processor.preprocess(x, return_tensors='pt')['pixel_values'])
        # pdb.set_trace()
        if len(image_prompt) == 0:
            image_prompt = None
        else:
            image_prompt = torch.cat(image_prompt)
            image_prompt = image_prompt.type(dtype).to(device)

        if has_image and not has_text:
            prompt = self.encode_image(image_prompt)
            # pdb.set_trace()
            if do_classifier_free_guidance:
                key = "[NULL_IMAGE]"
                if key not in negative_prompt:
                    negative_image = torch.zeros_like(image_prompt)
                    negative_prompt[key] = self.encode_image(negative_image)
                prompt = torch.cat([prompt, negative_prompt[key]], dim=0)
        else:
            prompt = self.generate_image(text=[text_prompt], image=image_prompt, tokenizer=tokenizer)
            if do_classifier_free_guidance:
                key = ""
                if key not in negative_prompt:
                    negative_prompt[key] = self.generate_image(text=[""], tokenizer=tokenizer)
                prompt = torch.cat([prompt, negative_prompt[key]], dim=0)
        
        gen_pooling = self.get_gen_pooling()
        n_query = self.get_n_query()
        num_img, _, c = prompt.shape
        if 'pool2d' in gen_pooling and has_text and not 'early' in gen_pooling:
            stride = int(gen_pooling.split('_')[1])
            sqrt_n = int(n_query**0.5)
            prompt = prompt.permute(0, 2, 1).reshape(num_img, -1, sqrt_n, sqrt_n)
            prompt = F.avg_pool2d(prompt, kernel_size=(stride, stride), stride=stride)
            prompt = prompt.reshape(num_img, c, -1).permute(0,2,1)
        return prompt


    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("blip3o_qwen_inference", blip3oQwenConfig)
AutoModelForCausalLM.register(blip3oQwenConfig, blip3oQwenForInferenceLM)
