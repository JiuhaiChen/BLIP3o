import torch
from transformers import AutoTokenizer

from blip3o.model import blip3oQwenForCausalLM
from blip3o.utils import rank0_print


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", torch_dtype="float16", attn_implementation="flash_attention_2", customized_config=None, overwrite_config=None, **kwargs):
    kwargs["device_map"] = device_map
    kwargs.pop("multimodal")

    if customized_config is not None:
        kwargs["config"] = customized_config

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    from blip3o.model.language_model.blip3o_qwen import blip3oQwenConfig

    breakpoint()
    if overwrite_config is not None:
        blip3o_cfg = blip3oQwenConfig.from_pretrained(model_path)
        rank0_print(f"Overwriting config with {overwrite_config}")
        for k, v in overwrite_config.items():
            setattr(blip3o_cfg, k, v)
        model = blip3oQwenForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, attn_implementation=attn_implementation, config=blip3o_cfg, **kwargs)
    else:
        model = blip3oQwenForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, attn_implementation=attn_implementation, **kwargs)

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model(device_map=device_map)
    if device_map != "auto":
        vision_tower.to(device="cuda", dtype=torch.float16)
    image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    elif hasattr(model.config, "max_position_embeddings"):
        context_len = model.config.max_position_embeddings
    elif hasattr(model.config, "tokenizer_model_max_length"):
        context_len = model.config.tokenizer_model_max_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len