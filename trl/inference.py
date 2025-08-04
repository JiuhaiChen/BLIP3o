from dataclasses import dataclass
import torch
from PIL import Image
from transformers import AutoTokenizer
from blip3o.model import *
import os


@dataclass
class T2IConfig:
    model_path: str = "/fsx/home/jiuhai.chen/BLIP3o-NEXT/models/debug"
    device: str = "cuda:0"
    dtype: torch.dtype = torch.bfloat16
    # generation config
    scale: int = 0  
    seq_len: int = 729  
    top_p: float = 0.95
    top_k: int = 1200

class TextToImageInference:
    def __init__(self, config: T2IConfig):
        self.config = config
        self.device = torch.device(config.device)
        self._load_models()
        
    def _load_models(self):
        self.model = blip3oQwenForInferenceLM.from_pretrained(self.config.model_path, torch_dtype=self.config.dtype).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)

    def generate_image(self, prompt: str) -> Image.Image:

        batch_messages = []


        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Please generate image based on the following caption: {prompt}"}
        ]
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True)
        input_text += f"<im_start><S{self.config.scale}>"
        
        batch_messages.append(input_text)

        # tokenize as a batch
        inputs = self.tokenizer(batch_messages, return_tensors="pt", padding=True, truncation=True, padding_side="left")
    
        gen_ids, output_image = self.model.generate_images(
            inputs.input_ids.to(self.device),
            inputs.attention_mask.to(self.device),
            max_new_tokens=self.config.seq_len,
            do_sample=True,
            top_p=self.config.top_p,
            top_k=self.config.top_k)

        print(output_image)
        return output_image[0]


def main():
    config = T2IConfig()
    inference = TextToImageInference(config)

    prompts = [
        'A cute cat'
    ]   

    output_dir = "BLIP3o-NEXT"
    os.makedirs(output_dir, exist_ok=True)

    for idx, prompt in enumerate(prompts):
        image_sana = inference.generate_image(prompt)
        save_path = os.path.join(output_dir, f"blip3o_next_{idx:02d}.png")
        image_sana.save(save_path)
        print(f"Saved: {save_path}")



if __name__ == "__main__":
    main()  

