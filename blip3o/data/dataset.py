import copy
import glob
import io
import json
import math
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence
import pyarrow.parquet as pq
import torch
import transformers
import yaml
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision import transforms
from datasets import load_dataset, concatenate_datasets
from blip3o.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)
from blip3o.utils import rank0_print


ImageFile.LOAD_TRUNCATED_IMAGES = True


## target transform for sana
target_transform = v2.Compose(
    [
        v2.Resize(1024),
        v2.CenterCrop(1024),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.5], [0.5]),
    ]
    )


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def preprocess_multimodal(sources: Sequence[str], data_args) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            replace_token = DEFAULT_IMAGE_TOKEN
            # NOTE: only add im_start_end when image generation
            if data_args.mm_use_im_start_end and sentence['from'] == 'gpt':
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

            # For videoInstruct-100k noisy_data. TODO: Ask Yuanhan to clean the data instead of leaving the noise code here.
            sentence["value"] = sentence["value"].replace("QA_GT_caption_based_noisy", "")

    return sources


def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") -> Dict:
    # roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}
    roles = {"human": "user", "gpt": "assistant"}

    #tokenizer = copy.deepcopy(tokenizer)
    # When there is actually an image, we add the image tokens as a special token
    if 'image_token_index' not in globals():
        tokenizer.add_tokens(["<image>"], special_tokens=True)
        global image_token_index
        image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    # if has_image:
    #     tokenizer.add_tokens(["<image>"], special_tokens=True)

    # image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    im_start, im_end = tokenizer.additional_special_tokens_ids[:2]
    # unmask_tokens = ["<|im_start|>", "<|im_start|>", "\n"]
    unmask_tokens_idx =  [198, im_start, im_end]
    # nl_tokens = tokenizer("\n").input_ids

    # Reset Qwen chat templates so that it won't include system message every time we apply
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    # _system = tokenizer("system").input_ids + nl_tokens
    # _user = tokenizer("user").input_ids + nl_tokens
    # _assistant = tokenizer("assistant").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        # New version, use apply chat template
        # Build system message for each sentence
        input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])


        # target += [IGNORE_INDEX] * len(input_id)
        target += input_id

        for conv in source:
            # Make sure blip3o data can load
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role =  roles.get(role, role)
            
            conv = [{"role" : role, "content" : content}]
            encode_id = tokenizer.apply_chat_template(conv)
            input_id += encode_id
            if role in ["user", "system"]:
                # target += [IGNORE_INDEX] * len(encode_id)
                target += encode_id

            else:
                target += encode_id
        
        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == image_token_index:
                input_id[idx] = IMAGE_TOKEN_INDEX
        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,  
        labels=targets,  
    )



class LazySupervisedMixDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        data_path: str,
        data_args
    ):
        super(LazySupervisedMixDataset, self).__init__()

        self.data_args = data_args
        list_data_dict = []


        train_dataset = load_dataset("webdataset", data_files='/fsx/home/jiuhai.chen/soda/overfit.tar', split="train", num_proc=1, cache_dir='/fsx/sfr/data/jiuhai/webdataset')
        train_dataset = train_dataset.rename_column("jpg", "image")
        train_dataset = train_dataset.add_column('type', len(train_dataset) * ['T2I'])
        train_dataset = train_dataset.remove_columns([col for col in train_dataset.column_names if not col in (
            ["image", "txt", "type"])])
        print(f"finish loading image {len(train_dataset)}")
        list_data_dict.append(train_dataset)



        if len(list_data_dict) > 1:
            list_data_dict = concatenate_datasets(list_data_dict)
        else:
            list_data_dict = list_data_dict[0]
        list_data_dict = list_data_dict.shuffle(seed=42)


        rank0_print(f"Totoal number of training instance: {len(list_data_dict)}")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.modality = torch.tensor(0) # 0 is for und task, 1 is for gen task


    def __len__(self):
        return len(self.list_data_dict)


    def process_image(self, image):
        processor = self.data_args.image_processor
        image_size = image.size
        image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        return image, image_size, self.modality


    def process_target_image(self, image):
        image = target_transform(image)
        return image


    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(sum(len(conv["value"].split()) for conv in sample["conversations"]) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
            cur_len = cur_len if "image" in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:

        while True:
            sources = self.list_data_dict[i]


            if sources["type"] == "T2I":

                sources["conversations"] = [
                    {"from": "human", "value": f"Please generate image based on the following caption: {sources['txt']}"},
                    {"from": "gpt", "value": "<image>"},
                ]


            elif sources["type"] == "I2I":
                sources["conversations"] = [
                    {
                        "from": "human",
                        "value": f"<image>\nPlease reconstruct the given image.",
                    },
                    {"from": "gpt", "value": ""},
                ]

            else:
                raise ValueError("Unknown source type. Please check the 'type' in 'sources'.")

            if "image" in sources:

                if sources["type"] == "T2I" or sources["type"] == "I2I":
                    image_files = self.list_data_dict[i]["image"]

                if not isinstance(image_files, list):
                    image_files = [image_files]

                images = []

                for img in image_files:
                    try:
                        if sources["type"] == "T2I" or sources["type"] == "I2I":
                            img = img.convert("RGB")
                        else:
                            raise ValueError("Unknown source type. Please check the 'type' in 'sources'.")
                        images.append(img)
                    except Exception as e:
                        print(f"Error opening image {img}: {e}")
                        images = None
                        break  # Skip to the next image if there's an error


                ## test if can apply img_process 
                if not images is None:
                    try:
                        process_images = [self.process_image(f) for f in images]
                    except Exception as e:
                        print(f"Error wrong number of channels: {e}")
                        images = None


                # If no valid images were found, randomly pick another item
                if images is None:
                    print(sources)
                    print(f"warning false image!!!!!!")
                    i = random.randint(0, len(self.list_data_dict) - 1)
                    continue

                sources = preprocess_multimodal(copy.deepcopy([sources["conversations"]]), self.data_args)
            else:
                sources = copy.deepcopy([sources["conversations"]])

            data_dict = preprocess_qwen(sources, self.tokenizer, has_image=("image" in self.list_data_dict[i]))
            if isinstance(i, int):
                data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])


            # image exist in the data
            if "image" in self.list_data_dict[i]:
                data_dict["image"] = process_images
                data_dict["target_image"] = [self.process_target_image(f) for f in images]

            data_dict["ids"] = self.list_data_dict[i]["id"] if "id" in self.list_data_dict[i] else "unk"
            return data_dict



@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [_input_ids[: self.tokenizer.model_max_length] for _input_ids in input_ids]
        labels = [_labels[: self.tokenizer.model_max_length] for _labels in labels]
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0 # This gets the best result. Don't know why.
        input_ids = self.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = self.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        batch = dict(input_ids=input_ids, labels=labels.long() if labels.dtype == torch.int32 else labels, attention_mask=input_ids.ne(self.tokenizer.pad_token_id))
        if "image" in instances[0]:
            images = [instance["image"] for instance in instances]

            batch["image_sizes"] = [im[1] for im_list in images for im in im_list]
            batch["modalities"] = [im[2] for im_list in images for im in im_list]
            images = [im[0] for im_list in images for im in im_list]

            batch["images"] = images

            target_images = [instance["target_image"][0] for instance in instances]
            target_images = torch.stack(target_images, dim=0) if target_images else None
            batch["target_images"] = target_images


        if "prompt" in instances[0]:
            batch["prompts"] = [instance["prompt"] for instance in instances]
        return batch

def get_dataset_cls(name):

    if name == 'mix':
        dataset_cls = LazySupervisedMixDataset
    else:
        raise ValueError(f'Unknown dataset class {name}')
    return dataset_cls

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = get_dataset_cls(data_args.dataset_cls)
    train_dataset = dataset_cls(tokenizer=tokenizer, data_path=data_args.data_path, data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)