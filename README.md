# 🌌 BLIP3-o

BLIP3-o is a unified multimodal model that combines the reasoning and instruction following strength of autoregressive models with the generative power of diffusion models. Unlike prior works that diffuse VAE features or raw pixels, BLIP3-o diffuses semantically rich **CLIP image features**, enabling a powerful and efficient architecture for both image understanding and generation.

## 📖 [Arxiv](http://arxiv.org/abs/2505.09568)


## Model Checkpoint

### BLIP3o-4B [4B](https://huggingface.co/BLIP3o/BLIP3o-Model-4B)

### BLIP3o-8B [8B](https://huggingface.co/BLIP3o/BLIP3o-Model)

## Evaluate EVA-CLIP Reconstruction

You can  download our checkpoint:

```Shell
python -c "from huggingface_hub import snapshot_download; print(snapshot_download(repo_id='BLIP3o/BLIP3o-Model', repo_type='model'))"
```

and copy the file pipeline_reconstruct.py to the /HF_model/checkpoint/path/ of the BLIP3o-Model. In the /HF_model/checkpoint/path/ folder you will see the diffusion-decoder folder. Copy pipeline_reconstruct.py to the diffusion-decoder folder.

Download the EVA-CLIP vision tower weights from here [EVA-CLIP](https://huggingface.co/jiuhai/eva_clip_vision_tower) Put the path to the EVA-CLIP encoder [here](https://github.com/JiuhaiChen/BLIP3o/blob/7bfef50bb660f41a8536352f2fdd6fa30b06c310/inference.py#L46) 

```Shell
python inference.py  /HF_model/checkpoint/path/
```


### Citation
To cite the paper and model
```
@article{chen2025blip3,
  title={BLIP3-o: A Family of Fully Open Unified Multimodal Models-Architecture, Training and Dataset},
  author={Chen, Jiuhai and Xu, Zhiyang and Pan, Xichen and Hu, Yushi and Qin, Can and Goldstein, Tom and Huang, Lifu and Zhou, Tianyi and Xie, Saining and Savarese, Silvio and others},
  journal={arXiv preprint arXiv:2505.09568},
  year={2025}
}
```
### Acknowledgement
We thanks [EMU2](https://github.com/baaivision/Emu) for the wonderful work of EVA-CLIP and SDXL decoder.
