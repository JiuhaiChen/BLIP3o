# BLIP3o-NEXT (In preparation)

Introducing BLIP3o-NEXT, next version of unified multimodal building on top of BLIP3o. 


- **Fully Open-Source:**
  - **Pretraining Data:** [27 Million Detailed Captions](https://huggingface.co/datasets/BLIP3o/BLIP3o-Pretrain-Long-Caption), [5 Million Short Captions](https://huggingface.co/datasets/BLIP3o/BLIP3o-Pretrain-Short-Caption)
  - **Instruction Tuning Data:** [BLIP3o-60k](https://huggingface.co/datasets/BLIP3o/BLIP3o-60k), [ShareGPT-4o-Image](https://huggingface.co/datasets/FreedomIntelligence/ShareGPT-4o-Image)
  - **Model Weights:** [Pretrain](https://huggingface.co/BLIP3o/BLIP3o-NEXT-Pretrain), [Instruction Tuning](https://huggingface.co/BLIP3o/BLIP3o-NEXT-SFT), [GRPO-Geneval](https://huggingface.co/BLIP3o/BLIP3o-NEXT-GRPO-Geneval), [GRPO-Text]()
  - **Training Code:** Pretrain, Instruction Tuning, GRPO



Install package for pretraining and instruction tuning
```Shell
conda create -n blip3o-next python=3.11 -y
conda activate blip3o-next
pip install --upgrade pip  setuptools
pip install -r requirements.txt
pip install -e .
```

please download the vision encoder: [VQ-SigLIP2](https://huggingface.co/csuhan/TA-Tok/blob/main/ta_tok.pth)
```Shell
huggingface-cli download csuhan/TA-Tok ta_tok.pth 
```
And import vision encoder full path to the script: [scripts/run.sh](https://github.com/JiuhaiChen/BLIP3o/blob/BLIP3o-NEXT/scripts/run.sh#L13)
```Shell
sbatch  scrips/run.sh
```


For GRPO, we recommend to install a new enviroment since some version conflicts for torch if using blip3o-next environment. Also you need to install the dependency from  [setup.py](https://github.com/JiuhaiChen/BLIP3o/blob/BLIP3o-NEXT/setup.py), please follow below


```Shell
cd trl
conda create -n grpo python=3.11 -y
conda activate grpo
pip install -r requirements.txt
cd ..
pip install -e .
```

