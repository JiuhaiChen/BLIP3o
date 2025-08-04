# 🌌 BLIP3o-NEXT

Introducing BLIP3o-NEXT, next version of unified multimodal building on top of BLIP3o. 


Install package for pretraining and instruction tuning
```Shell
conda create -n blip3o-next python=3.11 -y
conda activate blip3o-next
pip install --upgrade pip  setuptools
pip install -r requirements.txt
pip install -e .
```

please download the Vision encoder: [VQ-SigLIP2](https://huggingface.co/csuhan/TA-Tok/blob/main/ta_tok.pth)
```Shell
huggingface-cli download csuhan/TA-Tok ta_tok.pth 
```
And import vision encoder path to the script: [scripts/run.sh](https://github.com/JiuhaiChen/BLIP3o/blob/BLIP3o-NEXT/scripts/run.sh#L19)


Please cd to trl for GRPO, we recommend to install a new enviroment since some version conflict for torch. Also you need to install the dependency from  [setup.py](https://github.com/JiuhaiChen/BLIP3o/blob/BLIP3o-NEXT/setup.py), please follow below


```Shell
cd trl
conda create -n grpo python=3.11 -y
conda activate grpo
pip install -r requirements.txt
cd ..
pip install -e .
```

