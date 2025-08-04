Install package for pretraining and instruction tuning
```Shell
conda create -n blip3o-next python=3.11 -y
conda activate blip3o-next
pip install --upgrade pip  setuptools
pip install -r requirements.txt
pip install -e .
```

please download the TAR, VQ-SigLIP2
```Shell
huggingface-cli download csuhan/TA-Tok ta_tok.pth 
```
And import path to the script: 

Please cd to trl for GRPO, we recommend to install a new enviroment since some version conflict for torch. 

```Shell
cd trl
conda create -n grpo python=3.11 -y
conda activate grpo
pip install -r requirements.txt
cd ..
pip install -e .
```

