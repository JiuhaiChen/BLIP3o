# BLIP3o-NEXT (GRPO)

We use [trl](https://github.com/huggingface/trl) to implement the GRPO

We recommend to install a new enviroment since some package version conflicts if using blip3o-next environment. Also you need to install the dependency from  [setup.py](https://github.com/JiuhaiChen/BLIP3o/blob/BLIP3o-NEXT/setup.py), please follow below


```Shell
conda create -n grpo python=3.11 -y
conda activate grpo
pip install -r requirements.txt
cd ..
pip install -e .
```

For running GRPO
```Shell
bash run.sh
```

We use [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) to return the reward function, if you want to use Geneval, please follow [reward-server](https://github.com/yifan123/reward-server) to create the api call, and modify [OCR reward](https://github.com/JiuhaiChen/BLIP3o/blob/BLIP3o-NEXT/trl/trl/trainer/grpo_trainer.py#L1331)

