from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer


with open("ocr_train.txt", "r", encoding="utf-8") as f:
    prompts = [line.strip() for line in f if line.strip()]

train_dataset = Dataset.from_dict({"prompt": prompts})
train_dataset.shuffle(seed=0)


training_args = GRPOConfig(output_dir="BLIP3o-NEXT-Text-GRPO", use_liger_loss=True, per_device_train_batch_size=16, num_generations=16, save_steps=50, lr_scheduler_type="cosine", learning_rate=1e-6, beta=0.001)

print(training_args)

## dummy reward for testing
def reward_len(completions, **kwargs):
    return [-abs(20 - len(completion)) for completion in completions]

trainer = GRPOTrainer(
    model="/fsx/home/jiuhai.chen/BLIP3o-NEXT/models/debug",
    reward_funcs=reward_len,
    args=training_args,
    train_dataset=train_dataset,
)
trainer.train()
