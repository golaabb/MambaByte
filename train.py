from transformers import TrainingArguments, TrainerCallback
from utils import ByteDataset, ByteDatasetCollator, MambaTrainer, ByteDatasetConfig
from model import Mamba, ModelArgs
import torch

class PrintCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        print(f"Step {state.global_step}: {logs}")

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    
)

# Assuming you have a file 'data.txt' in the current directory for training
dataset_config = ByteDatasetConfig(filepath=r"C:\Users\tryam\SoftWatersAI\MambaByte\rabi.txt")
dataset = ByteDataset(config=dataset_config)

collator = ByteDatasetCollator()

model_args = ModelArgs(
    d_model=256,
    n_layer=4,
    vocab_size=256,  # Assuming ASCII range
)
model = Mamba(args=model_args)

trainer = MambaTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collator,
    callbacks=[PrintCallback()]
)

trainer.train()
trainer.save_model("./trained_model")
