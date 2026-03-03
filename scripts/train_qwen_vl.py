import torch
import json
from datasets import load_dataset

from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)

from peft import (
    LoraConfig,
    get_peft_model
)


MODEL = "Qwen/Qwen2-VL-7B-Instruct"

TRAIN_FILE = r"E:\data\unified\vlm_train.jsonl"
VAL_FILE   = r"E:\data\unified\vlm_val.jsonl"


print("Loading dataset...")

dataset = load_dataset(
    "json",
    data_files={
        "train": TRAIN_FILE,
        "val": VAL_FILE
    }
)


print("Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL)


print("Loading model (4bit)...")

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)


print("Adding LoRA...")

lora = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj","v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora)

model.print_trainable_parameters()


def preprocess(example):

    messages = [

        {
            "role": "user",
            "content": [
                {"type": "image", "image": example["image"]},
                {"type": "text", "text": example["instruction"]}
            ]
        },

        {
            "role": "assistant",
            "content": example["output"]
        }

    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False
    )

    inputs = processor(
        text=[text],
        images=[example["image"]],
        return_tensors="pt"
    )

    inputs["labels"] = inputs["input_ids"].clone()

    return inputs


print("Tokenizing dataset...")

dataset = dataset.map(
    preprocess,
    remove_columns=dataset["train"].column_names
)


args = TrainingArguments(

    output_dir="qwen_vl_transport",

    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,

    gradient_accumulation_steps=8,

    num_train_epochs=1,

    learning_rate=2e-4,

    logging_steps=50,

    save_steps=1000,

    eval_steps=1000,

    fp16=True,

    gradient_checkpointing=True,

    report_to="none"
)


trainer = Trainer(

    model=model,

    args=args,

    train_dataset=dataset["train"],

    eval_dataset=dataset["val"]

)


print("Starting training...")

trainer.train()


print("Saving...")

model.save_pretrained("qwen_vl_transport_lora")