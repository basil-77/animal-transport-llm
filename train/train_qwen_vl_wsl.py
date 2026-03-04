import json
import torch
from PIL import Image
from torch.utils.data import Dataset

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig
)

from peft import LoraConfig, get_peft_model

torch.backends.cudnn.benchmark = True


MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
CACHE_DIR = "/mnt/d/hf_cache"

TRAIN_JSON = "/mnt/d/mts/data/unified/vlm_train.jsonl"
VAL_JSON   = "/mnt/d/mts/data/unified/vlm_val.jsonl"

OUTPUT_DIR = "/mnt/d/mts/qwen_vl_transport_lora_v2"



# ============================================================
# Dataset (OPTIMIZED)
# ============================================================

class TransportDataset(Dataset):

    def __init__(self, jsonl_path, processor, max_cache=5000):

        self.processor = processor
        self.samples = []
        self.cache = {}

        print("Loading dataset:", jsonl_path)

        with open(jsonl_path, "r") as f:
            for line in f:
                try:
                    self.samples.append(json.loads(line))
                except:
                    pass

        print("Samples loaded:", len(self.samples))


    def load_image(self, path):

        if path in self.cache:
            return self.cache[path]

        try:
            img = Image.open(path).convert("RGB")

            if len(self.cache) < 5000:
                self.cache[path] = img

            return img

        except:
            return Image.new("RGB", (224,224))


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):

        sample = self.samples[idx]

        image = self.load_image(sample["image"])

        output = sample["output"]

        size = output["size_class"]
        weight = output["weight_class"]
        carrier = output["needs_carrier"]
        brachycephalic = output["brachycephalic"]

        target_dict = {
            "size_class": size,
            "weight_class": weight,
            "brachycephalic": bool(brachycephalic),
            "needs_carrier": carrier
        }

        target = json.dumps(target_dict, separators=(",", ":"))
        

        prompt = (
                    "Given the image of an animal, extract ONLY transport-related attributes.\n"
                    "Return JSON strictly in this format:\n"
                    "{"
                    "\"size_class\": \"small|medium|large|unknown\", "
                    "\"weight_class\": \"<1kg|1-5kg|5-8kg|8-20kg|20-50kg|>50kg|unknown\", "
                    "\"brachycephalic\": true|false, "
                    "\"needs_carrier\": \"yes|no|unknown\""
                    "}\n"
                    "Do not include species or any extra text."
                )

        messages = [
            {
                "role":"user",
                "content":[
                    {"type":"image"},
                    {"type":"text","text":prompt}
                ]
            },
            {
                "role":"assistant",
                "content":[
                    {"type":"text","text":target}
                ]
            }
        ]

        text_full = self.processor.apply_chat_template(
            messages,
            tokenize=False
        )

        inputs = self.processor(
            text=[text_full],
            images=[image],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        inputs = {k:v.squeeze(0) for k,v in inputs.items()}

        # prefix masking
        prefix = self.processor.apply_chat_template(
            [messages[0], {"role":"assistant","content":[{"type":"text","text":""}]}],
            tokenize=False
        )

        prefix_ids = self.processor.tokenizer(prefix)["input_ids"]

        labels = inputs["input_ids"].clone()
        labels[:len(prefix_ids)] = -100

        inputs["labels"] = labels

        return inputs
    

class VLMDataCollator:

    def __init__(self, processor):
        self.pad_id = processor.tokenizer.pad_token_id

    def __call__(self, features):

        batch = {}

        for key in features[0]:

            if key == "pixel_values":
                batch[key] = torch.stack([f[key] for f in features])

            elif key == "labels":
                batch[key] = torch.nn.utils.rnn.pad_sequence(
                    [f[key] for f in features],
                    batch_first=True,
                    padding_value=-100
                )

            else:
                batch[key] = torch.nn.utils.rnn.pad_sequence(
                    [f[key] for f in features],
                    batch_first=True,
                    padding_value=self.pad_id
                )

        return batch


# ============================================================
# MAIN
# ============================================================

def main():

    print("Loading processor...")

    processor = Qwen2VLProcessor.from_pretrained(
        MODEL_ID,
        cache_dir=CACHE_DIR,
    )

    processor.image_processor.size = {
        "shortest_edge":448,
        "longest_edge":448
    }


    print("Loading model 4bit...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        cache_dir=CACHE_DIR,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="sdpa"
    )

    for name, param in model.named_parameters():
        if "vision" in name:
            param.requires_grad = False

    model.gradient_checkpointing_enable()
    model.config.use_cache=False


    # LoRA
    lora = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "v_proj",
        ],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model,lora)

    model.print_trainable_parameters()


    train_dataset = TransportDataset(TRAIN_JSON,processor)
    val_dataset   = TransportDataset(VAL_JSON,processor)


    training_args = TrainingArguments(

        output_dir=OUTPUT_DIR,

        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,

        gradient_accumulation_steps=4,

        num_train_epochs=2,

        learning_rate=5e-5,

        logging_steps=50,

        save_steps=1000,

        fp16=True,

        dataloader_num_workers=8,

        dataloader_pin_memory=True,

        report_to="none",

        warmup_ratio=0.05,

        weight_decay=0.01,

        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",        

        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,        
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator = VLMDataCollator(processor)
    )


    trainer.train()

    trainer.save_model(OUTPUT_DIR)

    print("DONE")


if __name__ == "__main__":
    main()