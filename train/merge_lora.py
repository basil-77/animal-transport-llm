import torch
from transformers import Qwen2VLForConditionalGeneration
from peft import PeftModel

BASE = "Qwen/Qwen2-VL-7B-Instruct"
LORA = "/mnt/d/mts/qwen_vl_transport_lora_v2"

print("Loading base model in fp16...")

model = Qwen2VLForConditionalGeneration.from_pretrained(
    BASE,
    torch_dtype=torch.float16,
    device_map=None,      # обязательно убрать auto
    attn_implementation="sdpa"
)

model = model.to("cuda")


print("Loading LoRA...")

model = PeftModel.from_pretrained(model, LORA)

print("Merging...")

model = model.merge_and_unload()

print("Saving merged model...")

model.save_pretrained("/mnt/d/mts/qwen_vl_transport_merged_v2")

print("Done.")


from transformers import Qwen2VLProcessor

processor = Qwen2VLProcessor.from_pretrained(BASE)
processor.save_pretrained("/mnt/d/mts/qwen_vl_transport_merged_v2")

print("Saved.")