import json
import random
from pathlib import Path

SRC = Path(r"E:\data\unified\enriched_all.jsonl")

OUT_TRAIN = Path(r"E:\data\unified\vlm_train.jsonl")
OUT_VAL   = Path(r"E:\data\unified\vlm_val.jsonl")


INSTRUCTION = (
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


data = []

with SRC.open("r", encoding="utf-8") as f:
    for line in f:

        x = json.loads(line)

        img = x["image_path"].replace('E:\\data', 'N:\\data')

        tax = x.get("taxonomy", {})
        phy = x.get("physical_attributes", {})

        target = {

            "size_class":
                phy.get("size_class", {}).get("value", "unknown"),

            "weight_class":
                phy.get("weight_class", {}).get("value", "unknown"),

            "brachycephalic":
                phy.get("brachycephalic", {}).get("value", False),

            "needs_carrier":
                phy.get("needs_carrier", {}).get("value", "unknown")
        }

        sample = {

            "image": img,

            #"instruction": INSTRUCTION,

            "output": target

        }

        data.append(sample)


random.shuffle(data)

split = int(len(data) * 0.95)

train = data[:split]
val   = data[split:]


def save(path, dataset):

    with path.open("w", encoding="utf-8") as f:
        for s in dataset:
            f.write(json.dumps(s) + "\n")


save(OUT_TRAIN, train)
save(OUT_VAL, val)


print("Train:", len(train))
print("Val:", len(val))