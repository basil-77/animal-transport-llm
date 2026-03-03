import json
import random
from pathlib import Path

SRC = Path(r"E:\data\unified\enriched_all.jsonl")

OUT_TRAIN = Path(r"E:\data\unified\vlm_train.jsonl")
OUT_VAL   = Path(r"E:\ata\unified\vlm_val.jsonl")


INSTRUCTION = (
    "Analyze the animal in the image and return its transport-relevant profile "
    "as JSON with fields: species, size_class, weight_class, brachycephalic."
)


data = []

with SRC.open("r", encoding="utf-8") as f:
    for line in f:

        x = json.loads(line)

        img = x["image_path"]

        tax = x.get("taxonomy", {})
        phy = x.get("physical_attributes", {})

        target = {

            "species":
                tax.get("common_name", "unknown"),

            "size_class":
                phy.get("size_class", {}).get("value", "unknown"),

            "weight_class":
                phy.get("weight_class", {}).get("value", "unknown"),

            "brachycephalic":
                phy.get("brachycephalic", {}).get("value", False)

        }

        sample = {

            "image": img,

            "instruction": INSTRUCTION,

            "output": json.dumps(target)

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