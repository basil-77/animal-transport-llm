import json
from pathlib import Path

BASE = Path(r"E:\data\CID\images")
OUT  = Path(r"E:\data\unified\cid.jsonl")

OUT.parent.mkdir(parents=True, exist_ok=True)

count = 0

for cow_dir in BASE.iterdir():

    if not cow_dir.is_dir():
        continue

    cow_id = cow_dir.name

    for img in cow_dir.glob("*.jpg"):

        if img.name.startswith("._"):
            continue

        sample = {

            "image_path": str(img),

            "taxonomy":
            {
                "common_name": "cow",
                "class": "Mammalia",
                "family": "Bovidae"
            },

            "breed": None,

            "physical_attributes":
            {
                "size_class":
                    {"value": "large", "confidence": 1.0},

                "weight_class":
                    {"value": ">50kg", "confidence": 0.8},

                "brachycephalic":
                    {"value": False, "confidence": 1.0},

                "needs_carrier":
                    {"value": False, "confidence": 0.9}
            },

            "provenance":
            {
                "dataset": "CID"
            }

        }

        OUT.write_text(
            (OUT.read_text(encoding="utf-8") if OUT.exists() else "")
            + json.dumps(sample) + "\n",
            encoding="utf-8"
        )

        count += 1


print("CID images:", count)
print("Saved:", OUT)