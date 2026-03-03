import os
import json
from pathlib import Path

BASE = Path(r"E:\data\animals10\raw-img")
OUT  = Path(r"E:\data\unified\animals10.jsonl")

OUT.parent.mkdir(parents=True, exist_ok=True)

translate = {
    "cane": "dog",
    "cavallo": "horse",
    "elefante": "elephant",
    "farfalla": "butterfly",
    "gallina": "chicken",
    "gatto": "cat",
    "mucca": "cow",
    "pecora": "sheep",
    "ragno": "spider",
    "scoiattolo": "squirrel"
}


count = 0

with OUT.open("w", encoding="utf-8") as f:

    for folder in BASE.iterdir():

        if not folder.is_dir():
            continue

        italian = folder.name.lower()

        english = translate.get(italian)

        if english is None:
            print("Unknown class:", italian)
            continue

        for img in folder.glob("*.*"):

            sample = {

                "image_path": str(img),

                "taxonomy": {

                    "common_name": english,

                    "class": "Mammalia" if english not in ["butterfly", "spider", "chicken"] else "unknown"

                },

                "breed": None,

                "physical_attributes": {

                    "size_class":
                        {"value": "unknown", "confidence": 0},

                    "weight_class":
                        {"value": "unknown", "confidence": 0},

                    "brachycephalic":
                        {"value": False, "confidence": 0},

                    "needs_carrier":
                        {"value": "unknown", "confidence": 0}

                },

                "provenance":
                    {"dataset": "Animals-10"}

            }

            f.write(json.dumps(sample) + "\n")

            count += 1


print("Animals-10 samples:", count)
print("Saved:", OUT)