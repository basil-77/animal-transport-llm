import os
import json
import requests
from datasets import load_dataset
from tqdm import tqdm

OUT_IMG = "D:/mts/data/inat/images"
OUT_JSONL = "D:/mts/data/inat/inat_dataset.jsonl"

os.makedirs(OUT_IMG, exist_ok=True)

print("Loading dataset...")

ds = load_dataset("ba188/iNaturalist", split="train")

print("Downloading images...")

with open(OUT_JSONL, "w", encoding="utf-8") as f:

    for i, row in enumerate(tqdm(ds)):
        print (row)
        url = row["photos"]

        if not url:
            continue

        path = f"{OUT_IMG}/inat_{i}.jpg"

        try:

            r = requests.get(url, timeout=20)

            if r.status_code != 200:
                continue

            with open(path, "wb") as img:
                img.write(r.content)

        except Exception as e:
            continue


        taxon = row.get("taxon") or {}

        sample = {

            "image_path": path,

            "taxonomy": {

                "common_name":
                    row.get("common_name") or "unknown",

                "species_scientific":
                    taxon.get("name") if isinstance(taxon, dict) else "unknown",

                "genus":
                    taxon.get("genus") if isinstance(taxon, dict) else "unknown",

                "family":
                    taxon.get("family") if isinstance(taxon, dict) else "unknown",

                "order":
                    taxon.get("order") if isinstance(taxon, dict) else "unknown",

                "class":
                    taxon.get("class") if isinstance(taxon, dict) else "unknown"

            },

            "breed": None,

            "physical_attributes": {

                "size_class": {
                    "value": "unknown",
                    "confidence": 0.0
                },

                "weight_class": {
                    "value": "unknown",
                    "confidence": 0.0
                },

                "brachycephalic": {
                    "value": False,
                    "confidence": 0.0
                },

                "needs_carrier": {
                    "value": "unknown",
                    "confidence": 0.0
                }

            },

            "provenance": {
                "dataset": "iNaturalist"
            }

        }

        f.write(json.dumps(sample, ensure_ascii=False) + "\n")


print("Done")