import os
import json
import requests
from datasets import load_dataset
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

OUT_IMG = "D:/mts/data/inat/images"
OUT_JSONL = "D:/mts/data/inat/inat_dataset.jsonl"

MAX_WORKERS = 12

os.makedirs(OUT_IMG, exist_ok=True)

print("Loading dataset...")
ds = load_dataset("ba188/iNaturalist", split="train")

session = requests.Session()


def download_one(idx_row):

    idx, row = idx_row

    url = row["photos"]

    path = f"{OUT_IMG}/inat_{idx}.jpg"

    if os.path.exists(path):
        return idx, path, row

    try:

        r = session.get(url, timeout=20)

        if r.status_code != 200:
            return None

        with open(path, "wb") as f:
            f.write(r.content)

        return idx, path, row

    except:
        return None


print("Downloading images in parallel...")

results = []

with ThreadPoolExecutor(MAX_WORKERS) as executor:

    futures = [
        executor.submit(download_one, (i, ds[i]))
        for i in range(len(ds))
    ]

    for f in tqdm(as_completed(futures), total=len(futures)):

        res = f.result()

        if res:
            results.append(res)


print("Writing JSONL...")

with open(OUT_JSONL, "w", encoding="utf-8") as out:

    for idx, path, row in results:

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

        out.write(json.dumps(sample, ensure_ascii=False) + "\n")


print("Done.")
print("Images:", len(results))
print("JSON:", OUT_JSONL)