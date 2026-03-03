import json
import os
import requests
from tqdm import tqdm

BASE = "D:/mts/data/inat"

IMAGE_DIR = f"{BASE}/images"
OUTPUT_JSONL = f"{BASE}/inat_dataset.jsonl"

os.makedirs(IMAGE_DIR, exist_ok=True)

BASE_URL = "https://ml-inat-competition-datasets.s3.amazonaws.com/"

MAX_IMAGES = 20000

TARGET_CLASSES = {"Mammalia", "Aves", "Reptilia"}


print("Loading categories...")

with open(f"{BASE}/categories.json") as f:
    categories = json.load(f)

cat_map = {}

for c in categories:

    if c["class"] in TARGET_CLASSES:

        cat_map[c["id"]] = c


print("Loading train...")

with open(f"{BASE}/train2019.json") as f:
    train = json.load(f)


ann_map = {a["image_id"]: a["category_id"] for a in train["annotations"]}


count = 0

with open(OUTPUT_JSONL, "w") as out:

    for img in tqdm(train["images"]):

        if count >= MAX_IMAGES:
            break

        cat_id = ann_map.get(img["id"])

        if cat_id not in cat_map:
            continue


        file_name = img["file_name"]

        url = BASE_URL + file_name


        local_name = file_name.replace("/", "_")

        local_path = os.path.join(IMAGE_DIR, local_name)


        if not os.path.exists(local_path):

            try:

                r = requests.get(url, stream=True, timeout=20)

                if r.status_code != 200:
                    continue


                with open(local_path, "wb") as f:
                    for chunk in r.iter_content(1024):
                        f.write(chunk)

            except Exception as e:
                continue


        sample = {

            "image_path": local_path,

            "taxonomy": {

                "common_name": cat_map[cat_id]["name"],

                "species_scientific": cat_map[cat_id]["name"],

                "genus": cat_map[cat_id]["genus"],

                "family": cat_map[cat_id]["family"],

                "order": cat_map[cat_id]["order"],

                "class": cat_map[cat_id]["class"]

            }

        }


        out.write(json.dumps(sample) + "\n")

        count += 1


print("Downloaded:", count)