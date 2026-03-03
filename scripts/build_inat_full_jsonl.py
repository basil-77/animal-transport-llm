import json, random, os
from pathlib import Path

ROOT = Path(r"E:\data\inat_full")
IMG_ROOT = ROOT / "train_val2019"          # папка с картинками
TRAIN_JSON = ROOT / "train2019.json"       # аннотации
CATEGORIES_JSON = ROOT / "categories.json" # категории
OUT = Path(r"E:\data\unified\inat_full_sample.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

# сколько брать
TOTAL_SAMPLES = 150_000
TARGET_CLASSES = {"Mammalia", "Aves", "Reptilia"}  # можно расширить

def main():
    assert IMG_ROOT.exists(), f"Not found: {IMG_ROOT}"
    assert TRAIN_JSON.exists(), f"Not found: {TRAIN_JSON}"
    assert CATEGORIES_JSON.exists(), f"Not found: {CATEGORIES_JSON}"

    with open(CATEGORIES_JSON, "r", encoding="utf-8") as f:
        cats = json.load(f)
    cat_map = {c["id"]: c for c in cats if c.get("class") in TARGET_CLASSES}

    with open(TRAIN_JSON, "r", encoding="utf-8") as f:
        tr = json.load(f)

    ann_map = {a["image_id"]: a["category_id"] for a in tr["annotations"]}

    # отфильтруем изображения по нужным классам
    candidates = []
    for img in tr["images"]:
        cat_id = ann_map.get(img["id"])
        if cat_id in cat_map:
            candidates.append(img)

    print("Candidates:", len(candidates))

    random.shuffle(candidates)
    picked = candidates[:TOTAL_SAMPLES]

    with open(OUT, "w", encoding="utf-8") as out:
        for img in picked:
            rel = img["file_name"]  # например train_val2019/Mammalia/Canidae/xxx.jpg
            # на диске после распаковки обычно путь совпадает
            local = (ROOT / rel).resolve()
            if not local.exists():
                # fallback: если распаковали в IMG_ROOT без "train_val2019" в file_name
                alt = (IMG_ROOT / Path(rel).relative_to("train_val2019")).resolve()
                if alt.exists():
                    local = alt
                else:
                    continue

            cat = cat_map[ann_map[img["id"]]]
            sample = {
                "image_path": str(local),
                "taxonomy": {
                    "common_name": cat.get("name", "unknown"),
                    "species_scientific": cat.get("name", "unknown"),
                    "genus": cat.get("genus", "unknown"),
                    "family": cat.get("family", "unknown"),
                    "order": cat.get("order", "unknown"),
                    "class": cat.get("class", "unknown")
                },
                "breed": None,
                "physical_attributes": {
                    "size_class": {"value": "unknown", "confidence": 0.0},
                    "weight_class": {"value": "unknown", "confidence": 0.0},
                    "brachycephalic": {"value": False, "confidence": 0.0},
                    "needs_carrier": {"value": "unknown", "confidence": 0.0}
                },
                "provenance": {"dataset": "iNaturalist2019_full"}
            }
            out.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print("Wrote:", OUT)

if __name__ == "__main__":
    main()