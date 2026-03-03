import os, json
from pathlib import Path

IMG_DIR = Path(r"E:\data\oxford_pets\images")
OUT     = Path(r"E:\data\unified\oxford_pets.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

def norm_breed(filename: str) -> str:
    # Oxford file: "american_bulldog_123.jpg" / "Abyssinian_1.jpg"
    base = Path(filename).stem
    breed = "_".join(base.split("_")[:-1])
    return breed.lower()

def main():
    with OUT.open("w", encoding="utf-8") as f:
        for img in IMG_DIR.glob("*.jpg"):
            breed = norm_breed(img.name)
            sample = {
                "image_path": str(img),
                "taxonomy": {"common_name": "cat_or_dog", "class": "Mammalia"},
                "breed": breed,
                "physical_attributes": {
                    "size_class": {"value": "unknown", "confidence": 0.0},
                    "weight_class": {"value": "unknown", "confidence": 0.0},
                    "brachycephalic": {"value": False, "confidence": 0.0},
                    "needs_carrier": {"value": "unknown", "confidence": 0.0}
                },
                "provenance": {"dataset": "Oxford-IIIT Pet", "license": "CC BY-SA 4.0"}
            }
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()