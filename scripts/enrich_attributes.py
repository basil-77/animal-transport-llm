import json
from pathlib import Path

LOOKUP = Path(r"E:\data\unified\attribute_lookup.json")

INPUTS = [
    Path(r"E:\data\unified\animals10.jsonl"),
    Path(r"E:\data\unified\oxford_pets.jsonl"),
    #Path(r"E:\data\unified\cid.jsonl"),
    #Path(r"E:\data\unified\inat_full_sample.jsonl"),
    #Path(r"E:\data\inat\inat_dataset.jsonl")  # ваш HF subset (если хотите тоже включить)
]

carrier_map = {
    "dog": "yes",
    "cat": "yes",
    "cow": "no",
    "horse": "no",
    "sheep": "no",
    "elephant": "no",
    "chicken": "yes",
    "butterfly": "yes",
    "spider": "yes",
    "squirrel": "yes"
}


OUT = Path(r"E:\data\unified\enriched_all.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

def main():
    lookup = json.loads(LOOKUP.read_text(encoding="utf-8"))
    ox_map = lookup.get("oxford_pets_breed", {})
    a10_map = lookup.get("animals10_class", {})

    n = 0
    with OUT.open("w", encoding="utf-8") as out:
        for inp in INPUTS:
            if not inp.exists():
                continue
            with inp.open("r", encoding="utf-8") as f:
                for line in f:
                    x = json.loads(line)
                    prov = (x.get("provenance", {}) or {}).get("dataset", "")

                    # animals10 enrichment
                    if prov == "Animals-10":
                        cls = (x.get("taxonomy", {}) or {}).get("common_name", "").lower()
                        attrs = a10_map.get(cls)
                        if attrs:
                            x["physical_attributes"]["size_class"] = {"value": attrs["size_class"]}
                            x["physical_attributes"]["weight_class"] = {"value": attrs["weight_class"]}
                            x["physical_attributes"]["brachycephalic"] = {"value": bool(attrs["brachycephalic"])}
                            carrier = carrier_map.get(cls, "unknown")
                            x["physical_attributes"]["needs_carrier"] = {
                                "value": carrier}
                    # oxford enrichment
                    ox_hit = 0
                    ox_miss = 0

                    if prov.startswith("Oxford"):
                        breed = (x.get("breed") or "").strip().lower()
                        attrs = ox_map.get(breed)

                        if attrs:
                            ox_hit += 1
                            x["physical_attributes"]["size_class"] = {"value": attrs["size_class"]}
                            x["physical_attributes"]["weight_class"] = {"value": attrs["weight_class"]}
                            x["physical_attributes"]["brachycephalic"] = {"value": bool(attrs["brachycephalic"])}
                        else:
                            ox_miss += 1

                    # --- CLEAN PHYSICAL ATTRIBUTES (remove confidence everywhere) ---
                    pa = x.get("physical_attributes")
                    if isinstance(pa, dict):
                        for k, v in list(pa.items()):
                            if isinstance(v, dict) and "value" in v:
                                pa[k] = {"value": v["value"]}

                    out.write(json.dumps(x, ensure_ascii=False) + "\n")
                    n += 1

    print("Oxford hits:", ox_hit)
    print("Oxford miss:", ox_miss)

    print("Total enriched rows:", n)
    print("Saved:", OUT)

if __name__ == "__main__":
    main()