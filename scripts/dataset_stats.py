import json
from collections import Counter
from pathlib import Path

INP = Path(r"E:\data\unified\enriched_all.jsonl")

def main():
    c_total = 0
    known = Counter()
    classes = Counter()

    with INP.open("r", encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)
            c_total += 1
            phy = x.get("physical_attributes", {})
            for k in ["size_class", "weight_class", "brachycephalic"]:
                v = phy.get(k, {})
                if isinstance(v, dict):
                    val = v.get("value")
                    if val not in (None, "unknown"):
                        known[k] += 1

            tax = x.get("taxonomy", {})
            cls = tax.get("class", "unknown")
            classes[cls] += 1

    print("TOTAL:", c_total)
    print("KNOWN size_class:", known["size_class"])
    print("KNOWN weight_class:", known["weight_class"])
    print("KNOWN brachycephalic:", known["brachycephalic"])
    print("Top taxonomy classes:", classes.most_common(10))

if __name__ == "__main__":
    main()