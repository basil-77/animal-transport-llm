import json
import random
from pathlib import Path

INPUT  = Path(r"E:\data\unified\enriched_all.jsonl")
OUTDIR = Path(r"E:\data\unified")

TRAIN = OUTDIR / "vlm_train.jsonl"
VAL   = OUTDIR / "vlm_val.jsonl"

VAL_RATIO = 0.05

print("Loading dataset...")

samples = []

with open(INPUT, "r", encoding="utf-8") as f:
    for line in f:
        samples.append(json.loads(line))

print("Total:", len(samples))

random.shuffle(samples)

split = int(len(samples) * (1 - VAL_RATIO))

train = samples[:split]
val   = samples[split:]

print("Train:", len(train))
print("Val:", len(val))


def save(path, data):

    with open(path, "w", encoding="utf-8") as f:
        for s in data:
            f.write(json.dumps(s) + "\n")


save(TRAIN, train)
save(VAL, val)

print("Saved:")
print(TRAIN)
print(VAL)