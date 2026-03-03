from datasets import load_dataset

ds = load_dataset("ba188/iNaturalist", split="train")

print(ds[0]["photos"])
print(type(ds[0]["photos"]))