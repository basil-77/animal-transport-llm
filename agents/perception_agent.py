import torch
import open_clip
from PIL import Image

# load model once
device = "cuda" if torch.cuda.is_available() else "cpu"

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="openai"
)

model = model.to(device)
tokenizer = open_clip.get_tokenizer("ViT-B-32")


# classes relevant to transport
LABELS = [
    "a photo of a cat",
    "a photo of a dog",
    "a photo of a bird",
    "a photo of a rabbit",
    "a photo of a hamster",
    "a photo of a reptile",
]


def classify_animal(image_path: str):

    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    text = tokenizer(LABELS).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = (image_features @ text_features.T)[0]

    best_idx = similarity.argmax().item()

    label = LABELS[best_idx].replace("a photo of a ", "")

    confidence = similarity[best_idx].item()

    return {
        "animal": label,
        "confidence": round(confidence, 3)
    }


if __name__ == "__main__":

    result = classify_animal("test.jpg")

    print(result)