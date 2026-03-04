import base64
import requests
import json
import re

API_URL = "http://176.118.70.14:44000/v1/chat/completions"
MODEL = "Qwen/Qwen2-VL-7B-Instruct"


def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def extract_json(text: str):
    """
    Extract JSON object from model output.
    Works even if model adds extra text.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in response:\n{text}")

    return json.loads(match.group())


def normalize_output(data: dict):

    needs_carrier = data.get("needs_carrier", "unknown")

    if needs_carrier == "yes":
        needs_carrier = True
    elif needs_carrier == "no":
        needs_carrier = False

    return {
        "size_class": data.get("size_class", "unknown"),
        "weight_class": data.get("weight_class", "unknown"),
        "brachycephalic": data.get("brachycephalic", False),
        "needs_carrier": needs_carrier
    }


def analyze_image(image_path: str):

    image_base64 = encode_image(image_path)

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Given the image of an animal, extract ONLY transport-related attributes.\n"
                            "Return JSON strictly in this format:\n"
                            "{"
                            "\"size_class\": \"small|medium|large|unknown\", "
                            "\"weight_class\": \"<1kg|1-5kg|5-8kg|8-20kg|20-50kg|>50kg|unknown\", "
                            "\"brachycephalic\": true|false, "
                            "\"needs_carrier\": \"yes|no|unknown\""
                            "}\n"
                            "Do not include species or any extra text."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        },
                    },
                ],
            }
        ],
        "temperature": 0.0,
        "extra_body": {
            "lora": "transport"
        }
    }

    r = requests.post(API_URL, json=payload, timeout=120)
    r.raise_for_status()

    content = r.json()["choices"][0]["message"]["content"]

    parsed = extract_json(content)

    return normalize_output(parsed)