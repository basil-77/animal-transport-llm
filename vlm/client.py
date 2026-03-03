import base64
from openai import OpenAI


VLLM_ENDPOINT = "http://176.118.70.14:44000/v1"
MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"

client = OpenAI(
    base_url=VLLM_ENDPOINT,
    api_key="EMPTY"
)


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def analyze_image(image_path: str) -> dict:

    image_b64 = encode_image(image_path)

    prompt = """
You are an animal transport classifier.

Analyze the animal in the image and return STRICT JSON:

{
  "size_class": "small | medium | large",
  "weight_class": "<5kg | 5-10kg | 10-25kg | 25kg+",
  "brachycephalic": true | false,
  "needs_carrier": true | false
}

Return ONLY valid JSON.
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ],
            }
        ],
    )

    return response.choices[0].message.content