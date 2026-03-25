import requests
import io
import os
from PIL import Image

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
API_URL = "https://router.huggingface.co/hf-inference/models/Lykon/DreamShaper-7"
HF_TOKEN = os.getenv("HF_TOKEN")

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

ID_PROMPT = (
    "a realistic cinematic photo of a middle aged female teacher wearing glasses, "
    "same face, same hairstyle, consistent appearance"
)

FRAME_PROMPTS = [
    "standing near the blackboard explaining a lesson",
    "student raising hand and asking a question",
    "teacher smiling and responding",
    "student nodding with understanding"
]

NEGATIVE_PROMPT = (
    "extra people, different face, distorted face, "
    "cartoon, anime, illustration, low quality, blurry"
)

SEED = 42
WIDTH = 512
HEIGHT = 512

os.makedirs("api_results", exist_ok=True)

# -------------------------------------------------
# IMAGE GENERATION FUNCTION
# -------------------------------------------------
def generate_image(prompt, index):
    payload = {
        "inputs": prompt,
        "parameters": {
            "seed": SEED,
            "width": WIDTH,
            "height": HEIGHT,
            "negative_prompt": NEGATIVE_PROMPT
        }
    }

    response = requests.post(API_URL, headers=HEADERS, json=payload)

    if response.status_code == 200:
        image = Image.open(io.BytesIO(response.content))
        image.save(f"api_results/frame_{index}.png")
        print(f"✅ Saved frame_{index}.png")
    else:
        print("❌ Error:", response.text)

# -------------------------------------------------
# RUN STORY GENERATION
# -------------------------------------------------
for i, frame in enumerate(FRAME_PROMPTS, start=1):
    full_prompt = f"{ID_PROMPT}, {frame}"
    generate_image(full_prompt, i)
