import os
from datetime import datetime
from PIL import Image
from unet.utils import get_generator

NUM_STEPS = 20
GUIDANCE_SCALE = 5.8

def generate_story_images(
    pipe,
    prompts,
    negative_prompt,
    seed,
    height,
    width,
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("results", f"story_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    images = []

    for i, prompt in enumerate(prompts, start=1):
        generator = get_generator(seed, i)

        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            generator=generator,
            num_inference_steps=NUM_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            height=height,
            width=width,
        ).images[0]

        path = os.path.join(output_dir, f"frame_{i}.png")
        image.save(path)
        images.append(image)

    total_width = width * len(images)
    story_strip = Image.new("RGB", (total_width, height))

    x_offset = 0
    for img in images:
        story_strip.paste(img, (x_offset, 0))
        x_offset += width

    story_path = os.path.join(output_dir, "story_strip.png")
    story_strip.save(story_path)

    return story_path, output_dir
