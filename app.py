import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import gradio as gr
from main import run_pipeline

NEGATIVE_PROMPT = (
    "extra people, different characters, distorted faces, "
    "low quality, blurry"
)

def run_generation(id_prompt, frame_text, num_frames, height, width, seed):
    frame_prompts = [
        line.strip()
        for line in frame_text.split("\n")
        if line.strip()
    ][:int(num_frames)]

    story_path, output_dir = run_pipeline(
        id_prompt=id_prompt,
        frame_prompts=frame_prompts,
        negative_prompt=NEGATIVE_PROMPT,
        seed=int(seed),
        height=int(height),
        width=int(width),
    )

    return story_path, output_dir

with gr.Blocks() as demo:
    gr.Markdown("## 📖 One Prompt One Story (CPU)")
    gr.Markdown("Story image generation using a single diffusion model")

    id_prompt = gr.Textbox(label="Identity Prompt", lines=4)
    frame_prompts = gr.Textbox(label="Frame Prompts (one per line)", lines=6)

    num_frames = gr.Slider(1, 8, value=4, step=1, label="Number of Frames")

    with gr.Row():
        height = gr.Slider(256, 512, value=512, step=64, label="Height")
        width = gr.Slider(256, 512, value=512, step=64, label="Width")

    seed = gr.Number(value=42, label="Seed")

    generate_btn = gr.Button(" Generate Story")

    story_image = gr.Image(label="Story Strip", type="filepath")
    output_dir = gr.Textbox(label=" Output Folder")

    generate_btn.click(
        run_generation,
        inputs=[id_prompt, frame_prompts, num_frames, height, width, seed],
        outputs=[story_image, output_dir]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
    )
