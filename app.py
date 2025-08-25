import gradio as gr
from PIL import Image
import os
import time
import itertools
import subprocess

# Ensure we can import from the src directory
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from smt.inference import run_stage1_preprocessing, run_stage1_inpainting, run_stage2_generation

def data_processing_fn(ref_image_pil, driving_video_path):
    """Handles the data processing stage."""
    if ref_image_pil is None:
        raise gr.Error("Please upload a Reference Image.")
    if driving_video_path is None:
        raise gr.Error("Please upload a Driving Video.")
        
    gr.Info("🚀 Starting Stage 1: Data Preprocessing...")
    start_time = time.time()
    
    try:
        ref_image = Image.fromarray(ref_image_pil)
        processed_meta_path, temp_dir = run_stage1_preprocessing(ref_image, driving_video_path)
        processing_time = time.time() - start_time
        gr.Info(f"✅ Stage 1 complete! Time: {processing_time:.2f}s. Ready for video generation.")
        
        # --- Inpainting Stage ---
        gr.Info("🎨 Starting Stage 1: Inpainting...")
        inpaint_start_time = time.time()
        ref_imgs_folder = os.path.join(temp_dir, "reference_imgs")
        result = run_stage1_inpainting(ref_imgs_folder)

        if result.returncode != 0:
            print("--- Inpainting Error ---")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            raise gr.Error(f"Error during inpainting stage. Check console for details.")

        inpaint_time = time.time() - inpaint_start_time
        print(f"Inpainting complete! Time: {inpaint_time:.2f}s.")
        gr.Info(f"✅ Inpainting complete! Time: {inpaint_time:.2f}s.")
        
        # Return the paths to the state and enable the next button
        return processed_meta_path, temp_dir, gr.update(interactive=True)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise gr.Error(f"Error in Stage 1: {e}")

def video_generation_fn(processed_meta_path, temp_dir):
    """Handles the video generation stage."""
    if not processed_meta_path or not temp_dir:
        raise gr.Error("Please complete Stage 1 (Data Processing) successfully first.")

    gr.Info("🚀 Starting Stage 2: Video Generation...")
    start_time = time.time()

    try:
        output_video_path = run_stage2_generation(processed_meta_path, temp_dir)
        
        processing_time = time.time() - start_time
        gr.Info(f"✅ Video generation complete! Total time: {processing_time:.2f}s.")
        
        return output_video_path

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise gr.Error(f"Error in Stage 2: {e}")

# --- Gradio Interface Definition ---
with gr.Blocks(css="footer {display: none !important}") as demo:
    gr.Markdown("# StableMotion Gradio Demo")
    gr.Markdown(
        "Upload a **Reference Image** and a **Driving Video**. The model will generate a new video "
        "where the person from the reference image mimics the motion from the driving video."
    )
    
    # Hidden state to store the path to the processed data between steps
    processed_meta_path_state = gr.State()
    temp_dir_state = gr.State()

    with gr.Row():
        with gr.Column(scale=1):
            ref_image = gr.Image(type="numpy", label="Reference Image")
            driving_video = gr.Video(label="Driving Video")
            
            with gr.Row():
                process_btn = gr.Button("1. Process Data", variant="secondary")
                generate_btn = gr.Button("2. Generate Video", variant="primary", interactive=False)
        
        with gr.Column(scale=1):
            output_video = gr.Video(label="Generated Video")

    # --- Button Click Actions ---
    process_btn.click(
        fn=data_processing_fn,
        inputs=[ref_image, driving_video],
        outputs=[processed_meta_path_state, temp_dir_state, generate_btn]
    )
    
    generate_btn.click(
        fn=video_generation_fn,
        inputs=[processed_meta_path_state, temp_dir_state],
        outputs=output_video
    )
    
    # --- Examples ---b    
    examples_dir = Path(__file__).parent / "examples"
    ref_img_dir = examples_dir / "test_refimgs" / "images"
    video_dir = examples_dir / "test_videos"
    
    # Find example images and videos
    example_images = sorted([str(p) for p in ref_img_dir.glob("*.jpg")])
    example_videos = sorted([str(p) for p in video_dir.glob("**/*.mp4")])

    # Create a combined list of examples
    if example_images and example_videos:
        # Create a few explicit combinations
        example_pairs = [
            [example_images[0], example_videos[0]],
            [example_images[1], example_videos[1]]
        ]
        
        gr.Examples(
            examples=example_pairs,
            inputs=[ref_image, driving_video],
            label="Examples (Click to load)"
        )

if __name__ == "__main__":
    demo.launch(share=True)