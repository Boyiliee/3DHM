import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

def run_stage1_preprocessing(ref_image_pil: Image.Image, driving_video_path: str):
    # 1. create a temporary directory under the project root
    temp_root = PROJECT_ROOT / "temp_gradio_runs"
    temp_root.mkdir(exist_ok=True)
    temp_dir = tempfile.mkdtemp(prefix="3dhm_", dir=str(temp_root))
    print(f"create a temporary directory: {temp_dir}")

    try:
        # 2. create the required directory structure
        ref_imgs_dir = Path(temp_dir) / "reference_imgs" / "images"
        driving_videos_dir = Path(temp_dir) / "driving_videos" / Path(driving_video_path).stem
        
        ref_imgs_dir.mkdir(parents=True, exist_ok=True)
        driving_videos_dir.mkdir(parents=True, exist_ok=True)

        # save the uploaded reference image
        ref_image_name = "reference_image.png"
        ref_image_path = ref_imgs_dir / ref_image_name
        ref_image_pil.save(ref_image_path)

        # copy the driving video
        shutil.copy(driving_video_path, driving_videos_dir / "video.mp4")

        # 3. prepare and execute track.py command
        # assume track.py is in the project root
        track_script_path = PROJECT_ROOT / "track.py"
        if not track_script_path.exists():
            raise FileNotFoundError(f"track.py not found at {track_script_path}")

        # track.py needs to run in the 4D-Humans environment.
        # Use 'conda run' to execute the command in the specified environment.
        command = [
            "conda",
            "run",
            "-n",
            "4D-Humans",
            "python",
            str(track_script_path),
            f"driving_videos={driving_videos_dir.parent}",
            f"reference_imgs={ref_imgs_dir.parent}"
        ]

        print(f"Executing command: {' '.join(command)}")
        # We run the command in temp_dir so that the output files are generated there
        result = subprocess.run(command, capture_output=True, text=True, check=False, cwd=temp_dir)

        if result.returncode != 0:
            print("--- track.py Error ---")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            raise RuntimeError(f"track.py failed. Check console for details.")

        print("track.py STDOUT:", result.stdout)
        
        processed_meta_path = Path(temp_dir) / "processed_datas.json"
        if not processed_meta_path.exists():
            raise FileNotFoundError(f"processed_datas.json not found in {temp_dir} after running track.py")
        
        return str(processed_meta_path), temp_dir

    except Exception as e:
        shutil.rmtree(temp_dir)
        print(f"Clean temp folders: {temp_dir}")
        raise e

def run_stage1_inpainting(ref_imgs_folder: str):
    # Ensure the script path is correct relative to this app.py
    inpaint_script_path = os.path.join(PROJECT_ROOT, "stage1_inpaint.py")

    # Command to run the inpainting script
    # Note: We need to find the correct python executable.
    # For now, we assume 'python' is in the path and points to the correct environment.
    command = [
        "conda", "run", "-n", "3dhm",
        sys.executable, # Use the same python that runs the app
        inpaint_script_path,
        ref_imgs_folder
    ]

    print(f"Running inpainting script... {command}")
    # Execute the command
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return result

def run_stage2_generation(processed_meta_path: str, temp_dir: str):
    try:
        # 1. prepare output directory and task name
        task_name = f"task_{Path(temp_dir).name}"
        output_dir = Path(temp_dir) / "stage2_output" / task_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # 2. prepare and execute smt/eval.py command
        eval_script_path = PROJECT_ROOT / "smt" / "eval.py"
        weights_path = PROJECT_ROOT / "models" / "stage2.pt"

        if not eval_script_path.exists():
            raise FileNotFoundError(f"eval.py not found at {eval_script_path}")
        if not weights_path.exists():
            raise FileNotFoundError(f"Stage 2 weights not found at {weights_path}")

        command = [
            "conda",
            "run",
            "-n",
            "3dhm",
            "python",
            str(eval_script_path),
            "-m",
            "--config-name", "eval.yaml",
            f"task_name={task_name}",
            "trainer=deepspeed",
            f"configs.sdiff.task_name={task_name}",
            f"configs.weights_path={weights_path}",
            f"logger.tensorboard.save_dir={output_dir}",
            f"configs.sdiff.video_save_path={output_dir}",
            "configs.test_batch_size=1",
            "configs.test_num_workers=1",
            "configs.base_model=stablemotion_video_v2",
            "configs.timesteps_max=20",
            "configs.sdiff.guidance_scale_control=2.5",
            f"configs.metadata_path={processed_meta_path}"
        ]

        print(f"Executing command: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True, check=False, cwd=PROJECT_ROOT)

        if result.returncode != 0:
            print("--- smt/eval.py Error ---")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            raise RuntimeError(f"smt/eval.py failed. Check console for details.")
        
        print("smt/eval.py STDOUT:", result.stdout)

        output_video_path = output_dir / f"{task_name}.mp4"
        if not output_video_path.exists():
            # If not found, try to find the only mp4 file in the directory
            mp4_files = list(output_dir.glob("*.mp4"))
            if not mp4_files:
                raise FileNotFoundError(f"Generated video not found in {output_dir}")
            output_video_path = mp4_files[0]

        return str(output_video_path)

    except Exception as e:
        # Do not clean up the directory for debugging
        print(f"Stage 2 failed. Temporary files are in {temp_dir}")
        raise e