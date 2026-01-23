import glob
import os
import json
import argparse

# Set environment variables for headless rendering
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["EGL_PLATFORM"] = "surfaceless"

REMOVE_CAMS = "brics-odroid-003_cam0,brics-odroid-003_cam1,\
brics-odroid-004_cam0,\
brics-odroid-014_cam0,\
brics-odroid-018_cam0,brics-odroid-018_cam1,\
brics-odroid-019_cam0,\
"
VIS_CAM_IDX = 0

def run_inference(base_path, ep_idx):
    case_name = f"episode_{ep_idx}"
    cmd = f"python inference_warp.py --base_path {base_path} --case_name {case_name} --remove_cams {REMOVE_CAMS} --vis_cam_idx {VIS_CAM_IDX}"
    print(f"Running: {cmd}")
    os.system(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on episodes")
    parser.add_argument("--base_path", type=str, required=True, help="Base path to episodes")
    parser.add_argument("--ep_idx", type=int, default=None, help="Specific episode index to run inference on")
    
    args = parser.parse_args()
    
    if args.ep_idx is not None:
        run_inference(args.base_path, args.ep_idx)
    else:
        # Fallback to the original behavior if no ep_idx is provided (but base_path is required now)
        dir_names = sorted(glob.glob(f"{args.base_path}/episode_*"))
        for dir_name in dir_names:
            case_name = dir_name.split("/")[-1]
            try:
                ep_idx = int(case_name.split("_")[-1])
                run_inference(args.base_path, ep_idx)
            except ValueError:
                print(f"Warning: Could not parse episode index from {case_name}, skipping")
