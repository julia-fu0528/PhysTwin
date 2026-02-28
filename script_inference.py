import glob
import os
import json
import argparse
from camera_utils import find_camera_index

# Set environment variables for headless rendering
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["EGL_PLATFORM"] = "surfaceless"

REMOVE_CAMS = "brics-odroid-003_cam0,brics-odroid-003_cam1,\
brics-odroid-004_cam0,\
brics-odroid-014_cam0,\
brics-odroid-018_cam0,brics-odroid-018_cam1,\
brics-odroid-019_cam0,\
"


def run_inference(base_path, ep_idx, cam_name=None):
    """
    Run inference on a specific episode with specified camera.
    
    Args:
        base_path: Path to the object directory
        ep_idx: Episode index
        cam_name: Camera name (e.g., 'brics-odroid-022_cam1'). Falls back to first camera if None.
    """
    case_name = f"episode_{ep_idx}"
    episode_path = os.path.join(base_path, case_name)
    
    # Find camera index from name
    cam_idx, actual_cam_name = find_camera_index(episode_path, cam_name)
    print(f"Using camera: {actual_cam_name} (index {cam_idx})")
    
    cmd = f"python inference_warp.py --base_path {base_path} --case_name {case_name} --remove_cams {REMOVE_CAMS} --vis_cam_idx {cam_idx}"
    print(f"Running: {cmd}")
    os.system(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on episodes")
    parser.add_argument("--base_path", type=str, required=True, help="Base path to episodes")
    parser.add_argument("--ep_idx", type=int, default=None, help="Specific episode index to run inference on")
    parser.add_argument("--cam_name", type=str, default=None, help="Camera name to use for rendering (e.g., 'brics-odroid-022_cam1'). Falls back to first camera if not specified.")
    
    args = parser.parse_args()
    
    if args.ep_idx is not None:
        run_inference(args.base_path, args.ep_idx, args.cam_name)
    else:
        # Fallback to the original behavior if no ep_idx is provided (but base_path is required now)
        dir_names = sorted(glob.glob(f"{args.base_path}/episode_*"))
        for dir_name in dir_names:
            case_name = dir_name.split("/")[-1]
            try:
                ep_idx = int(case_name.split("_")[-1])
                run_inference(args.base_path, ep_idx, args.cam_name)
            except ValueError:
                print(f"Warning: Could not parse episode index from {case_name}, skipping")

