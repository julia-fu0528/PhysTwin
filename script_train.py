import glob
import os
import json
import sys
import argparse

REMOVE_CAMS = "brics-odroid-003_cam0,brics-odroid-003_cam1,\
brics-odroid-004_cam0,\
brics-odroid-014_cam0,\
brics-odroid-018_cam0,brics-odroid-018_cam1,\
brics-odroid-019_cam0,\
"
def train_episode(base_path, ep_idx, remove_cams=None, no_gui=False):
    """Train a single episode."""
    case_name = f"episode_{ep_idx}"
    case_path = f"{base_path}/{case_name}"
    
    if not os.path.exists(case_path):
        print(f"Error: {case_path} does not exist")
        return False
    
    print(f"Training {case_name}")
    # Read the train test split
    split_path = f"{case_path}/split.json"
    if not os.path.exists(split_path):
        print(f"Error: {split_path} does not exist")
        return False
    
    with open(split_path, "r") as f:
        split = json.load(f)

    train_frame = split["train"][1]
    print(f"Train frame: {train_frame}")
    
    cmd = (
        f"python train_warp.py --base_path {base_path} --case_name {case_name} "
        f"--train_frame {train_frame} --remove_cams {remove_cams}"
    )
    if no_gui:
        cmd += " --no-gui"
    print(f"Running: {cmd}")
    result = os.system(cmd)
    return result == 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train episodes")
    parser.add_argument("--base_path", type=str, 
                       default="/mnt/data/ParticleData/processed/2025-12-15-163-bear",
                       help="Base path to episodes")
    parser.add_argument("--ep_idx", type=int, default=0,
                       help="Specific episode index to train (if not provided, trains all)")
    parser.add_argument("--no-gui", action="store_true", help="Disable GUI visualizations")
    
    args = parser.parse_args()
    
    if args.ep_idx is not None:
        # Train a specific episode
        train_episode(args.base_path, args.ep_idx, REMOVE_CAMS, no_gui=args.no_gui)
    else:
        # Train all episodes
        dir_names = sorted(glob.glob(f"{args.base_path}/episode_*"))
        for dir_name in dir_names:
            case_name = dir_name.split("/")[-1]
            # Extract episode index from case_name (e.g., "episode_0001" -> 1)
            try:
                ep_idx = int(case_name.split("_")[-1])
                train_episode(args.base_path, ep_idx, REMOVE_CAMS, no_gui=args.no_gui)
            except ValueError:
                print(f"Warning: Could not parse episode index from {case_name}, skipping")
