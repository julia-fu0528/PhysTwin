import glob
import os
import json
import sys
import argparse

def optimize_episode(base_path, ep_idx, max_iter=20):
    """Optimize a single episode."""
    case_name = f"episode_{ep_idx:04d}"
    case_path = f"{base_path}/{case_name}"
    
    if not os.path.exists(case_path):
        print(f"Error: {case_path} does not exist")
        return False
    
    print(f"Optimizing {case_name}")
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
        f"TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 "
        f"python optimize_cma.py --base_path {base_path} --case_name {case_name} "
        f"--train_frame {train_frame} --max_iter {max_iter}"
    )
    print(f"Running: {cmd}")
    result = os.system(cmd)
    return result == 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize episodes")
    parser.add_argument("--base_path", type=str, 
                       default="/oscar/data/gdk/hli230/projects/vitac-particle/163-bear",
                       help="Base path to episodes")
    parser.add_argument("--ep_idx", type=int, default=None,
                       help="Specific episode index to optimize (if not provided, optimizes all)")
    parser.add_argument("--max_iter", type=int, default=20,
                       help="Maximum iterations for optimization")
    
    args = parser.parse_args()
    
    if args.ep_idx is not None:
        # Optimize a specific episode
        optimize_episode(args.base_path, args.ep_idx, args.max_iter)
    else:
        # Optimize all episodes
        dir_names = sorted(glob.glob(f"{args.base_path}/episode_*"))
        for dir_name in dir_names:
            case_name = dir_name.split("/")[-1]
            # Extract episode index from case_name (e.g., "episode_3" -> 3)
            try:
                ep_idx = int(case_name.split("_")[-1])
                optimize_episode(args.base_path, ep_idx, args.max_iter)
            except ValueError:
                print(f"Warning: Could not parse episode index from {case_name}, skipping")