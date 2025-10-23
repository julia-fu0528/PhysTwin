import glob
import os
import json
import sys

base_path = "/users/wfu16/data/users/wfu16/datasets/2025-10-14_julia_umi"
dir_names = glob.glob(f"{base_path}/episode_*") 
for i, dir_name in enumerate(dir_names):
    if i != 0:
        continue
    case_name = dir_name.split("/")[-1]
    
    print(f"Optimizing {case_name}")
    # Read the train test split
    with open(f"{base_path}/{case_name}/split.json", "r") as f:
        split = json.load(f)

    train_frame = split["train"][1]
    print(f"Train frame: {train_frame}")
    os.system(
        f"TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python optimize_cma.py --base_path {base_path} --case_name {case_name} --train_frame {train_frame} --max_iter 20"
    )