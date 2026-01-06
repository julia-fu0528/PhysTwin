import glob
import os
import json

base_path = "/oscar/data/gdk/hli230/projects/vitac-particle/processed/008-pink-cloth"
REMOVE_CAMS = "brics-odroid-003_cam0,brics-odroid-003_cam1,\
brics-odroid-004_cam0,\
brics-odroid-014_cam0,\
brics-odroid-018_cam0,brics-odroid-018_cam1,\
brics-odroid-019_cam0,\
"
dir_names = glob.glob(f"{base_path}/episode_0") 
for i, dir_name in enumerate(dir_names):
    if i != 0:
        continue
    case_name = dir_name.split("/")[-1]

    os.system(
        f"python inference_warp.py --base_path {base_path} --case_name {case_name} --remove_cams {REMOVE_CAMS}"
    )
