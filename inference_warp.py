from qqtt import InvPhyTrainerWarp
from qqtt.utils import logger, cfg
from datetime import datetime
import random
import numpy as np
import torch
from argparse import ArgumentParser
import glob
import os
import pickle
import json


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed = 42
set_all_seeds(seed)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--case_name", type=str, required=True)
    args = parser.parse_args()

    base_path = args.base_path
    case_name = args.case_name

    # if "cloth" in case_name or "package" in case_name:
    # cfg.load_from_yaml("configs/cloth.yaml")
    # else:
        # cfg.load_from_yaml("configs/real.yaml")

    logger.info(f"[DATA TYPE]: {cfg.data_type}")

    base_dir = f"{base_path}/experiments/{case_name}"

    # Read the first-satage optimized parameters to set the indifferentiable parameters
    optimal_path = f"{base_path}/experiments_optimization/{case_name}/optimal_params.pkl"
    logger.info(f"Load optimal parameters from: {optimal_path}")
    assert os.path.exists(
        optimal_path
    ), f"{case_name}: Optimal parameters not found: {optimal_path}"
    with open(optimal_path, "rb") as f:
        optimal_params = pickle.load(f)
    cfg.set_optimal_params(optimal_params)
    T_marker2world = np.array([[ 9.92457290e-01, -1.22580045e-01,  1.63125912e-03,  3.31059452e-01],
                              [ 2.70205336e-04, -1.11191912e-02, -9.99938143e-01,  1.90897759e-01],
                              [ 1.22590601e-01,  9.92396340e-01, -1.10022006e-02,  2.75183546e-01],
                              [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
    # invert the ground transform
    T_world2marker = np.linalg.inv(T_marker2world)
    cfg.T_world2marker = T_world2marker
    # Set the intrinsic and extrinsic parameters for visualization
    with open(f"{base_path}/{case_name}/calibrate.pkl", "rb") as f:
        c2ws = pickle.load(f)
    c2ws = [T_world2marker @ c2w for c2w in c2ws]
    w2cs = [np.linalg.inv(c2w) for c2w in c2ws]
    cfg.c2ws = np.array(c2ws)
    cfg.w2cs = np.array(w2cs)
    with open(f"{base_path}/{case_name}/metadata.json", "r") as f:
        data = json.load(f)
    cfg.intrinsics = np.array(data["intrinsics"])
    cfg.WH = data["WH"]
    cfg.overlay_path = f"{base_path}/{case_name}"
    cfg.cameras = [subdir for subdir in os.listdir(cfg.overlay_path) if "cam" in subdir]
    cfg.start_frame = data["start_frame"]
    cfg.end_frame = data["end_frame"]

    logger.set_log_file(path=base_dir, name="inference_log")
    trainer = InvPhyTrainerWarp(
        data_path=f"{base_path}/{case_name}/final_data.pkl",
        base_dir=base_dir,
        pure_inference_mode=True,
    )
    assert len(glob.glob(f"{base_dir}/train/best_*.pth")) > 0
    best_model_path = glob.glob(f"{base_dir}/train/best_*.pth")[0]
    trainer.test(best_model_path)
