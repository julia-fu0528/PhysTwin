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
import sys

# test torch inverse
torch.inverse(torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32))

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
    parser.add_argument(
        "--base_path",
        type=str,
        # default="./data/different_types",
        default="/oscar/data/gdk/hli230/projects/vitac-particle/processed/008-pink-cloth",
    )
    parser.add_argument(
        "--gaussian_path",
        type=str,
        default="./gaussian_output",
    )
    parser.add_argument(
        "--bg_img_path",
        type=str,
        default="./data/bg.jpg",
    )
    parser.add_argument("--case_name", type=str, default="episode_0")
    parser.add_argument("--n_ctrl_parts", type=int, default=1)
    parser.add_argument(
        "--inv_ctrl", action="store_true", help="invert horizontal control direction"
    )
    parser.add_argument(
        "--virtual_key_input", action="store_true", help="use virtual key input"
    )
    args = parser.parse_args()

    base_path = args.base_path
    case_name = args.case_name

    if "cloth" in case_name or "package" in case_name:
        cfg.load_from_yaml("configs/cloth.yaml")
    else:
        cfg.load_from_yaml("configs/real.yaml")

    base_dir = f"./temp_experiments/{case_name}"

    # Read the first-satage optimized parameters to set the indifferentiable parameters
    optimal_path = f"{base_path}/experiments_optimization/{case_name}/optimal_params.pkl"
    logger.info(f"Load optimal parameters from: {optimal_path}")
    assert os.path.exists(
        optimal_path
    ), f"{case_name}: Optimal parameters not found: {optimal_path}"
    with open(optimal_path, "rb") as f:
        optimal_params = pickle.load(f)
    cfg.set_optimal_params(optimal_params)  
    T_marker2world = np.array([[ 9.92500579e-01, -1.22225711e-01,  1.86443478e-03,  1.36186366e-01],
                              [ 5.43975403e-04, -1.08359291e-02, -9.99941142e-01, -1.88119571e-02],
                              [ 1.22238720e-01,  9.92443176e-01, -1.06881781e-02,  7.19721945e-02],
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
    cfg.bg_img_path = args.bg_img_path
    cfg.cameras = sorted([subdir for subdir in os.listdir(cfg.overlay_path) if "cam" in subdir])
    cfg.start_frame = data["start_frame"]
    cfg.end_frame = data["end_frame"]

    exp_name = "init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0"
    # gaussians_path = f"{args.gaussian_path}/{case_name}/{exp_name}/point_cloud/iteration_10000/point_cloud.ply"
    gaussians_path = sorted(glob.glob(f"{base_path}/{case_name}/splatfacto/*.ply"))[0]
    # gaussians_path = "base_cloth.ply"
    # gaussians_path = f"{base_path}/{case_name}/start_obj_pcd.ply"
    logger.set_log_file(path=base_dir, name="inference_log")
    trainer = InvPhyTrainerWarp(
        data_path=f"{base_path}/{case_name}/final_data.pkl",
        base_dir=base_dir,
        pure_inference_mode=True,
    )
    # print(f"ground transform: {ground_transform}")
    # trainer.set_ground_transform(ground_transform)

    best_model_path = glob.glob(f"{base_path}/experiments/{case_name}/train/best_*.pth")[0]
    trainer.interactive_playground(
        best_model_path,
        gaussians_path,
        args.n_ctrl_parts,
        args.inv_ctrl,
        virtual_key_input=args.virtual_key_input,
    )
