# The first stage to optimize the sparse parameters using CMA-ES
from qqtt import OptimizerCMA
from qqtt.utils import logger, cfg
from qqtt.utils.logger import StreamToLogger, logging
import random
import numpy as np
import sys
import torch
import pickle
import json
from argparse import ArgumentParser
import os


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

sys.stdout = StreamToLogger(logger, logging.INFO)
sys.stderr = StreamToLogger(logger, logging.ERROR)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--case_name", type=str, required=True)
    parser.add_argument("--train_frame", type=int, required=True)
    parser.add_argument("--max_iter", type=int, default=20)
    parser.add_argument("--start_iter", type=int, default=0, help="Iteration to start from (for resuming)")
    parser.add_argument("--remove_cams", type=str, default=None)
    args = parser.parse_args()

    base_path = args.base_path
    case_name = args.case_name
    train_frame = args.train_frame
    max_iter = args.max_iter
    start_iter = args.start_iter
    remove_cams = args.remove_cams
    if remove_cams is not None:
        remove_cams = [c for c in remove_cams.split(',') if c]
        print(f"remove cams: {remove_cams}")
    if "cloth" in case_name or "package" in case_name:
        cfg.load_from_yaml("configs/cloth.yaml")
    else:
        cfg.load_from_yaml("configs/real.yaml")

    base_dir = f"{base_path}/experiments_optimization/{case_name}"
    # T_marker2world = np.array([[ 9.92457290e-01, -1.22580045e-01,  1.63125912e-03,  3.31059452e-01],
    #                           [ 2.70205336e-04, -1.11191912e-02, -9.99938143e-01,  1.90897759e-01],
    #                           [ 1.22590601e-01,  9.92396340e-01, -1.10022006e-02,  2.75183546e-01],
    #                           [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
    T_marker2world = np.array([[ 9.92500579e-01, -1.22225711e-01,  1.86443478e-03,  1.36186366e-01],
                              [ 5.43975403e-04, -1.08359291e-02, -9.99941142e-01, -1.88119571e-02],
                              [ 1.22238720e-01,  9.92443176e-01, -1.06881781e-02,  7.19721945e-02],
                              [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
    # Load avg_marker2world from multi-camera ArUco calibration (read_aruco.py)
    # This is more accurate than single-camera calibration as it averages across all cameras
    # aruco_results_path = '/users/wfu16/data/users/wfu16/datasets/2025-10-23_snapshot_julia_aruco/aruco_results/avg_marker2world.npy'
    # if os.path.exists(aruco_results_path):
    #     T_marker2world = np.load(aruco_results_path)
    #     print(f"Loaded T_marker2world from ArUco calibration: {aruco_results_path}")
    # else:
    #     # Fallback to old hardcoded value if file doesn't exist
    #     print(f"Warning: ArUco calibration file not found at {aruco_results_path}, using fallback value")
    #     T_marker2world = np.array([[ 9.92457290e-01, -1.22580045e-01,  1.63125912e-03,  3.31059452e-01],
    #                               [ 2.70205336e-04, -1.11191912e-02, -9.99938143e-01,  1.90897759e-01],
    #                               [ 1.22590601e-01,  9.92396340e-01, -1.10022006e-02,  2.75183546e-01],
    #                               [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
    # invert the ground transform
    T_world2marker = np.linalg.inv(T_marker2world)
    # T_world2marker = T_marker2world
    # Set the intrinsic and extrinsic parameters for visualization
    with open(f"{base_path}/{case_name}/calibrate.pkl", "rb") as f:
        c2ws = pickle.load(f)
        print(f"c2ws shape: {c2ws.shape}")
    c2ws = [T_world2marker @ c2w for c2w in c2ws]
    w2cs = [np.linalg.inv(c2w) for c2w in c2ws]
    cfg.T_world2marker = T_world2marker
    cfg.c2ws = np.array(c2ws)
    cfg.w2cs = np.array(w2cs)
    with open(f"{base_path}/{case_name}/metadata.json", "r") as f:
        data = json.load(f)
    cfg.intrinsics = np.array(data["intrinsics"])
    cfg.WH = data["WH"]
    # cfg.overlay_path = f"{base_path}/{case_name}/color"
    cfg.overlay_path = f"{base_path}/{case_name}"
    cfg.cameras = [subdir for subdir in os.listdir(cfg.overlay_path) if "cam" in subdir]
    cfg.start_frame = data["start_frame"]
    cfg.end_frame = data["end_frame"]
    
    

    logger.set_log_file(path=base_dir, name="optimize_cma_log")
    optimizer = OptimizerCMA(
        data_path=f"{base_path}/{case_name}/final_data.pkl",
        base_dir=base_dir,
        train_frame=train_frame,
        remove_cams=remove_cams,
    )
    optimizer.optimize(max_iter=max_iter, start_iter=start_iter)
