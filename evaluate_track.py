import pickle
import glob
import csv
import json
import numpy as np
import os
import argparse
import wandb
from scipy.spatial import KDTree


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Track Error")
    parser.add_argument("--base_path", type=str, default="./data/different_types",
                        help="Path to ground truth data")
    parser.add_argument("--prediction_dir", type=str, default="./experiments",
                        help="Path to experiment outputs")
    parser.add_argument("--output_file", type=str, default="results/track_results.csv",
                        help="Path to output CSV file")
    parser.add_argument("--ep_idx", type=int, default=None,
                        help="Specific episode index to evaluate")
    return parser.parse_args()


args = parse_args()
base_path = args.base_path
prediction_path = args.prediction_dir
output_file = args.output_file

if args.ep_idx is not None and output_file == "results/track_results.csv":
    output_file = f"results/episode_{args.ep_idx}_track.csv"

os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)


def evaluate_prediction(start_frame, end_frame, vertices, gt_track_3d, idx, mask):
    track_errors = []
    for frame_idx in range(start_frame, end_frame):
        # Get the new mask and see
        if frame_idx >= len(gt_track_3d):
            break
        new_mask = ~np.isnan(gt_track_3d[frame_idx][mask]).any(axis=1)
        gt_track_points = gt_track_3d[frame_idx][mask][new_mask]
        pred_x = vertices[frame_idx][idx][new_mask]
        if len(pred_x) == 0:
            track_error = 0
        else:
            track_error = np.mean(np.linalg.norm(pred_x - gt_track_points, axis=1))
        
        track_errors.append(track_error)
    return np.mean(track_errors)


file = open(output_file, mode="w", newline="", encoding="utf-8")
writer = csv.writer(file)
writer.writerow(
    [
        "Case Name",
        "Train Track Error",
        "Test Track Error",
    ]
)

if args.ep_idx is not None:
    case_name = f"episode_{args.ep_idx}"
    dir_names = [os.path.join(prediction_path, case_name)]
else:
    dir_names = glob.glob(f"{prediction_path}/episode_*")

for dir_name in dir_names:
    case_name = os.path.basename(dir_name)
    print(f"Processing {case_name}!!!!!!!!!!!!!!!")

    with open(f"{base_path}/{case_name}/split.json", "r") as f:
        split = json.load(f)
    frame_len = split["frame_len"]
    offset = split["train"][0]
    train_frame = split["train"][1] - offset
    test_frame = split["test"][1] - offset

    with open(f"{prediction_path}/{case_name}/inference.pkl", "rb") as f:
        vertices = pickle.load(f)

    with open(f"{base_path}/{case_name}/final_data.pkl", "rb") as f:
        data = pickle.load(f)
    gt_track_3d = data["object_points"]
    
    # Transform points from world space to marker space
    T_marker2world = np.array([[ 9.92500579e-01, -1.22225711e-01,  1.86443478e-03,  1.36186366e-01],
                              [ 5.43975403e-04, -1.08359291e-02, -9.99941142e-01, -1.88119571e-02],
                              [ 1.22238720e-01,  9.92443176e-01, -1.06881781e-02,  7.19721945e-02],
                              [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
    T_world2marker = np.linalg.inv(T_marker2world)
    
    # Transform gt_track_3d (n_frames, n_pts, 3)
    orig_shape = gt_track_3d.shape
    gt_points_homog = np.concatenate([gt_track_3d.reshape(-1, 3), np.ones((gt_track_3d.size // 3, 1))], axis=-1)
    gt_track_3d = (T_world2marker @ gt_points_homog.T).T[:, :3].reshape(orig_shape)

    # Locate the index of corresponding point index in the vertices, if nan, then ignore the points
    mask = ~np.isnan(gt_track_3d[0]).any(axis=1)

    kdtree = KDTree(vertices[0])
    dis, idx = kdtree.query(gt_track_3d[0][mask])

    num_frames = min(len(vertices), len(gt_track_3d))
    print(f"Eval frames: {num_frames} (Split says: {split['test'][1]})")

    valid_train_frame = min(train_frame, num_frames)
    valid_test_frame = min(test_frame, num_frames)

    train_track_error = evaluate_prediction(
        1, valid_train_frame, vertices, gt_track_3d, idx, mask
    )
    test_track_error = evaluate_prediction(
        valid_train_frame, valid_test_frame, vertices, gt_track_3d, idx, mask
    )
    writer.writerow([case_name, train_track_error, test_track_error])

    # WandB logging
    if args.ep_idx is not None:
        # Infer object name from base_path
        obj_name = os.path.basename(base_path)
        run_name = f"{obj_name}_ep_{args.ep_idx}"
        
        # Check if there is already an active run
        if wandb.run is None:
            wandb.init(project="deformable_dynamics", name=run_name, resume="allow", config={"method": "PhysTwin"})
        
        wandb.log({
            "train/track_error": train_track_error,
            "test/track_error": test_track_error,
        })

file.close()
