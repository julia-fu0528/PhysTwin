import glob
import pickle
import json
import torch
import csv
import numpy as np
import os
import argparse
import wandb
from pytorch3d.loss import chamfer_distance


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Chamfer Distance")
    parser.add_argument("--base_path", type=str, default="./data/different_types",
                        help="Path to ground truth data")
    parser.add_argument("--prediction_dir", type=str, default="./experiments",
                        help="Path to experiment outputs")
    parser.add_argument("--output_file", type=str, default="results/chamfer_results.csv",
                        help="Path to output CSV file")
    parser.add_argument("--ep_idx", type=int, default=None,
                        help="Specific episode index to evaluate")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Skip WandB logging")
    return parser.parse_args()


args = parse_args()
prediction_dir = args.prediction_dir
base_path = args.base_path
output_file = args.output_file

if args.ep_idx is not None and output_file == "results/chamfer_results.csv":
    obj_name = os.path.basename(args.base_path.rstrip("/"))
    output_file = f"results/{obj_name}_ep_{args.ep_idx}_chamfer.csv"

os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

def evaluate_prediction(
    start_frame,
    end_frame,
    vertices,
    object_points,
    object_visibilities,
    object_motions_valid,
    num_original_points,
    num_surface_points,
):
    chamfer_errors = []

    if not isinstance(vertices, torch.Tensor):
        vertices = torch.tensor(vertices, dtype=torch.float32)
    if not isinstance(object_points, torch.Tensor):
        object_points = torch.tensor(object_points, dtype=torch.float32)
    if not isinstance(object_visibilities, torch.Tensor):
        object_visibilities = torch.tensor(object_visibilities, dtype=torch.bool)
    if not isinstance(object_motions_valid, torch.Tensor):
        object_motions_valid = torch.tensor(object_motions_valid, dtype=torch.bool)

    for frame_idx in range(start_frame, end_frame):
        if frame_idx >= len(vertices) or frame_idx >= len(object_points):
            break
            
        x = vertices[frame_idx]
        current_object_points = object_points[frame_idx]
        current_object_visibilities = object_visibilities[frame_idx]
        
        # Compute the single-direction chamfer loss for the object points
        chamfer_object_points = current_object_points[current_object_visibilities]
        chamfer_x = x[:num_surface_points]
        
        if len(chamfer_object_points) == 0 or len(chamfer_x) == 0:
            continue
            
        # The GT chamfer_object_points can be partial,first find the nearest in second
        chamfer_error = chamfer_distance(
            chamfer_object_points.unsqueeze(0),
            chamfer_x.unsqueeze(0),
            single_directional=True,
            norm=1,  # Get the L1 distance
        )[0]

        chamfer_errors.append(chamfer_error.item())

    if len(chamfer_errors) == 0:
        return {
            "frame_len": 0,
            "chamfer_error": 0.0,
        }

    chamfer_errors = np.array(chamfer_errors)

    results = {
        "frame_len": len(chamfer_errors),
        "chamfer_error": np.mean(chamfer_errors),
    }

    return results


if __name__ == "__main__":
    file = open(output_file, mode="w", newline="", encoding="utf-8")
    writer = csv.writer(file)

    writer.writerow(
        [
            "Case Name",
            "Train Frame Num",
            "Train Chamfer Error",
            "Test Frame Num",
            "Test Chamfer Error",
        ]
    )

    if args.ep_idx is not None:
        case_name = f"episode_{args.ep_idx}"
        dir_names = [os.path.join(prediction_dir, case_name)]
    else:
        dir_names = sorted(glob.glob(f"{prediction_dir}/episode_*"))

    for dir_name in dir_names:
        case_name = os.path.basename(dir_name)
        print(f"Processing {case_name}")

        # Read the trajectory data
        with open(f"{dir_name}/inference.pkl", "rb") as f:
            vertices = pickle.load(f)

        # Read the GT object points and masks
        with open(f"{base_path}/{case_name}/final_data.pkl", "rb") as f:
            data = pickle.load(f)

        object_points = data["object_points"]
        surface_points = data["surface_points"]
        
        # Transform points from world space to marker space
        T_marker2world = np.array([[ 9.92500579e-01, -1.22225711e-01,  1.86443478e-03,  1.36186366e-01],
                                  [ 5.43975403e-04, -1.08359291e-02, -9.99941142e-01, -1.88119571e-02],
                                  [ 1.22238720e-01,  9.92443176e-01, -1.06881781e-02,  7.19721945e-02],
                                  [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
        T_world2marker = np.linalg.inv(T_marker2world)
        
        # Transform object_points (n_frames, n_pts, 3)
        orig_shape = object_points.shape
        object_points_homog = np.concatenate([object_points.reshape(-1, 3), np.ones((object_points.size // 3, 1))], axis=-1)
        object_points = (T_world2marker @ object_points_homog.T).T[:, :3].reshape(orig_shape)
        
        # Transform surface_points (n_surface_pts, 3)
        surface_points_homog = np.concatenate([surface_points, np.ones((surface_points.shape[0], 1))], axis=-1)
        surface_points = (T_world2marker @ surface_points_homog.T).T[:, :3]
        
        object_visibilities = data["object_visibilities"]
        object_motions_valid = data["object_motions_valid"]
        num_original_points = object_points.shape[1]
        num_surface_points = num_original_points + surface_points.shape[0]

        # read the train/test split
        with open(f"{base_path}/{case_name}/split.json", "r") as f:
            split = json.load(f)
        
        offset = split["train"][0]
        train_frame = split["train"][1] - offset
        test_frame = split["test"][1] - offset

        num_frames = min(vertices.shape[0], object_points.shape[0])
        print(f"Eval frames: {num_frames} (Split says: {split['test'][1]})")

        if num_frames == 0:
            print(f"Skipping {case_name} since num_frames is 0")
            writer.writerow([case_name, 0, 0.0, 0, 0.0])
            continue

        valid_train_frame = min(train_frame, num_frames)
        valid_test_frame = min(test_frame, num_frames)

        # Do the statistics on train split, only evalaute from the 2nd frame
        results_train = evaluate_prediction(
            1,
            valid_train_frame,
            vertices,
            object_points,
            object_visibilities,
            object_motions_valid,
            num_original_points,
            num_surface_points,
        )
        results_test = evaluate_prediction(
            valid_train_frame,
            valid_test_frame,
            vertices,
            object_points,
            object_visibilities,
            object_motions_valid,
            num_original_points,
            num_surface_points,
        )

        writer.writerow(
            [
                case_name,
                results_train["frame_len"],
                results_train["chamfer_error"],
                results_test["frame_len"],
                results_test["chamfer_error"],
            ]
        )

        # WandB logging
        if args.ep_idx is not None and not args.no_wandb:
            # Infer object name from base_path
            obj_name = os.path.basename(base_path)
            run_name = f"{obj_name}_ep_{args.ep_idx}"
            
            # Check if there is already an active run
            if wandb.run is None:
                wandb.init(project="deformable_dynamics", name=run_name, resume="allow", config={"method": "PhysTwin"})
            
            wandb.log({
                "train/chamfer_error": results_train["chamfer_error"],
                "test/chamfer_error": results_test["chamfer_error"],
                "train/chamfer_frame_num": results_train["frame_len"],
                "test/chamfer_frame_num": results_test["frame_len"],
            })
            
    file.close()
