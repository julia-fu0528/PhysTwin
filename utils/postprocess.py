import os
import sys
import glob
import numpy as np
import open3d as o3d
import pickle
import json
import cv2

from brics_utils import read_params, vis_extr, get_extr


def farthest_point_sampling_with_indices(points, num_samples):
    """
    return the indices of the sampled points
    
    Args:
        points: (N, 3) numpy array
        num_samples: int
    
    Returns:
        indices: (num_samples,) the indices of the sampled points
    """
    N = points.shape[0]
    
    if num_samples >= N:
        return np.arange(N)
    
    # initialize
    sampled_indices = np.zeros(num_samples, dtype=np.int32)
    distances = np.full(N, np.inf)
    
    # randomly select the first point
    current_idx = np.random.randint(0, N)
    sampled_indices[0] = current_idx
    
    # iteratively select the farthest point
    for i in range(1, num_samples):
        # update the distances
        current_point = points[current_idx]
        dists = np.linalg.norm(points - current_point, axis=1)
        distances = np.minimum(distances, dists)
        
        # select the farthest point
        current_idx = np.argmax(distances)
        sampled_indices[i] = current_idx
    
    return sampled_indices

control_pcd_dir = "/users/wfu16/data/users/wfu16/datasets/2025-10-14_julia_umi/episode_0000/urdf_pcds"
control_pcd_files = sorted(glob.glob(f"{control_pcd_dir}/*.ply"))
start_frame = int(control_pcd_files[0].split("/")[-1].split(".")[0])
# end_frame = int(control_pcd_files[-1].split("/")[-1].split(".")[0])
end_frame = 200
num_frames = end_frame - start_frame + 1
control_pcd_files = control_pcd_files[:num_frames]
print(f"Control pcd files: {len(control_pcd_files)}")
print(f"Start frame: {start_frame}, End frame: {end_frame}")

object_pcd_dir = "/users/wfu16/data/users/wfu16/datasets/2025-10-14_julia_umi/episode_0000/pcd"
object_pcd_files = sorted(glob.glob(f"{object_pcd_dir}/frame_020000_time_*.ply"))[start_frame:end_frame+1]
colors_dir = "/users/wfu16/data/users/wfu16/datasets/2025-10-14_julia_umi/episode_0000"
cameras = [subdir for subdir in os.listdir(colors_dir) if "cam" in subdir]
# colors_files = sorted(glob.glob(f"{colors_dir}/*.png"))[start_frame:end_frame+1]
print(f"Object pcd files: {len(object_pcd_files)}")
# print(f"Colors files: {len(colors_files)}")
assert len(control_pcd_files) == len(object_pcd_files), "Control and object pcd files have different lengths"

# save split.json
split = {
    "frame_len": num_frames,
    "train": [start_frame, start_frame + int(num_frames * 0.7)],
    "test": [int(num_frames * 0.7), end_frame+1]
}
with open(f"/users/wfu16/data/users/wfu16/datasets/2025-10-14_julia_umi/episode_0000/split.json", "w") as f:
    json.dump(split, f)

print(f"Saved split.json to /users/wfu16/data/users/wfu16/datasets/2025-10-14_julia_umi/episode_0000/split.json")

# save metadata.json and calibrate.pkl
camera_dir = "/users/wfu16/data/users/wfu16/datasets/2025-10-14_julia_umi/calibration"
c2ws = np.load(f"{camera_dir}/extrinsics.npy")
# vis_extr(c2ws)
optim_path = "/users/wfu16/data/users/wfu16/datasets/2025-10-14_julia_umi/optim_params_undistorted.txt"
optim_params = read_params(optim_path)
extr_params = read_params("/users/wfu16/data/users/wfu16/datasets/2025-10-14_julia_umi/optim_params_undistorted.txt")

# c2ws = []
# for extr_param in extr_params:
#     if extr_param["cam_name"] not in cameras:
#         continue
#     c2w = get_extr(extr_param)
#     c2ws.append(c2w)
# c2ws = np.array(c2ws)
# print(f"c2ws: {c2ws.shape}")
intrs = []
# c2ws = []
for i in range(len(optim_params)):
    K = np.eye(3, dtype=float)
    K[0, 0] = optim_params[i]["fx"]
    K[1, 1] = optim_params[i]["fy"]
    K[0, 2] = optim_params[i]["cx"]
    K[1, 2] = optim_params[i]["cy"]
    intrs.append(K)
    extr = get_extr(optim_params[i])
    # c2ws.append(extr)
intrs = np.array(intrs)
# c2ws = np.array(c2ws)
# intrs = np.load(f"{camera_dir}/intrinsics.npy")
print(f"intrs: {intrs.shape}")
img_shape = cv2.imread(os.path.join(colors_dir, cameras[0], "undistorted_raw", "000000.png")).shape
print(f"img_shape type: {img_shape}")

metadata = {
    "intrinsics": intrs.tolist(),
    "WH": [img_shape[1], img_shape[0]],
    "fps": 30,
    "frame_num": num_frames,
    "start_frame": start_frame,
    "end_frame": end_frame,
    "cameras": cameras,
}

print(f"start_frame: {start_frame}, end_frame: {end_frame}")
with open(f"/users/wfu16/data/users/wfu16/datasets/2025-10-14_julia_umi/episode_0000/metadata.json", "w") as f:
    json.dump(metadata, f)

with open(f"/users/wfu16/data/users/wfu16/datasets/2025-10-14_julia_umi/episode_0000/calibrate.pkl", "wb") as f:
    pickle.dump(c2ws, f)

print(f"Saved metadata.json and calibrate.pkl to /users/wfu16/data/users/wfu16/datasets/2025-10-14_julia_umi/episode_0000")

# # fps
# fps_dir = "/users/wfu16/data/users/wfu16/datasets/2025-10-14_julia_umi/episode_0000/fps"
# num_samples = 4000
# os.makedirs(fps_dir, exist_ok=True)
# sampled_indices = None
# for i, object_file in enumerate(object_pcd_files):
#     output_path = object_file.replace("pcd", "fps")
#     object_pcd = o3d.io.read_point_cloud(object_file)
#     original_points = np.asarray(object_pcd.points)
#     if i == 0:
#         sampled_indices = farthest_point_sampling_with_indices(original_points, num_samples)
#         print(f"Frame {i}: {len(original_points)} -> {len(sampled_indices)} points")
#     downsampled_pcd = object_pcd.select_by_index(sampled_indices.tolist())
#     o3d.io.write_point_cloud(output_path, downsampled_pcd)
#     print(f"Saved downsampled pcd to {output_path}")

# Create final_data.pkl
# Process all frames
all_control_points = []
all_object_points = []
all_object_colors = []
sampled_indices = None
num_samples = 4000

for i, (control_file, object_file) in enumerate(zip(control_pcd_files, object_pcd_files)):
        print(f"{i}th: control file: {control_file}, object file: {object_file}")
        
        # Load control points
        control_pcd = o3d.io.read_point_cloud(control_file)
        control_points = np.asarray(control_pcd.points)
        all_control_points.append(control_points)
        
        # Load object points  
        object_pcd = o3d.io.read_point_cloud(object_file)
        object_points = np.asarray(object_pcd.points)
        if i == 0:
            sampled_indices = farthest_point_sampling_with_indices(object_points, num_samples)
            print(f"Frame {i}: {len(object_points)} -> {len(sampled_indices)} points")
        object_points = object_points[sampled_indices]
        objects_colors = np.asarray(object_pcd.colors)[sampled_indices]
        assert len(object_points) == len(objects_colors)
        all_object_points.append(object_points)
        all_object_colors.append(objects_colors)

# Convert to numpy arrays
control_points_array = np.array(all_control_points)
object_points_array = np.array(all_object_points)
object_colors_array = np.array(all_object_colors)

print(f"Control points array shape: {control_points_array.shape}")
print(f"Object points array shape: {object_points_array.shape}")
print(f"Object colors array shape: {object_colors_array.shape}")

object_visibilities = np.ones((object_points_array.shape[0], object_points_array.shape[1]), dtype=bool)
object_motions_valid = np.ones((object_points_array.shape[0], object_points_array.shape[1]), dtype=bool)

# Create final_data structure
final_data = {
    "object_points": object_points_array,
    "object_colors": object_colors_array,
    "object_visibilities": object_visibilities,
    "object_motions_valid": object_motions_valid,
    "controller_points": control_points_array,
    "surface_points": np.zeros((0, 3)),
    "interior_points": np.zeros((0, 3)),
}

# Save final_data.pkl
output_path = "/users/wfu16/data/users/wfu16/datasets/2025-10-14_julia_umi/episode_0000/final_data.pkl"

with open(output_path, "wb") as f:
    pickle.dump(final_data, f)
    
    print(f"Saved final_data.pkl to {output_path}")