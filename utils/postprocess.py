import os
import sys
import glob
import numpy as np
import open3d as o3d
import pickle


control_pcd_dir = "/users/wfu16/data/users/wfu16/datasets/2025-10-14_julia_umi/episode_0000/urdf_pcds"
control_pcd_files = sorted(glob.glob(f"{control_pcd_dir}/*.ply"))
start_frame = int(control_pcd_files[0].split("/")[-1].split(".")[0])
end_frame = int(control_pcd_files[-1].split("/")[-1].split(".")[0])
print(f"Control pcd files: {len(control_pcd_files)}")
print(f"Start frame: {start_frame}, End frame: {end_frame}")

object_pcd_dir = "/users/wfu16/data/users/wfu16/datasets/2025-10-14_julia_umi/episode_0000/pcd"
object_pcd_files = sorted(glob.glob(f"{object_pcd_dir}/frame_020000_time_*.ply"))[start_frame:end_frame+1]
print(f"Object pcd files: {len(object_pcd_files)}")
assert len(control_pcd_files) == len(object_pcd_files), "Control and object pcd files have different lengths"

# Process all frames
all_control_points = []
all_object_points = []
all_object_colors = []

for i, (control_file, object_file) in enumerate(zip(control_pcd_files, object_pcd_files)):
        print(f"{i}th: control file: {control_file}, object file: {object_file}")
        
        # Load control points
        control_pcd = o3d.io.read_point_cloud(control_file)
        control_points = np.asarray(control_pcd.points)
        all_control_points.append(control_points)
        
        # Load object points  
        object_pcd = o3d.io.read_point_cloud(object_file)
        object_points = np.asarray(object_pcd.points)
        objects_colors = np.asarray(object_pcd.colors)
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