# Load your point cloud data and analyze Z values
import numpy as np
import glob
# Load your observation.ply or point cloud data
root_dir = "/users/wfu16/data/users/wfu16/datasets/2025-10-14_julia_umi/episode_0000/pcd_clean"
points = np.load(sorted(glob.glob(f"{root_dir}/*.npz"))[600])["pts"]
print(f"x range: {points[:, 0].min():.3f} to {points[:, 0].max():.3f} meters")
print(f"y range: {points[:, 1].min():.3f} to {points[:, 1].max():.3f} meters")
print(f"z range: {points[:, 2].min():.3f} to {points[:, 2].max():.3f} meters")
print(f"x range: {points[:, 0].max() - points[:, 0].min():.3f} meters")
print(f"y range: {points[:, 1].max() - points[:, 1].min():.3f} meters")
print(f"z range: {points[:, 2].max() - points[:, 2].min():.3f} meters")
print(f"x mean: {points[:, 0].mean():.3f} meters")
print(f"y mean: {points[:, 1].mean():.3f} meters")
print(f"z mean: {points[:, 2].mean():.3f} meters")
print(f"x median: {np.median(points[:, 0]):.3f} meters")
print(f"y median: {np.median(points[:, 1]):.3f} meters")
print(f"z median: {np.median(points[:, 2]):.3f} meters")
z_values = points[:, 2]  # Z coordinates

print(f"Z range: {z_values.min():.3f} to {z_values.max():.3f} meters")
print(f"Z mean: {z_values.mean():.3f} meters")
print(f"Z median: {np.median(z_values):.3f} meters")

# Find the most common Z value (likely ground)
hist, bins = np.histogram(z_values, bins=100)
ground_z = bins[np.argmax(hist)]
print(f"Most common Z (likely ground): {ground_z:.3f} meters")