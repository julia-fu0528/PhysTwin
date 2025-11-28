import numpy as np
import os
import sys
import cv2
def compute_marker_to_world_from_multiple_cameras(cam2worlds, cam2markers):
    """
    Compute marker-to-world from multiple camera views.
    
    Args:
        cam2worlds: (N_cams, 4, 4) Camera-to-world extrinsics
        cam2markers: (N_cams, 4, 4) Camera-to-marker transformations
    
    Returns:
        marker2world: (4, 4) Average marker-to-world transformation
    """
    marker2worlds = []
    
    for cam2world, cam2marker in zip(cam2worlds, cam2markers):
        marker2cam = np.linalg.inv(cam2marker)
        marker2world = cam2world @ marker2cam
        marker2worlds.append(marker2world)
    
    marker2worlds = np.array(marker2worlds)
    
    # Average rotation matrices (with SVD for proper averaging)
    avg_R = np.mean([T[:3, :3] for T in marker2worlds], axis=0)
    U, _, Vt = np.linalg.svd(avg_R)
    avg_R = U @ Vt
    
    # Average translation
    avg_t = np.mean([T[:3, 3] for T in marker2worlds], axis=0)
    
    avg_marker2world = np.eye(4)
    avg_marker2world[:3, :3] = avg_R
    avg_marker2world[:3, 3] = avg_t
    
    return avg_marker2world, marker2worlds


# Usage:
extrs = np.load('/oscar/data/gdk/hli230/projects/vitac-particle/2025-11-17/calibration_optim/extrinsics.npy')  # Shape: (N_cams, 4, 4)
cameras = [subdir for subdir in os.listdir('/users/wfu16/data/users/wfu16/datasets/2025-10-23_snapshot_julia_aruco') if "cam" in subdir]

marker2cam_rots  = np.load('/users/wfu16/data/users/wfu16/datasets/2025-10-23_snapshot_julia_aruco/aruco_results/all_rvecs.npy', allow_pickle=True).item()
marker2cam_trans = np.load('/users/wfu16/data/users/wfu16/datasets/2025-10-23_snapshot_julia_aruco/aruco_results/all_tvecs.npy', allow_pickle=True).item()
# print(f"marker2cam_rots: {marker2cam_rots['brics-odroid-025_cam1']}")
valid_cameras = marker2cam_rots.keys()
valid_cameras_t = marker2cam_trans.keys()
assert valid_cameras == valid_cameras_t, "Valid cameras and valid cameras t are not the same"

cam2worlds = []
for i, extr in enumerate(extrs):
    if cameras[i] not in valid_cameras:
        continue
    cam2worlds.append(extr)
cam2worlds = np.array(cam2worlds)
print(f"cam2worlds: {cam2worlds.shape}")
assert cam2worlds.shape[0] == len(valid_cameras), "cam2worlds and valid_cameras have different lengths"

marker2cams = []
cam2markers = []
for key in marker2cam_rots.keys():
    rot = marker2cam_rots[key].reshape(3, 1)
    trans = marker2cam_trans[key].reshape(3, )
    marker2cam = np.eye(4)
    marker2cam[:3, :3] = cv2.Rodrigues(rot)[0]
    marker2cam[:3, 3] = trans
    try:
        cam2marker = np.linalg.inv(marker2cam)
    except np.linalg.LinAlgError:
        # Use pseudoinverse as fallback
        print(f"Using pseudoinverse for {key}")
        cam2marker = np.linalg.pinv(marker2cam)
    
    marker2cams.append(marker2cam)
    cam2markers.append(cam2marker)
marker2cams = np.array(marker2cams)
cam2markers = np.array(cam2markers)
print(f"marker2cams: {marker2cams.shape}")
assert marker2cams.shape[0] == len(valid_cameras), "marker2cams and valid_cameras have different lengths"
assert cam2markers.shape[0] == len(valid_cameras), "cam2markers and valid_cameras have different lengths"


avg_marker2world, all_marker2worlds = compute_marker_to_world_from_multiple_cameras(
    cam2worlds, cam2markers
)
if avg_marker2world[2,3] > 1.0:  # if > 1m above world origin
    avg_marker2world[:3,2] *= -1 
print("Marker position in world:", avg_marker2world[:3,3])
print("Marker up vector (Z):", avg_marker2world[:3,2])
print(f"avg_marker2world: {avg_marker2world}")
print(f"all_marker2worlds: {all_marker2worlds.shape}")
assert all_marker2worlds.shape[0] == len(valid_cameras), "all_marker2worlds and valid_cameras have different lengths"

# save the avg_marker2world and all_marker2worlds to a file
np.save('/users/wfu16/data/users/wfu16/datasets/2025-10-23_snapshot_julia_aruco/aruco_results/avg_marker2world.npy', avg_marker2world)
np.save('/users/wfu16/data/users/wfu16/datasets/2025-10-23_snapshot_julia_aruco/aruco_results/all_marker2worlds.npy', all_marker2worlds)

# def compute_marker_to_world(cam2world, cam2marker):
#     """
#     Compute marker-to-world transformation.
    
#     Args:
#         cam2world: (4, 4) Camera-to-world extrinsics
#         cam2marker: (4, 4) Camera-to-marker transformation
    
#     Returns:
#         marker2world: (4, 4) Marker-to-world transformation
#     """
#     # Invert camera-to-marker to get marker-to-camera
#     marker2cam = np.linalg.inv(cam2marker)
    
#     # Chain: marker -> camera -> world
#     marker2world = cam2world @ marker2cam
    
#     return marker2world


# # Example usage:
# cam2world = np.load('/users/wfu16/data/users/wfu16/datasets/2025-10-14_julia_umi/calibration/extrinsics.npy')  # Shape: (N_cams, 4, 4)
# marker2cam_rots  = np.load('/users/wfu16/data/users/wfu16/datasets/2025-10-23_snapshot_julia_aruco/aruco_results/all_rvecs.npy', allow_pickle=True).item()
# marker2cam_trans = np.load('/users/wfu16/data/users/wfu16/datasets/2025-10-23_snapshot_julia_aruco/aruco_results/all_tvecs.npy', allow_pickle=True).item()
# rot = marker2cam_rots['brics-odroid-001_cam1']
# trans = marker2cam_trans['brics-odroid-001_cam1']
# marker2cam = np.eye(4)
# marker2cam[:3, :3] = rot
# marker2cam[:3, 3] = trans[:, 0]
# cam2marker = np.linalg.inv(marker2cam)
# marker2world = compute_marker_to_world(cam2world, cam2marker)

# print(f"Marker-to-World transformation:\n{marker2world}")
# print(f"Marker position in world: {marker2world[:3, 3]}")
# print(f"Marker orientation in world:\n{marker2world[:3, :3]}")