#!/usr/bin/env python3
"""
Simple script to convert ArUco results to PhysTwin ground transform
"""

import numpy as np
import cv2
import os
import argparse

def aruco_to_ground_transform(aruco_output_dir, reference_camera='cam0'):
    """
    Convert ArUco detection results to ground transform for PhysTwin
    
    Args:
        aruco_output_dir: Directory containing ArUco detection results
        reference_camera: Which camera to use as reference (default: 'cam0')
    
    Returns:
        tuple: (ground_transform_matrix, R_world_avg, t_world_avg)
    """
    
    # Load average pose for reference camera
    cam_output_dir = os.path.join(aruco_output_dir, reference_camera)
    avg_rvec_path = os.path.join(cam_output_dir, 'avg_rvec.npy')
    avg_tvec_path = os.path.join(cam_output_dir, 'avg_tvec.npy')
    
    if not os.path.exists(avg_rvec_path) or not os.path.exists(avg_tvec_path):
        print(f"Error: ArUco results not found for camera {reference_camera}")
        print(f"Expected files:")
        print(f"  {avg_rvec_path}")
        print(f"  {avg_tvec_path}")
        return None, None, None
    
    # Load rotation and translation vectors
    avg_rvec = np.load(avg_rvec_path)
    avg_tvec = np.load(avg_tvec_path)
    
    print(f"Loaded ArUco results for camera {reference_camera}:")
    print(f"  Rotation vector: {avg_rvec.ravel()}")
    print(f"  Translation vector: {avg_tvec.ravel()}")
    
    # Convert rotation vector to rotation matrix
    R_marker_to_camera, _ = cv2.Rodrigues(avg_rvec)
    t_marker_to_camera = avg_tvec.ravel()
    
    print(f"\nMarker-to-camera transformation:")
    print(f"  Rotation matrix:\n{R_marker_to_camera}")
    print(f"  Translation: {t_marker_to_camera}")
    
    # Create marker-to-camera transformation matrix
    T_marker_to_camera = np.eye(4)
    T_marker_to_camera[:3, :3] = R_marker_to_camera
    T_marker_to_camera[:3, 3] = t_marker_to_camera
    
    # Invert to get camera-to-marker transformation
    T_camera_to_marker = np.linalg.inv(T_marker_to_camera)
    
    print(f"\nCamera-to-marker transformation:")
    print(f"  Rotation matrix:\n{T_camera_to_marker[:3, :3]}")
    print(f"  Translation: {T_camera_to_marker[:3, 3]}")
    
    # Extract rotation and translation for PhysTwin format
    R_world_avg = T_camera_to_marker[:3, :3]
    t_world_avg = T_camera_to_marker[:3, 3]
    
    # Create ground transform (this is what you use in PhysTwin)
    ground_transform = np.eye(4)
    ground_transform[:3, :3] = R_world_avg
    ground_transform[:3, 3] = t_world_avg
    
    print(f"\n=== PhysTwin Ground Transform ===")
    print(f"R_world_avg = np.array([")
    print(f"    [{R_world_avg[0,0]:.8e}, {R_world_avg[0,1]:.8e}, {R_world_avg[0,2]:.8e}],")
    print(f"    [{R_world_avg[1,0]:.8e}, {R_world_avg[1,1]:.8e}, {R_world_avg[1,2]:.8e}],")
    print(f"    [{R_world_avg[2,0]:.8e}, {R_world_avg[2,1]:.8e}, {R_world_avg[2,2]:.8e}]")
    print(f"])")
    print(f"")
    print(f"t_world_avg = np.array([{t_world_avg[0]:.8f}, {t_world_avg[1]:.8f}, {t_world_avg[2]:.8f}])")
    
    return ground_transform, R_world_avg, t_world_avg

def main():
    """Example usage"""
    
    parser = argparse.ArgumentParser(description='Convert ArUco results to PhysTwin ground transform')
    parser.add_argument('--aruco-output-dir', type=str, 
                       default="/users/wfu16/data/users/wfu16/datasets/2025-10-23_snapshot_julia_aruco/aruco_results",
                       help='Output directory from your ArUco script')
    parser.add_argument('--reference-camera', type=str, 
                       default="brics-odroid-001_cam0",
                       help='Which camera to use as reference')
    args = parser.parse_args()
    
    print("=== Converting ArUco Results to PhysTwin Ground Transform ===")
    
    ground_transform, R_world_avg, t_world_avg = aruco_to_ground_transform(
        args.aruco_output_dir, args.reference_camera
    )
    
    if ground_transform is not None:
        print(f"\n=== Usage in PhysTwin ===")
        print("Copy these values into your interactive_playground.py:")
        print("")
        print("R_world_avg = np.array([")
        print(f"    [{R_world_avg[0,0]:.8e}, {R_world_avg[0,1]:.8e}, {R_world_avg[0,2]:.8e}],")
        print(f"    [{R_world_avg[1,0]:.8e}, {R_world_avg[1,1]:.8e}, {R_world_avg[1,2]:.8e}],")
        print(f"    [{R_world_avg[2,0]:.8e}, {R_world_avg[2,1]:.8e}, {R_world_avg[2,2]:.8e}]")
        print("])")
        print("")
        print(f"t_world_avg = np.array([{t_world_avg[0]:.8f}, {t_world_avg[1]:.8f}, {t_world_avg[2]:.8f}])")
        print("")
        print("ground_transform = np.eye(4)")
        print("ground_transform[:3, :3] = R_world_avg")
        print("ground_transform[:3, 3] = t_world_avg")
        print("ground_transform = np.linalg.inv(ground_transform)  # Invert for PhysTwin")
        print("trainer.set_ground_transform(ground_transform)")
    else:
        print("Failed to generate ground transform")

if __name__ == "__main__":
    main()
