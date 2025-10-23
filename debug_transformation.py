#!/usr/bin/env python3
"""
Debug script to show the transformation matrix issue
"""

import numpy as np

def debug_transformation_matrices():
    """Debug the transformation matrix inconsistency"""
    
    print("=== Transformation Matrix Debug ===")
    
    # Your transformation values
    R_world_avg = np.array([
        [-9.92562955e-01, -1.21731464e-01, 4.81e-01],
        [ 1.96876498e-03, -1.20983548e-02, 9.99924874e-01],
        [-1.21716494e-01, 9.92489335e-01, 1.22480394e-02]
    ])
    
    t_world_avg = np.array([0.38417431, 0.18427508, 0.37144267])
    
    print(f"R_world_avg:\n{R_world_avg}")
    print(f"t_world_avg: {t_world_avg}")
    print()
    
    # What optimize_cma.py does:
    print("=== optimize_cma.py approach ===")
    T_marker2world = np.eye(4)
    T_marker2world[:3, :3] = R_world_avg
    T_marker2world[:3, 3] = t_world_avg
    T_world2marker = np.linalg.inv(T_marker2world)
    
    print(f"T_marker2world:\n{T_marker2world}")
    print(f"T_world2marker (inverse):\n{T_world2marker}")
    print()
    
    # What interactive_playground.py does:
    print("=== interactive_playground.py approach ===")
    ground_transform = np.eye(4)
    ground_transform[:3, :3] = R_world_avg
    ground_transform[:3, 3] = t_world_avg
    ground_transform_inv = np.linalg.inv(ground_transform)
    
    print(f"ground_transform:\n{ground_transform}")
    print(f"ground_transform_inv:\n{ground_transform_inv}")
    print()
    
    # Check if they're the same
    print("=== Comparison ===")
    print(f"T_marker2world == ground_transform: {np.allclose(T_marker2world, ground_transform)}")
    print(f"T_world2marker == ground_transform_inv: {np.allclose(T_world2marker, ground_transform_inv)}")
    print()
    
    # Test with a sample point
    test_point = np.array([0.0, 0.0, 0.5])  # 0.5m above origin
    test_point_homogeneous = np.append(test_point, 1)
    
    print("=== Point Transformation Test ===")
    print(f"Original point: {test_point}")
    
    # Transform using T_world2marker (from optimize_cma.py)
    transformed_1 = (T_world2marker @ test_point_homogeneous)[:3]
    print(f"Using T_world2marker: {transformed_1}")
    
    # Transform using ground_transform_inv (from interactive_playground.py)
    transformed_2 = (ground_transform_inv @ test_point_homogeneous)[:3]
    print(f"Using ground_transform_inv: {transformed_2}")
    
    print(f"Results are identical: {np.allclose(transformed_1, transformed_2)}")
    print()
    
    # The issue: Both are using the SAME transformation!
    print("=== THE PROBLEM ===")
    print("Both optimize_cma.py and interactive_playground.py are using the SAME")
    print("transformation matrix (R_world_avg, t_world_avg) but inverting it!")
    print()
    print("This means:")
    print("1. optimize_cma.py: T_world2marker = inv(T_marker2world)")
    print("2. interactive_playground.py: ground_transform_inv = inv(ground_transform)")
    print("3. But T_marker2world == ground_transform!")
    print()
    print("So both are applying the SAME transformation to points!")
    print("This could cause points to be 'a bit off' if the transformation")
    print("is being applied multiple times or inconsistently.")

def suggest_fix():
    """Suggest how to fix the transformation issue"""
    
    print("=== SUGGESTED FIX ===")
    print("The issue is likely that you need DIFFERENT transformations for:")
    print("1. Data processing (optimize_cma.py)")
    print("2. Ground plane visualization (interactive_playground.py)")
    print()
    print("Possible solutions:")
    print("1. Use the SAME transformation matrix in both places")
    print("2. Or use DIFFERENT transformation matrices if they serve different purposes")
    print("3. Check if the transformation is being applied multiple times")
    print()
    print("To debug further:")
    print("1. Print the transformation matrices before and after application")
    print("2. Check if points are being transformed multiple times")
    print("3. Verify the coordinate system conventions")

if __name__ == "__main__":
    debug_transformation_matrices()
    suggest_fix()
