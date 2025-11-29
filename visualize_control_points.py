"""
Visualize control points over time as spheres/points without Gaussians.
This helps debug whether control point trajectories are correct.
Also visualizes subsampled Gaussians to verify coordinate system alignment.
"""
import os
import pickle
import numpy as np
import torch
import open3d as o3d
from argparse import ArgumentParser
from tqdm import tqdm
from gaussian_splatting.scene import Scene
from gaussian_splatting.arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_splatting.gaussian_renderer import GaussianModel
import sys

def visualize_control_points_trajectory(ctrl_pts_path, gaussians=None, output_dir=None, frame_skip=1, sphere_radius=0.01, max_points=100, max_gaussians=500, transform_gaussians=False):
    """
    Visualize control points from inference.pkl as spheres over time.
    Optionally also visualize subsampled Gaussians to verify coordinate alignment.
    
    Args:
        ctrl_pts_path: Path to inference.pkl file
        gaussians: Optional GaussianModel to visualize alongside control points
        output_dir: Directory to save visualization images/videos (optional)
        frame_skip: Only visualize every Nth frame (for performance)
        sphere_radius: Radius of spheres representing control points
        max_points: Maximum number of points to visualize (subsamples if needed)
        max_gaussians: Maximum number of Gaussians to visualize (subsamples if needed)
    """
    # Load control points
    print(f"Loading control points from: {ctrl_pts_path}")
    with open(ctrl_pts_path, "rb") as f:
        ctrl_pts = pickle.load(f)  # (n_frames, n_ctrl_pts, 3)
    
    if isinstance(ctrl_pts, torch.Tensor):
        ctrl_pts = ctrl_pts.cpu().numpy()
    
    print(f"Control points shape: {ctrl_pts.shape}")
    print(f"Number of frames: {ctrl_pts.shape[0]}")
    print(f"Number of control points per frame: {ctrl_pts.shape[1]}")
    
    # Subsample points if there are too many
    n_points = ctrl_pts.shape[1]
    if n_points > max_points:
        # Uniformly subsample points
        indices = np.linspace(0, n_points - 1, max_points, dtype=int)
        ctrl_pts = ctrl_pts[:, indices, :]
        print(f"Subsampled to {max_points} points (from {n_points})")
    else:
        print(f"Using all {n_points} points")
    
    # Check if control points are actually moving
    first_frame = ctrl_pts[0]
    last_frame = ctrl_pts[-1]
    displacement = np.linalg.norm(last_frame - first_frame, axis=-1)  # (n_ctrl_pts,)
    
    print(f"\n=== Control Point Movement Analysis ===")
    print(f"Max displacement: {displacement.max():.4f}")
    print(f"Min displacement: {displacement.min():.4f}")
    print(f"Mean displacement: {displacement.mean():.4f}")
    print(f"Control points that moved > 0.01: {(displacement > 0.01).sum()}/{len(displacement)}")
    
    if displacement.max() < 0.01:
        print("\n⚠️  WARNING: Control points barely moved! Problem is likely upstream:")
        print("   - Check if T_world2marker transformation is correct")
        print("   - Check if inference.pkl contains the right data")
        print("   - Check if control points are in the correct coordinate frame")
    else:
        print("\n✓ Control points are moving. If Gaussians don't move correctly,")
        print("  the problem is likely in weights/mapping to Gaussians (LBS).")
    
    # Create visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1920, height=1080, visible=True)
    
    # Create coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    vis.add_geometry(coord_frame)
    
    # Create spheres for each control point
    n_ctrl_pts = ctrl_pts.shape[1]
    ctrl_spheres = []
    ctrl_colors = np.random.rand(n_ctrl_pts, 3)  # Random color per control point
    
    for i in range(n_ctrl_pts):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
        sphere.paint_uniform_color(ctrl_colors[i])
        ctrl_spheres.append(sphere)
        vis.add_geometry(sphere)
    
    # Optionally add Gaussians for coordinate system verification
    gaussian_spheres = []
    if gaussians is not None:
        gaussian_xyz = gaussians.get_xyz.cpu().detach().numpy()
        
        # Optionally transform Gaussians from world to marker space
        # (if they're already in marker space, this won't be needed)
        if transform_gaussians:
            aruco_results_path = '/users/wfu16/data/users/wfu16/datasets/2025-10-23_snapshot_julia_aruco/aruco_results/avg_marker2world.npy'
            if os.path.exists(aruco_results_path):
                T_marker2world = np.load(aruco_results_path)
                T_marker2world = np.array([[ 9.92457290e-01, -1.22580045e-01,  1.63125912e-03,  3.31059452e-01],
                              [ 2.70205336e-04, -1.11191912e-02, -9.99938143e-01,  1.90897759e-01],
                              [ 1.22590601e-01,  9.92396340e-01, -1.10022006e-02,  2.75183546e-01],
                              [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
                print(f"EXISTS T_marker2world")
                T_world2marker = np.linalg.inv(T_marker2world)
                # Transform Gaussian positions: gaussian_xyz_marker = T_world2marker @ gaussian_xyz_world
                gaussian_xyz_homogeneous = np.concatenate([gaussian_xyz, np.ones((gaussian_xyz.shape[0], 1))], axis=-1)  # (n_gaussians, 4)
                gaussian_xyz_transformed = (T_world2marker @ gaussian_xyz_homogeneous.T).T[:, :3]  # (n_gaussians, 3)
                gaussian_xyz = gaussian_xyz_transformed
                print(f"Transformed Gaussians from world to marker space using T_world2marker")
            else:
                print(f"Warning: ArUco calibration file not found, Gaussians not transformed")
        
        n_gaussians = gaussian_xyz.shape[0]
        
        # Subsample Gaussians if needed
        if n_gaussians > max_gaussians:
            indices = np.linspace(0, n_gaussians - 1, max_gaussians, dtype=int)
            gaussian_xyz = gaussian_xyz[indices]
            print(f"Subsampled Gaussians to {max_gaussians} points (from {n_gaussians})")
        else:
            print(f"Using all {n_gaussians} Gaussians")
        
        # Check distance between Gaussians and control points
        dist_to_ctrl = np.linalg.norm(gaussian_xyz[:, None, :] - ctrl_pts[0][None, :, :], axis=-1)  # (n_gaussians, n_ctrl_pts)
        min_dist = dist_to_ctrl.min()
        mean_dist = dist_to_ctrl.mean()
        print(f"\n=== Coordinate System Check ===")
        print(f"Gaussians xyz range: [{gaussian_xyz.min():.4f}, {gaussian_xyz.max():.4f}]")
        print(f"Gaussians xyz mean: {gaussian_xyz.mean(axis=0)}")
        print(f"Control points (frame 0) range: [{ctrl_pts[0].min():.4f}, {ctrl_pts[0].max():.4f}]")
        print(f"Control points (frame 0) mean: {ctrl_pts[0].mean(axis=0)}")
        print(f"Min distance from Gaussians to control points: {min_dist:.4f}")
        print(f"Mean distance from Gaussians to control points: {mean_dist:.4f}")
        if min_dist > 1.0:
            print(f"⚠️  WARNING: Gaussians and control points are far apart! Coordinate mismatch likely.")
        if mean_dist > 0.5:
            print(f"⚠️  WARNING: Mean distance is large. Check coordinate system alignment.")
        
        # Create spheres for Gaussians (smaller, different color)
        gaussian_radius = sphere_radius * 0.5  # Make Gaussians smaller
        gaussian_color = np.array([0.0, 1.0, 0.0])  # Green for Gaussians
        
        for i in range(len(gaussian_xyz)):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=gaussian_radius)
            sphere.paint_uniform_color(gaussian_color)
            sphere.translate(gaussian_xyz[i])
            gaussian_spheres.append(sphere)
            vis.add_geometry(sphere)
        
        print(f"Added {len(gaussian_spheres)} Gaussian points (green) for coordinate verification")
    
    # Animation loop
    print("\n=== Starting Visualization ===")
    print("Controls:")
    print("  - Close window to quit")
    print("  - Mouse: Rotate/zoom view")
    print("  - Animation will loop continuously")
    
    frame_idx = 0
    
    # Main animation loop
    import time
    frame_time = 0.05  # 20 fps
    
    while True:
        # Update sphere positions
        current_positions = ctrl_pts[frame_idx]
        
        for i, sphere in enumerate(ctrl_spheres):
            # Reset to origin first
            center = sphere.get_center()
            sphere.translate(-center, relative=True)
            # Move to new position
            sphere.translate(current_positions[i], relative=True)
            vis.update_geometry(sphere)
        
        # Gaussians stay fixed (they don't move with control points)
        
        vis.poll_events()
        vis.update_renderer()
        
        # Check if window was closed
        if not vis.poll_events():
            break
        
        frame_idx = (frame_idx + frame_skip) % ctrl_pts.shape[0]
        
        if frame_idx == 0:
            print(f"Loop completed. Total frames: {ctrl_pts.shape[0]}")
        
        time.sleep(frame_time)
    
    vis.destroy_window()
    print("Visualization closed.")


if __name__ == "__main__":
    parser = ArgumentParser(description="Visualize control points from inference.pkl")
    parser.add_argument("--base_path", type=str, required=True,
                        help="Base path to dataset (e.g., /oscar/data/gdk/hli230/projects/vitac-particle)")
    parser.add_argument("--case_name", type=str, required=True,
                        help="Case name (e.g., 008-pink-cloth)")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="Experiment name (default: uses case_name)")
    parser.add_argument("--frame_skip", type=int, default=1,
                        help="Only visualize every Nth frame (default: 1)")
    parser.add_argument("--sphere_radius", type=float, default=0.01,
                        help="Radius of spheres representing control points (default: 0.01)")
    parser.add_argument("--max_points", type=int, default=100,
                        help="Maximum number of points to visualize (subsamples if needed, default: 100)")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to Gaussian Splatting model directory (e.g., /path/to/gaussian_output/scene_name)")
    parser.add_argument("--source_path", type=str, default=None,
                        help="Path to source data directory (e.g., /path/to/dataset/scene_name/episode_0000)")
    parser.add_argument("--iteration", type=int, default=-1,
                        help="Iteration to load (default: -1 for latest)")
    parser.add_argument("--max_gaussians", type=int, default=500,
                        help="Maximum number of Gaussians to visualize (default: 500)")
    parser.add_argument("--show_gaussians", action="store_true",
                        help="Load and visualize Gaussians alongside control points")
    parser.add_argument("--transform_gaussians", action="store_true",
                        help="Transform Gaussians from world to marker space using T_world2marker")
    args = parser.parse_args()
    
    # Construct path to inference.pkl (same logic as gs_render_dynamics.py)
    if args.exp_name is None:
        exp_name = args.case_name
    else:
        exp_name = args.exp_name
    
    ctrl_pts_path = f"{args.base_path}/{args.case_name}/experiments/episode_0000/inference.pkl"
    
    if not os.path.exists(ctrl_pts_path):
        print(f"Error: inference.pkl not found at: {ctrl_pts_path}")
        print(f"Please check --base_path and --case_name")
        exit(1)
    
    # Load and optionally transform control points
    with open(ctrl_pts_path, "rb") as f:
        ctrl_pts = pickle.load(f)
    
    if isinstance(ctrl_pts, torch.Tensor):
        ctrl_pts = ctrl_pts.cpu().numpy()
    
    # Optionally load Gaussians
    gaussians = None
    if args.show_gaussians:
        if args.model_path is None or args.source_path is None:
            print("Error: --model_path and --source_path required when using --show_gaussians")
            exit(1)
        
        print(f"\n=== Loading Gaussian Splatting Model ===")
        print(f"Model path: {args.model_path}")
        print(f"Source path: {args.source_path}")
        print(f"Iteration: {args.iteration}")
        
        # Create ModelParams similar to gs_render_dynamics.py
        class SimpleModelParams:
            def __init__(self):
                self.model_path = args.model_path
                self.source_path = args.source_path
                self.sh_degree = 3
                self.white_background = False
                self.isotropic = False
                self.eval = False
                self.data_device = "cuda"  # Required by camera_utils.py
                self.images = "images"  # For scene loading
                self.depths = ""  # For scene loading
                self.train_test_exp = False  # For scene loading
                self.use_masks = False  # For QQTT scene loading
                self.gs_init_opt = "pcd"  # For QQTT scene loading
                self.pts_per_triangles = 30  # For QQTT scene loading
                self.use_high_res = False  # For QQTT scene loading
        
        dataset = SimpleModelParams()
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False, 
                     start_frame=0, end_frame=60000, num_frames=60000)
        print(f"Loaded {gaussians.get_xyz.shape[0]} Gaussians")
    
    # Save transformed points temporarily for visualization
    temp_path = ctrl_pts_path.replace("inference.pkl", "inference_vis.pkl")
    with open(temp_path, "wb") as f:
        pickle.dump(ctrl_pts, f)
    
    visualize_control_points_trajectory(
        temp_path,
        gaussians=gaussians,
        frame_skip=args.frame_skip,
        sphere_radius=args.sphere_radius,
        max_points=args.max_points,
        max_gaussians=args.max_gaussians,
        transform_gaussians=args.transform_gaussians
    )
    
    # Clean up temp file
    if os.path.exists(temp_path):
        os.remove(temp_path)

