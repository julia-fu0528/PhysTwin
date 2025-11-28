"""
Visualize control points over time as spheres/points without Gaussians.
This helps debug whether control point trajectories are correct.
"""
import os
import pickle
import numpy as np
import torch
import open3d as o3d
from argparse import ArgumentParser
from tqdm import tqdm


def visualize_control_points_trajectory(ctrl_pts_path, output_dir=None, frame_skip=1, sphere_radius=0.01, max_points=100):
    """
    Visualize control points from inference.pkl as spheres over time.
    
    Args:
        ctrl_pts_path: Path to inference.pkl file
        output_dir: Directory to save visualization images/videos (optional)
        frame_skip: Only visualize every Nth frame (for performance)
        sphere_radius: Radius of spheres representing control points
        max_points: Maximum number of points to visualize (subsamples if needed)
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
    spheres = []
    colors = np.random.rand(n_ctrl_pts, 3)  # Random color per control point
    
    for i in range(n_ctrl_pts):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
        sphere.paint_uniform_color(colors[i])
        spheres.append(sphere)
        vis.add_geometry(sphere)
    
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
        
        for i, sphere in enumerate(spheres):
            # Reset to origin first
            center = sphere.get_center()
            sphere.translate(-center, relative=True)
            # Move to new position
            sphere.translate(current_positions[i], relative=True)
            vis.update_geometry(sphere)
        
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
    parser.add_argument("--transform_to_marker", action="store_true",
                        help="Transform control points from world to marker space (same as gs_render_dynamics.py)")
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
    
    # Save transformed points temporarily for visualization
    temp_path = ctrl_pts_path.replace("inference.pkl", "inference_vis.pkl")
    with open(temp_path, "wb") as f:
        pickle.dump(ctrl_pts, f)
    
    visualize_control_points_trajectory(
        temp_path,
        frame_skip=args.frame_skip,
        sphere_radius=args.sphere_radius,
        max_points=args.max_points
    )
    
    # Clean up temp file
    if os.path.exists(temp_path):
        os.remove(temp_path)

