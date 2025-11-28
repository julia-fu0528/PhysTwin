import numpy as np
import scipy.interpolate
import pickle
import os
from argparse import ArgumentParser


def normalize(v):
    """Normalize a vector."""
    return v/np.linalg.norm(v)


def viewmatrix(lookdir: np.ndarray, up: np.ndarray,
               position: np.ndarray) -> np.ndarray:
    """Construct lookat view matrix."""
    vec2 = normalize(lookdir)
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, position], axis=1)
    return m


def generate_interpolated_path(poses: np.ndarray,
                               n_interp: int,
                               spline_degree: int = 5,
                               smoothness: float = .03,
                               rot_weight: float = .1):
    """Creates a smooth spline path between input keyframe camera poses.
    Adapted from https://github.com/google-research/multinerf/blob/main/internal/camera_utils.py
    Spline is calculated with poses in format (position, lookat-point, up-point).

    Args:
        poses: (n, 3, 4) array of input pose keyframes.
        n_interp: returned path will have n_interp * (n - 1) total poses.
        spline_degree: polynomial degree of B-spline.
        smoothness: parameter for spline smoothing, 0 forces exact interpolation.
        rot_weight: relative weighting of rotation/translation in spline solve.

    Returns:
        Array of new camera poses with shape (n_interp * (n - 1), 3, 4).
    """

    def poses_to_points(poses, dist):
        """Converts from pose matrices to (position, lookat, up) format."""
        pos = poses[:, :3, -1]
        lookat = poses[:, :3, -1] - dist * poses[:, :3, 2]
        up = poses[:, :3, -1] + dist * poses[:, :3, 1]
        return np.stack([pos, lookat, up], 1)

    def points_to_poses(points):
        """Converts from (position, lookat, up) format to pose matrices."""
        return np.array([viewmatrix(p - l, u - p, p) for p, l, u in points])

    def interp(points, n, k, s):
        """Runs multidimensional B-spline interpolation on the input points."""
        sh = points.shape
        pts = np.reshape(points, (sh[0], -1))
        k = min(k, sh[0] - 1)
        tck, _ = scipy.interpolate.splprep(pts.T, k=k, s=s)
        u = np.linspace(0, 1, n, endpoint=False)
        new_points = np.array(scipy.interpolate.splev(u, tck))
        new_points = np.reshape(new_points.T, (n, sh[1], sh[2]))
        return new_points

    points = poses_to_points(poses, dist=rot_weight)
    new_points = interp(points,
                        n_interp * (points.shape[0] - 1),
                        k=spline_degree,
                        s=smoothness)
    return points_to_poses(new_points)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--base_path", type=str, required=True, help="Base path to data directory")
    parser.add_argument("--case_name", type=str, required=True, help="Case/scene name")
    parser.add_argument("--n_interp", type=int, default=50, help="Number of interpolated poses between keyframes")
    parser.add_argument("--use_all_cameras", action="store_true", help="Use all cameras instead of just first 3")
    parser.add_argument("--close_loop", action="store_true", help="Close the loop by returning to first camera")
    args = parser.parse_args()
    
    base_path = args.base_path
    case_name = args.case_name
    n_interp = args.n_interp
    
    scene_dir = os.path.join(base_path, case_name)
    print(f'Processing {case_name}')
    
    # Load camera data using same approach as train_warp.py
    camera_path = os.path.join(scene_dir, 'episode_0000', 'calibrate.pkl')
    assert os.path.exists(camera_path), f"Camera file not found: {camera_path}"
    with open(camera_path, 'rb') as f:
        c2ws = pickle.load(f)
    
    # # Apply T_world2marker transformation (same as train_warp.py)
    # # Try to load from ArUco calibration, fallback to hardcoded value
    # aruco_results_path = '/users/wfu16/data/users/wfu16/datasets/2025-10-23_snapshot_julia_aruco/aruco_results/avg_marker2world.npy'
    # if os.path.exists(aruco_results_path):
    #     T_marker2world = np.load(aruco_results_path)
    #     print(f"Loaded T_marker2world from ArUco calibration: {aruco_results_path}")
    # else:
    #     # Fallback to hardcoded value
    T_marker2world = np.array([[ 9.92457290e-01, -1.22580045e-01,  1.63125912e-03,  3.31059452e-01],
                                [ 2.70205336e-04, -1.11191912e-02, -9.99938143e-01,  1.90897759e-01],
                                [ 1.22590601e-01,  9.92396340e-01, -1.10022006e-02,  2.75183546e-01],
                                [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
    print("Using hardcoded T_marker2world (ArUco calibration file not found)")
    T_world2marker = np.linalg.inv(T_marker2world)
    c2ws = [T_world2marker @ c2w for c2w in c2ws]
    
    print(f"Found {len(c2ws)} camera poses")
    
    # Select keyframe poses
    if args.use_all_cameras:
        # Use all cameras
        keyframe_poses = c2ws
        if len(keyframe_poses) < 2:
            raise ValueError(f"Need at least 2 camera poses, but found {len(keyframe_poses)}")
        print(f"Using all {len(keyframe_poses)} cameras for interpolation")
    else:
        # Use first 3 cameras (original behavior)
        if len(c2ws) < 3:
            raise ValueError(f"Need at least 3 camera poses, but found {len(c2ws)}")
        keyframe_poses = c2ws[:3]
        print(f"Using first 3 cameras for interpolation")
    
    # Generate interpolated paths between consecutive keyframes
    interp_poses_list = []
    num_segments = len(keyframe_poses) - 1
    
    for i in range(num_segments):
        pose_start = keyframe_poses[i]
        pose_end = keyframe_poses[i + 1]
        poses_segment = np.stack([pose_start, pose_end], 0)[:, :3, :]
        interp_poses_segment = generate_interpolated_path(poses_segment, n_interp)
        interp_poses_list.append(interp_poses_segment)
        print(f"  Segment {i+1}/{num_segments}: {len(interp_poses_segment)} interpolated poses")
    
    # Optionally close the loop by returning to first camera
    if args.close_loop:
        pose_start = keyframe_poses[-1]
        pose_end = keyframe_poses[0]
        poses_segment = np.stack([pose_start, pose_end], 0)[:, :3, :]
        interp_poses_segment = generate_interpolated_path(poses_segment, n_interp)
        interp_poses_list.append(interp_poses_segment)
        print(f"  Closing loop: {len(interp_poses_segment)} interpolated poses")
    
    # Concatenate all interpolated poses
    interp_poses = np.concatenate(interp_poses_list, 0)
    
    # Convert to 4x4 matrices
    output_poses = [np.vstack([pose, np.array([0, 0, 0, 1])]) for pose in interp_poses]
    
    # Save interpolated poses
    output_path = os.path.join(scene_dir, 'interp_poses.pkl')
    pickle.dump(output_poses, open(output_path, 'wb'))
    print(f'\nTotal: {len(output_poses)} interpolated poses generated')
    print(f'Saved to {output_path}')
        