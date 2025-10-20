import numpy as np
import cv2
import os
import argparse, csv, re, shutil
from pathlib import Path
from bisect import bisect_left
from typing import List, Tuple
import glob
import sys
import open3d as o3d

#########################################
# Camera calibration utils
#########################################

def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def get_intr(param, undistort=False):
    intr = np.eye(3)
    intr[0, 0] = param["fx_undist" if undistort else "fx"]
    intr[1, 1] = param["fy_undist" if undistort else "fy"]
    intr[0, 2] = param["cx_undist" if undistort else "cx"]
    intr[1, 2] = param["cy_undist" if undistort else "cy"]

    # TODO: Make work for arbitrary dist params in opencv
    dist = np.asarray([param["k1"], param["k2"], param["p1"], param["p2"]])

    return intr, dist


def get_rot_trans(param):
    qvec = np.asarray([param["qvecw"], param["qvecx"], param["qvecy"], param["qvecz"]])
    tvec = np.asarray([param["tvecx"], param["tvecy"], param["tvecz"]])
    r = qvec2rotmat(-qvec)
    
    # r = np.transpose(r)
    return r, tvec


def get_w2c(param):
    r, tvec = get_rot_trans(param)
    extr = np.vstack([np.hstack([r, tvec[:, None]]), np.zeros((1, 4))])
    extr[3, 3] = 1
    # extr = extr[:3]

    return extr

def get_extr_old(param):
    w2c = get_w2c(param)
    c2w = np.linalg.inv(w2c)
    
    # c2w_corrected = c2w.copy()
    # c2w_corrected[:3, [1, 2]] = c2w_corrected[:3, [2, 1]]  
    # c2w_corrected[:3, 2] = -c2w_corrected[:3, 2]  
    
    return c2w

def colmap_cam_to_world(param):
    qvec = np.asarray([param["qvecw"], param["qvecx"], param["qvecy"], param["qvecz"]])
    tvec = np.asarray([param["tvecx"], param["tvecy"], param["tvecz"]])
    R_w2c = qvec2rotmat(np.asarray(qvec, float))  # world->camera
    t_w2c = np.asarray(tvec, float)

    R_c2w = R_w2c.T
    t_c2w = -R_c2w @ t_w2c       # camera center in world

    T_c2w = np.eye(4)
    T_c2w[:3,:3] = R_c2w
    T_c2w[:3, 3] = t_c2w
    return T_c2w

def colmap_world_to_cam(param):
    qvec = np.asarray([param["qvecw"], param["qvecx"], param["qvecy"], param["qvecz"]])
    tvec = np.asarray([param["tvecx"], param["tvecy"], param["tvecz"]])
    R_w2c = qvec2rotmat(np.asarray(qvec, float))
    t_w2c = np.asarray(tvec, float)
    T_w2c = np.eye(4)
    T_w2c[:3,:3] = R_w2c
    T_w2c[:3, 3] = t_w2c
    return T_w2c


def get_extr(param):
    qvec = np.asarray([param["qvecw"], param["qvecx"], param["qvecy"], param["qvecz"]])
    tvec = np.asarray([param["tvecx"], param["tvecy"], param["tvecz"]])
    R_w2c = qvec2rotmat(qvec)           # NO transpose, NO minus
    t_w2c = tvec.astype(float)

    # Assemble both
    w2c = np.eye(4)
    w2c[:3, :3] = R_w2c
    w2c[:3, 3]  = t_w2c

    c2w = np.eye(4)
    c2w[:3, :3] = R_w2c.T
    c2w[:3, 3]  = -R_w2c.T @ t_w2c
    
    
    return c2w

def get_extr_metric(param):
    qvec = np.asarray([param["qvecw"], param["qvecx"], param["qvecy"], param["qvecz"]])
    tvec = np.asarray([0.8 * param["tvecx"], 0.8 * param["tvecy"], 0.8 * param["tvecz"]])
    R_w2c = qvec2rotmat(qvec)           # NO transpose, NO minus
    t_w2c = tvec.astype(float)

    # Assemble both
    w2c = np.eye(4)
    w2c[:3, :3] = R_w2c
    w2c[:3, 3]  = t_w2c

    c2w = np.eye(4)
    c2w[:3, :3] = R_w2c.T
    c2w[:3, 3]  = -R_w2c.T @ t_w2c
    
    
    return c2w

def reconvert_w2c_from_c2w(c2w):
    R_w2c = c2w[:3, :3].T
    t_c2w = c2w[:3, 3]
    t_w2c = -R_w2c @ t_c2w
    return t_w2c


def vis_extr(extrs):
    cam_frames = []
    for extr in extrs:
        cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        cam_frame.transform(extr)
        cam_frames.append(cam_frame)
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
    cam_frames.append(coordinate_frame)
    o3d.visualization.draw_geometries(cam_frames)

def read_params(params_path):
    params = np.loadtxt(
        params_path,
        dtype=[
            ("cam_id", int),
            ("width", int),
            ("height", int),
            ("fx", float),
            ("fy", float),
            ("cx", float),
            ("cy", float),
            ("k1", float),
            ("k2", float),
            ("p1", float),
            ("p2", float),
            ("cam_name", "<U22"),
            ("qvecw", float),
            ("qvecx", float),
            ("qvecy", float),
            ("qvecz", float),
            ("tvecx", float),
            ("tvecy", float),
            ("tvecz", float),
        ]
    )
    params = np.sort(params, order="cam_name")

    return params

def get_undistort_params(intr, dist, img_size):
    # new_intr = cv2.getOptimalNewCameraMatrix(intr, dist, img_size, alpha=1)
    new_intr = cv2.getOptimalNewCameraMatrix(intr, dist, img_size, alpha=0, centerPrincipalPoint=True)
    return new_intr

def undistort_image(intr, dist_intr, dist, img):
    result = cv2.undistort(img, intr, dist, None, dist_intr)
    return result

def read_to_skip(to_skip_path):
    to_skip = []
    if os.path.exists(to_skip_path):
        with open(to_skip_path, "r") as f:
            for cam_name in f.readlines():
                to_skip.append(cam_name.rstrip('\n'))

    return to_skip




#########################################
# Timestamp matching utils
#########################################

def parse_timestamp_file(timestamp_file: Path) -> List[Tuple[str, int]]:
    """
    Parse timestamp file to extract timestamp -> frame_number mapping.
    
    Args:
        timestamp_file: Path to the timestamp file
        
    Returns:
        List of (timestamp, frame_number) tuples, sorted by timestamp
    """
    timestamp_frame_pairs = []
    
    with open(timestamp_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                # Extract timestamp and frame number
                # Format: frame_1755632608552018_000000000000
                parts = line.split('_')
                if len(parts) >= 3:
                    timestamp = '_'.join(parts[1:-1])  # Get the timestamp part
                    frame_num = int(parts[-1])  # Get the frame number
                    timestamp_frame_pairs.append((timestamp, frame_num))
    
    # Sort by timestamp to maintain temporal order
    timestamp_frame_pairs.sort(key=lambda x: x[0])
    return timestamp_frame_pairs

def find_anchor_camera(camera_dirs: List[Path], vid_idx: int = 0) -> Tuple[Path, List[Tuple[str, int]]]:
    """
    Find the camera directory with the fewest frames (anchor camera).
    
    Args:
        camera_dirs: List of camera directory paths
        
    Returns:
        Tuple of (anchor_camera_dir, timestamp_frame_pairs)
    """
    min_frames = float('inf')
    anchor_camera = None
    anchor_timestamps = None
    
    for cam_dir in camera_dirs:
        ts_paths = [ts for ts in os.listdir(cam_dir) if ts.endswith(".txt")]
        # assert len(ts_paths) == 1, f"Expected 1 timestamps.txt file, got {len(ts_paths)}"
        timestamp_file = cam_dir / ts_paths[vid_idx]
        if timestamp_file.exists():
            timestamps = parse_timestamp_file(timestamp_file)
            if len(timestamps) < min_frames:
                min_frames = len(timestamps)
                anchor_camera = cam_dir
                anchor_timestamps = timestamps
    
    return anchor_camera, anchor_timestamps

def find_closest_timestamp(target_timestamp: str, available_timestamps: List[str]) -> Tuple[str, float]:
    """
    Find the closest timestamp from available timestamps.
    
    Args:
        target_timestamp: Target timestamp to match
        available_timestamps: List of available timestamps
        
    Returns:
        Tuple of (closest_timestamp, time_difference)
    """
    # Convert timestamp to numeric value for comparison
    target_numeric = int(target_timestamp.replace('_', ''))
    
    min_diff = float('inf')
    closest_timestamp = None
    
    for ts in available_timestamps:
        ts_numeric = int(ts.replace('_', ''))
        diff = abs(target_numeric - ts_numeric)
        if diff < min_diff:
            min_diff = diff
            closest_timestamp = ts
    
    return closest_timestamp, min_diff

def align_camera_to_anchor(cam_dir: Path, anchor_timestamps: List[Tuple[str, int]], 
                          output_dir: Path, max_time_diff: int = 1000000, vid_idx: int = 0):
    """
    Align a single camera to the anchor camera timestamps.
    
    Args:
        cam_dir: Camera directory to align
        anchor_timestamps: List of (timestamp, frame_number) from anchor camera
        output_dir: Output directory for aligned frames
        max_time_diff: Maximum allowed time difference for matching (in timestamp units)
    """
    print(f"Aligning camera: {cam_dir.name}")
    
    # Create output camera directory
    cam_output_dir = output_dir / cam_dir.name
    cam_output_dir.mkdir(exist_ok=True)
    
    # Create rgb subdirectory
    rgb_output_dir = cam_output_dir / "rgb"
    rgb_output_dir.mkdir(exist_ok=True)
    
    # Parse timestamps for this camera
    ts_paths = [ts for ts in os.listdir(os.path.join(cam_dir.parent.parent, cam_dir.name)) if ts.endswith(".txt")]
    # assert len(ts_paths) == 1, f"Expected 1 timestamps.txt file, got {len(ts_paths)}"
    timestamp_file = cam_dir.parent.parent / cam_dir.name / ts_paths[vid_idx]
        
    camera_timestamps = parse_timestamp_file(timestamp_file)
    camera_timestamp_dict = {ts: frame_num for ts, frame_num in camera_timestamps}
    
    # Find source rgb directory
    rgb_source_dir = cam_dir / "rgb"
    if not rgb_source_dir.exists():
        print(f"Warning: No rgb directory found in {cam_dir.name}")
        return
    
    # Get all rgb files
    rgb_files = sorted(glob.glob(str(rgb_source_dir / "*.jpg")))
    if not rgb_files:
        rgb_files = sorted(glob.glob(str(rgb_source_dir / "*.png")))
    
    print(f"Found {len(rgb_files)} RGB files in {cam_dir.name}")
    
    # Create aligned timestamps file
    aligned_timestamps_file = cam_output_dir / "aligned_timestamps.txt"
    
    # Process each anchor timestamp
    matched_count = 0
    with open(aligned_timestamps_file, 'w') as f:
        for frame_idx, (anchor_ts, anchor_frame) in enumerate(anchor_timestamps):
            # Find closest timestamp in this camera
            closest_ts, time_diff = find_closest_timestamp(anchor_ts, list(camera_timestamp_dict.keys()))
            
            if closest_ts and time_diff <= max_time_diff:
                # Found a match within acceptable time difference
                camera_frame_num = camera_timestamp_dict[closest_ts]
                
                # Find the corresponding RGB file
                source_file = None
                for rgb_file in rgb_files:
                    if f"{camera_frame_num:06d}" in rgb_file:
                        source_file = rgb_file
                        break
                
                if source_file:
                    # Copy and rename to output
                    output_filename = f"{frame_idx:06d}.jpg"
                    output_path = rgb_output_dir / output_filename
                    
                    # Copy the file
                    shutil.copy2(source_file, output_path)
                    
                    # Write to aligned timestamps file
                    f.write(f"frame_{anchor_ts}_{frame_idx:012d}\n")
                    
                    matched_count += 1
                    print(f"  Frame {frame_idx:06d}: {anchor_ts} -> {closest_ts} (diff: {time_diff})")
                else:
                    print(f"  Warning: Could not find RGB file for frame {camera_frame_num}")
                    # Write placeholder to maintain frame count
                    f.write(f"frame_{anchor_ts}_{frame_idx:012d}\n")
            else:
                print(f"  Warning: No suitable match for {anchor_ts} (min diff: {time_diff})")
                # Write placeholder to maintain frame count
                f.write(f"frame_{anchor_ts}_{frame_idx:012d}\n")
    
    print(f"Matched {matched_count}/{len(anchor_timestamps)} frames for {cam_dir.name}")

def align_and_rename_frames(data_dir: Path, output_dir: Path, camera_names: List[str] = None, 
                           max_time_diff: int = 1000000):
    """
    Align frames across multiple camera views using the anchor camera approach.
    
    Args:
        data_dir: Input directory containing camera folders
        output_dir: Output directory for aligned frames
        camera_names: List of camera names to process (if None, process all)
        max_time_diff: Maximum allowed time difference for matching
    """
    # Find camera directories
    print(f"length of camera_names: {len(camera_names)}")
    # get the parent dir of data_dir
    parent_dir = data_dir.parent
    episode = data_dir.name
    # get the nonzero digit in the episode
    episode_num = int(re.search(r'\d+', episode).group())
    if camera_names:
        camera_dirs = [data_dir / cam for cam in camera_names if (data_dir / cam).exists()]
    else:
        camera_dirs = [d for d in data_dir.iterdir() if d.is_dir() and (d / "rgb").exists()]
    
    txt_dirs =  [parent_dir / cam for cam in camera_names if (parent_dir / cam).exists()]
    
    print(f"Found {len(camera_dirs)} camera directories: {[d.name for d in camera_dirs]}")
    
    if not camera_dirs:
        print("No camera directories found!")
        return
    
    # Find anchor camera (fewest frames)
    anchor_camera, anchor_timestamps = find_anchor_camera(txt_dirs, episode_num)
    if not anchor_camera:
        print("No valid camera directories found!")
        return
    
    print(f"\nAnchor camera: {anchor_camera.name} with {len(anchor_timestamps)} frames")
    # Create output directory structure
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # First, copy anchor camera as-is
    print(f"\nCopying anchor camera: {anchor_camera.name}")
    anchor_output_dir = output_dir / anchor_camera.name
    anchor_output_dir.mkdir(exist_ok=True)
    
    # Copy RGB files
    rgb_source_dir = anchor_camera.parent / episode / anchor_camera.name / "rgb"
    print(f"rgb_source_dir: {rgb_source_dir}")
    rgb_output_dir = anchor_output_dir / "rgb"
    print(f"rgb_output_dir: {rgb_output_dir}")
    rgb_output_dir.mkdir(exist_ok=True)
    
    rgb_files = sorted(glob.glob(str(rgb_source_dir / "*.jpg")))
    if not rgb_files:
        rgb_files = sorted(glob.glob(str(rgb_source_dir / "*.png")))
    
    for frame_idx, (timestamp, frame_num) in enumerate(anchor_timestamps):
        # Find source file
        source_file = None
        for rgb_file in rgb_files:
            if f"{frame_num:06d}" in rgb_file:
                source_file = rgb_file
                break
        
        if source_file:
            output_filename = f"{frame_idx:06d}.jpg"
            output_path = rgb_output_dir / output_filename
            shutil.copy2(source_file, output_path)
    
    # Create aligned timestamps file for anchor camera
    aligned_timestamps_file = anchor_output_dir / f"aligned_timestamps.txt"
    with open(aligned_timestamps_file, 'w') as f:
        for frame_idx, (timestamp, frame_num) in enumerate(anchor_timestamps):
            f.write(f"frame_{timestamp}_{frame_idx:012d}\n")
    
    print(f"Copied {len(anchor_timestamps)} frames from anchor camera")
    
    # Align all other cameras to the anchor
    for cam_dir in camera_dirs:
        if cam_dir != anchor_camera:
            align_camera_to_anchor(cam_dir, anchor_timestamps, output_dir, max_time_diff, episode_num)
    
    print(f"\nAlignment complete! Output saved to: {output_dir}")
    print(f"All cameras now have {len(anchor_timestamps)} frames aligned to anchor camera")
    
    