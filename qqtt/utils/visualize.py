import open3d as o3d
import numpy as np
import torch
import time
import cv2
from .config import cfg
from . import logger
import pyrender
import trimesh
import os   
import sys
import decord
import subprocess

def visualize_pc(
    object_points,
    object_colors=None,
    controller_points=None,
    object_visibilities=None,
    object_motions_valid=None,
    visualize=True,
    save_video=False,
    save_path=None,
    vis_cam_idx=23,
    return_frames=False,
):
    if cfg.no_gui and not save_video and not return_frames:
        logger.info("visualize_pc called but cfg.no_gui=True and no output requested - skipping")
        return None

    FPS = cfg.FPS
    width, height = cfg.WH
    
    # Ensure camera index is valid
    if vis_cam_idx >= len(cfg.intrinsics):
        logger.warning(f"vis_cam_idx {vis_cam_idx} out of bounds. Using 0.")
        vis_cam_idx = 0

    # Camera Intrinsics
    intrinsic_matrix = cfg.intrinsics[vis_cam_idx]
    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
    
    # Pyrender Camera
    camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.01, zfar=100.0)
    
    # Camera Pose (Camera to World)
    # Pyrender camera looks down -Z. 
    # Provided c2w should be compatible if it was working for Open3D/GL.
    c2w = cfg.c2ws[vis_cam_idx]
    # We might need to flip axes depending on conventions, but let's try direct use first akin to Open3D
    # Open3D: Y down, Z forward? Or standard GL? 
    # Usually real data c2w is GL style (X right, Y up, -Z look) or CV style (X right, Y down, Z look).
    # Pyrender expects GL style camera. 
    # If c2ws come from OpenCV calibration, they are likely CV style.
    # To convert CV (Y down, Z forward) to GL (Y up, -Z forward), rotate 180 deg around X.
    # Let's assume CV style input and convert to GL for Pyrender.
    
    # Data conversion
    if isinstance(object_points, torch.Tensor):
        object_points = object_points.cpu().numpy()
    if isinstance(object_colors, torch.Tensor):
        object_colors = object_colors.cpu().numpy()
    if isinstance(object_visibilities, torch.Tensor):
        object_visibilities = object_visibilities.cpu().numpy()
    if isinstance(controller_points, torch.Tensor):
        controller_points = controller_points.cpu().numpy()

    # Default colors
    if object_colors is None:
        object_colors = np.tile([1, 0, 0], (object_points.shape[0], object_points.shape[1], 1))
    
    # Pyrender Renderer
    r = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    
    # Video Writer
    video_writer = None
    if save_video:
        if save_path is None:
            logger.error("save_video=True but save_path is None")
            return None
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        temp_save_path = save_path.replace(".mp4", "_temp.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(temp_save_path, fourcc, FPS, (width, height))
        logger.info(f"Saving video to {temp_save_path}")

    frames = [] if return_frames else None
    
    # Pre-load overlay / undistorted video reader if needed
    overlay_reader = None
    target_camera_name = None
    if cfg.overlay_path is not None:
         all_cameras = sorted([subdir for subdir in os.listdir(cfg.overlay_path) if "cam" in subdir])
         if vis_cam_idx < len(all_cameras):
             target_camera_name = all_cameras[vis_cam_idx]
             video_path = f"{cfg.overlay_path}/{target_camera_name}/undistorted.mp4"
             if os.path.exists(video_path):
                 overlay_reader = decord.VideoReader(video_path, ctx=decord.cpu(0))

    logger.info(f"Starting Pyrender loop: {object_points.shape[0]} frames")
    from tqdm import tqdm
    
    # Create light
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    
    # Create conversion matrix from CV to GL if needed (Rotate 180 around X)
    # CV: x-right, y-down, z-forward
    # GL: x-right, y-up, z-back
    cv_to_gl = np.array([[1, 0, 0, 0], 
                         [0, -1, 0, 0], 
                         [0, 0, -1, 0], 
                         [0, 0, 0, 1]])
    
    camera_pose = c2w @ cv_to_gl
    
    # Pre-compute controller sphere mesh
    controller_mesh = None
    if controller_points is not None:
        sm = trimesh.creation.icosphere(radius=0.01)
        # Create pyrender mesh once
        controller_mesh = pyrender.Mesh.from_trimesh(sm, poses=np.eye(4))
        # Simple red material
        red_material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[1.0, 0.0, 0.0, 1.0],
            metallicFactor=0.0,
            roughnessFactor=0.5
        )
        for prim in controller_mesh.primitives:
             prim.material = red_material
    
    # Test render once to ensure it works before loop
    try:
        print("Testing single frame render...")
        test_scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0])
        test_scene.add(camera, pose=camera_pose)
        test_scene.add(light, pose=camera_pose)
        r.render(test_scene)
        print("Test render successful.")
    except Exception as e:
        logger.error(f"Test render failed: {e}")
        return None

    for i in tqdm(range(object_points.shape[0]), desc="Rendering frames"):
        scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0]) # White background for easy masking
        
        # Add Camera
        scene.add(camera, pose=camera_pose)
        
        # Add Light (attached to camera or fixed?) -> Fixed in simple scene or attached to camera
        scene.add(light, pose=camera_pose)
        
        # Add Object Points
        curr_points = object_points[i]
        curr_colors = object_colors[i]
        
        if object_visibilities is not None:
             vis_mask = np.where(object_visibilities[i])[0]
             curr_points = curr_points[vis_mask]
             curr_colors = curr_colors[vis_mask]
             
        # Pyrender doesn't support PointCloud directly as a specialized primitive with size, 
        # but Mesh with POINTS mode works. 
        m = pyrender.Mesh.from_points(curr_points, colors=curr_colors)
        scene.add(m)
        
        # Add Controller Points (Spheres)
        if controller_points is not None and controller_mesh is not None:
            for j in range(controller_points.shape[1]):
                cp_pos = controller_points[i, j]
                # Set pose 
                pose = np.eye(4)
                pose[:3, 3] = cp_pos
                scene.add(controller_mesh, pose=pose)

        # Render
        color, depth = r.render(scene)
        
        # Color is RGB
        frame = color.copy()
        
        # Overlay Logic
        if target_camera_name is not None:
            frame_num = cfg.start_frame + i
            # Create mask where background is white
            # Tolerance might be needed due to lighting/shading? 
            # If bg_color is exactly [1,1,1] and lighting doesn't affect background (it shouldn't in pyrender unless there's geometry)
            mask = np.all(frame >= [250, 250, 250], axis=-1)
            
            overlay_img = None
            if overlay_reader is not None and frame_num < len(overlay_reader):
                overlay_img = overlay_reader[frame_num].asnumpy()
            else:
                 # Check individual files
                 image_path = f"{cfg.overlay_path}/{target_camera_name}/undistorted_refined/{frame_num:06d}.png"
                 if os.path.exists(image_path):
                     overlay_img_bgr = cv2.imread(image_path)
                     if overlay_img_bgr is not None:
                         overlay_img = cv2.cvtColor(overlay_img_bgr, cv2.COLOR_BGR2RGB)
            
            if overlay_img is not None:
                 if overlay_img.shape[:2] != (height, width):
                      overlay_img = cv2.resize(overlay_img, (width, height))
                 frame[mask] = overlay_img[mask]
        
        # Save/Store
        if video_writer:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)
            
        if return_frames:
            frames.append(frame)

    # Cleanup
    r.delete()
    if video_writer:
        video_writer.release()
        # FFmpeg conversion (same as before)
        if save_path:
             try:
                 subprocess.run([
                     "ffmpeg", "-y", "-i", temp_save_path, 
                     "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", 
                     save_path
                 ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                 os.remove(temp_save_path)
                 logger.info(f"Video saved to {save_path}")
             except Exception as e:
                 logger.error(f"FFmpeg failed: {e}")
                 os.rename(temp_save_path, save_path)
    
    if return_frames:
        return frames


def visualize_pc_grid(
    object_points,
    object_colors=None,
    controller_points=None,
    object_visibilities=None,
    object_motions_valid=None,
    save_video=False,
    save_path=None,
    vis_cam_indices=None,
    grid_cols=None,
):
    """
    Visualize point cloud from multiple camera views in a grid layout.
    """
    if cfg.no_gui:
        logger.info("visualize_pc_grid called but cfg.no_gui=True - skipping")
        return None
    FPS = cfg.FPS
    width, height = cfg.WH
    
    # Get camera indices
    if vis_cam_indices is None:
        num_cameras = len(cfg.intrinsics)
        vis_cam_indices = list(range(num_cameras))
    
    num_cameras = len(vis_cam_indices)
    
    # Calculate grid layout
    if grid_cols is None:
        grid_cols = int(np.ceil(np.sqrt(num_cameras)))
    grid_rows = int(np.ceil(num_cameras / grid_cols))
    
    # Calculate individual view size
    view_width = width // grid_cols
    view_height = height // grid_rows
    
    # Output video size
    output_width = view_width * grid_cols
    output_height = view_height * grid_rows
    
    logger.info(f"Visualizing point cloud from {num_cameras} cameras in {grid_rows}x{grid_cols} grid layout.")
    
    # Collect frames from each camera
    all_camera_frames = []
    for vis_cam_idx in vis_cam_indices:
        frames = visualize_pc(
            object_points,
            object_colors,
            controller_points,
            object_visibilities,
            object_motions_valid,
            visualize=True,
            save_video=True,
            save_path=save_path.replace(".mp4", f"_camera_{vis_cam_idx}.mp4"),
            return_frames=True,
            vis_cam_idx=vis_cam_idx,
        )
        all_camera_frames.append(frames)
    
    # Combine frames into grid for each timestep
    num_frames = len(all_camera_frames[0])
    
    # Initialize video writer
    if save_video:
        temp_save_path = save_path.replace(".mp4", "_temp.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(temp_save_path, fourcc, FPS, (output_width, output_height))
    
    for frame_idx in range(num_frames):
        # Create grid frame
        grid_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        
        for cam_idx_idx, vis_cam_idx in enumerate(vis_cam_indices):
            row = cam_idx_idx // grid_cols
            col = cam_idx_idx % grid_cols
            
            y_start = row * view_height
            y_end = y_start + view_height
            x_start = col * view_width
            x_end = x_start + view_width
            
            # Get frame from this camera
            frame = all_camera_frames[cam_idx_idx][frame_idx]
            
            # Resize frame to fit grid cell if needed
            if frame.shape[0] != view_height or frame.shape[1] != view_width:
                frame = cv2.resize(frame, (view_width, view_height))
            
            grid_frame[y_start:y_end, x_start:x_end] = frame
        
        # Write to video
        if save_video:
            grid_frame_bgr = cv2.cvtColor(grid_frame, cv2.COLOR_RGB2BGR)
            video_writer.write(grid_frame_bgr)
    
    if save_video:
        video_writer.release()
        temp_save_path = save_path.replace(".mp4", "_temp.mp4")
        if os.path.exists(temp_save_path):
            logger.info(f"Converting grid video codec: {temp_save_path} -> {save_path}")
            try:
                subprocess.run([
                    "ffmpeg", "-y",
                    "-i", str(temp_save_path),
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-crf", "23",
                    str(save_path)
                ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                os.remove(temp_save_path)
                logger.info(f"Grid video converted and saved successfully: {save_path}")
            except subprocess.CalledProcessError as e:
                logger.error(f"FFmpeg conversion failed for grid video: {e}")
                # Fallback: rename temp to final if conversion failed
                os.rename(temp_save_path, save_path)
                logger.warning(f"Fallback: Saved unconverted grid video to {save_path}")
        else:
            logger.error(f"Grid video file was not created: {temp_save_path}")
