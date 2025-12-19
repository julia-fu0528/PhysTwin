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
    print(f"visualize: {visualize}, save_video: {save_video}, return_frames: {return_frames}")
    # Check if we're in headless mode (no DISPLAY)
    is_headless = not os.environ.get('DISPLAY') or os.environ.get('DISPLAY') == ''
    
    # In headless mode, video saving won't work without OSMesa/OpenGL - disable it automatically
    if is_headless and save_video and not visualize:
        logger.warning("Headless mode detected: Video saving disabled (requires OpenGL/OSMesa which is not available).")
        logger.warning("Set DISPLAY or install OSMesa to enable video saving in headless mode.")
        save_video = False
        if save_path:
            logger.info(f"Video would have been saved to: {save_path}")
    
    # Early return if neither visualization nor video saving is requested
    if not visualize and not save_video and not return_frames:
        logger.info("visualize_pc called but visualize=False, save_video=False, and return_frames=False - skipping")
        return None
    
    print(f"calling visualize_pc")
    # Deprecated function, use visualize_pc instead
    FPS = cfg.FPS
    width, height = cfg.WH
    intrinsic = cfg.intrinsics[vis_cam_idx]
    w2c = cfg.w2cs[vis_cam_idx]

    # Convert the stuffs to numpy if it's tensor
    if isinstance(object_points, torch.Tensor):
        object_points = object_points.cpu().numpy()
    if isinstance(object_colors, torch.Tensor):
        object_colors = object_colors.cpu().numpy()
    if isinstance(object_visibilities, torch.Tensor):
        object_visibilities = object_visibilities.cpu().numpy()
    if isinstance(object_motions_valid, torch.Tensor):
        object_motions_valid = object_motions_valid.cpu().numpy()
    if isinstance(controller_points, torch.Tensor):
        controller_points = controller_points.cpu().numpy()

    if object_colors is None:
        object_colors = np.tile(
            [1, 0, 0], (object_points.shape[0], object_points.shape[1], 1)
        )
    else:
        if object_colors.shape[1] < object_points.shape[1]:
            # If the object_colors is not the same as object_points, fill the colors with black
            object_colors = np.concatenate(
                [
                    object_colors,
                    np.ones(
                        (
                            object_colors.shape[0],
                            object_points.shape[1] - object_colors.shape[1],
                            3,
                        )
                    )
                    * 0.3,
                ],
                axis=1,
            )

    # The pcs is a 4d pcd numpy array with shape (n_frames, n_points, 3)
    vis = o3d.visualization.Visualizer()
    window_created = False
    headless_mode = False
    
    try:
        vis.create_window(visible=visualize, width=width, height=height)
        print(f"Creating window with visible: {visualize}, width: {width}, height: {height}")
        window_created = True
    except Exception as e:
        if not visualize and save_video:
            # In headless mode, try to create window for offscreen rendering
            print(f"Failed to create visible window: {e}. Attempting headless rendering...")
            logger.warning(f"Failed to create visible window: {e}. Attempting headless rendering...")
            try:
                vis.create_window(visible=False, width=width, height=height)
                print(f"Successfully created headless window")
                window_created = True
                headless_mode = True
            except Exception as e2:
                logger.error(f"Failed to create headless window: {e2}. Cannot proceed with visualization.")
                print(f"Failed to create headless window: {e2}. Cannot proceed with visualization.")
                if save_video:
                    logger.warning("Video saving disabled in headless mode (no OpenGL/OSMesa available)")
                return None
        else:
            logger.error(f"Failed to create window: {e}")
            print(f"Failed to create window: {e}")
            return None
    
    # Note: We've already disabled save_video in headless mode above, so no need to test capture here
    # vis.create_window(visible=visualize and not return_frames, width=width, height=height)
    # === Add Ground Plane at z=0 ===
    plane_color = [0.5, 0.5, 0.5]  # light gray
    plane_x = np.arange(-0.8, 0.8, 0.01)
    plane_y = np.arange(-0.8, 0.8, 0.01)
    plane_points = np.array([[x, y, 0] for x in plane_x for y in plane_y])

    ground_plane = o3d.geometry.PointCloud()
    ground_plane.points = o3d.utility.Vector3dVector(plane_points)
    ground_plane.colors = o3d.utility.Vector3dVector(np.tile(plane_color, (plane_points.shape[0], 1)))

    # vis.add_geometry(ground_plane)
    # T_marker2world = np.linalg.inv(cfg.T_world2marker)
    # T_marker2world = np.array([[ 9.92457290e-01, -1.22580045e-01,  1.63125912e-03,  3.31059452e-01],
    #                           [ 2.70205336e-04, -1.11191912e-02, -9.99938143e-01,  1.90897759e-01],
    #                           [ 1.22590601e-01,  9.92396340e-01, -1.10022006e-02,  2.75183546e-01],
    #                           [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
    # origin = T_marker2world[:3, 3]
    # axis_len = 1
    # x_end = origin + T_marker2world[:3, 0] * axis_len
    # y_end = origin + T_marker2world[:3, 1] * axis_len
    # z_end = origin + T_marker2world[:3, 2] * axis_len
    # x_axis = o3d.geometry.LineSet(
    #     points=o3d.utility.Vector3dVector([origin, x_end]),
    #     lines=o3d.utility.Vector2iVector([[0, 1]]),
    # )
    # x_axis.paint_uniform_color([1, 0, 0])
    # vis.add_geometry(x_axis)
    # y_axis = o3d.geometry.LineSet(
    #     points=o3d.utility.Vector3dVector([origin, y_end]),
    #     lines=o3d.utility.Vector2iVector([[0, 1]]),
    # )
    # y_axis.paint_uniform_color([0, 1, 0])
    # vis.add_geometry(y_axis)
    # z_axis = o3d.geometry.LineSet(
    #     points=o3d.utility.Vector3dVector([origin, z_end]),
    #     lines=o3d.utility.Vector2iVector([[0, 1]]),
    # )
    # z_axis.paint_uniform_color([0, 0, 1])
    # vis.add_geometry(z_axis)

    # if save_video and visualize:
    #     raise ValueError("Cannot save video and visualize at the same time.")
    # Initialize video writer if save_video is True
    if save_video:
        if save_path is None:
            logger.error("save_video=True but save_path is None. Cannot save video.")
            return None
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        logger.info(f"Saving video to: {save_path}")
        # fourcc = cv2.VideoWriter_fourcc(*"avc1")  # Codec for .mp4 file format
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for .mp4 file format
        video_writer = cv2.VideoWriter(save_path, fourcc, FPS, (width, height))
        if not video_writer.isOpened():
            print(f"Failed to open video writer for {save_path}")
            logger.error(f"Failed to open video writer for {save_path}")
            return None
        # video_writer = cv2.VideoWriter(save_path, fourcc, FPS, (width, height))
    # Initialize frames list if return_frames is True
    if return_frames:
        frames = []

    logger.info(f"Starting visualization loop: {object_points.shape[0]} frames, save_video={save_video}, visualize={visualize}")
    if controller_points is not None:
        controller_meshes = []
        prev_center = []
    for i in range(object_points.shape[0]):
        object_pcd = o3d.geometry.PointCloud()
        if object_visibilities is None:
            object_pcd.points = o3d.utility.Vector3dVector(object_points[i])
            object_pcd.colors = o3d.utility.Vector3dVector(object_colors[i])
        else:
            object_pcd.points = o3d.utility.Vector3dVector(
                object_points[i, np.where(object_visibilities[i])[0], :]
            )
            print(f"visibilities:{object_visibilities[i]}")
            object_pcd.colors = o3d.utility.Vector3dVector(
                object_colors[i, np.where(object_visibilities[i])[0], :]
            )
        if i == 0:
            render_object_pcd = object_pcd
            vis.add_geometry(render_object_pcd)
            if controller_points is not None:
                # Use sphere mesh for each controller point
                for j in range(controller_points.shape[1]):
                    origin = controller_points[i, j]
                    origin_color = [1, 0, 0]
                    controller_mesh = o3d.geometry.TriangleMesh.create_sphere(
                        radius=0.01
                    ).translate(origin)
                    controller_mesh.compute_vertex_normals()
                    controller_mesh.paint_uniform_color(origin_color)
                    controller_meshes.append(controller_mesh)
                    vis.add_geometry(controller_meshes[-1])
                    prev_center.append(origin)
            # Adjust the viewpoint - only needed if we're actually rendering
            # Skip if we're just saving video in headless mode and view_control isn't available
            view_control = vis.get_view_control()
            if view_control is not None:
                try:
                    camera_params = o3d.camera.PinholeCameraParameters()
                    intrinsic_parameter = o3d.camera.PinholeCameraIntrinsic(
                        width, height, intrinsic
                    )
                    camera_params.intrinsic = intrinsic_parameter
                    camera_params.extrinsic = w2c
                    view_control.convert_from_pinhole_camera_parameters(
                        camera_params, allow_arbitrary=True
                    )
                    print(f"Successfully set camera parameters")
                except Exception as e:
                    logger.warning(f"Failed to set camera parameters: {e}. Continuing without camera setup.")
                    print(f"Failed to set camera parameters: {e}. Continuing without camera setup.")
            else:
                # Headless mode - view control not available, skip camera setup
                # This is OK if we're just saving video - Open3D will use default view
                print(f"view_control is None")
                if save_video:
                    logger.info("View control not available (headless mode), using default camera view for video")
                    print(f"View control not available (headless mode), using default camera view for video")
                else:
                    logger.info("View control not available (headless mode), skipping camera setup")
        else:
            render_object_pcd.points = o3d.utility.Vector3dVector(object_pcd.points)
            render_object_pcd.colors = o3d.utility.Vector3dVector(object_pcd.colors)
            vis.update_geometry(render_object_pcd)
            if controller_points is not None:
                for j in range(controller_points.shape[1]):
                    origin = controller_points[i, j]
                    controller_meshes[j].translate(origin - prev_center[j])
                    vis.update_geometry(controller_meshes[j])
                    prev_center[j] = origin
        try:
            vis.poll_events()
            vis.update_renderer()
        except Exception as e:
            if not visualize:
                # In headless mode, these might fail - log and continue
                logger.warning(f"Failed to poll events/update renderer (headless mode?): {e}")
            else:
                raise

        cameras = [subdir for subdir in os.listdir(cfg.overlay_path) if "cam" in subdir and subdir not in cfg.remove_cams]

        print(f"cameras: {cameras}")
        print(f"len(cameras): {len(cameras)}")

        # Capture frame and write to video file if save_video is True, or collect if return_frames
        if save_video or return_frames:
            try:
                if headless_mode and i == 0:
                    logger.warning("Attempting frame capture in headless mode - this may hang if OpenGL is not available")
                frame = np.asarray(vis.capture_screen_float_buffer(do_render=True))
                if i == 0 and save_video:
                    logger.info(f"Successfully captured first frame (shape: {frame.shape})")
            except Exception as e:
                logger.error(f"Failed to capture screen buffer at frame {i}: {e}")
                if save_video:
                    logger.error("Video saving failed - likely due to headless mode without OpenGL/OSMesa support")
                    video_writer.release()
                return None if not return_frames else []
            frame = (frame * 255).astype(np.uint8)
            if cfg.overlay_path is not None:
                frame_num = cfg.start_frame + i
                mask = np.all(frame == [255, 255, 255], axis=-1)
                image_path = f"{cfg.overlay_path}/{cameras[vis_cam_idx]}/undistorted_refined/{frame_num:06d}.png"
                overlay = cv2.imread(image_path)
                if overlay is not None:
                    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                    frame[mask] = overlay[mask]  # Replace background
            if save_video:
                # Convert RGB to BGR for video writer
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                success = video_writer.write(frame_bgr)
                if i == 0:
                    logger.info(f"Writing first frame to video: {success}")
                if i % 10 == 0:
                    logger.debug(f"Written {i+1}/{object_points.shape[0]} frames to video")
            if return_frames:
                # Keep RGB format for frames list
                frames.append(frame)

        if visualize:
            time.sleep(1 / FPS)

    try:
        vis.destroy_window()
    except Exception as e:
        logger.warning(f"Failed to destroy window: {e}")
    if save_video:
        print(f"Releasing video writer")
        video_writer.release()
        if os.path.exists(save_path):
            file_size = os.path.getsize(save_path)
            logger.info(f"Video saved successfully: {save_path} ({file_size / 1024 / 1024:.2f} MB)")
        else:
            logger.error(f"Video file was not created: {save_path}")
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
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(save_path, fourcc, FPS, (output_width, output_height))
    
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
        logger.info(f"Saved grid video to {save_path}")