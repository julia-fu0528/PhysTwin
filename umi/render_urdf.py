import glob
import json
import os
from pathlib import Path

# Set EGL platform if not already set, often needed for headless rendering on Linux
if 'PYOPENGL_PLATFORM' not in os.environ:
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

try:
    import moviepy.editor as mpy
except ImportError:
    import moviepy as mpy
import numpy as np
import pyrender
import trimesh
import yourdfpy

from real_world.utils.brics_utils import get_extr, read_params
from real_world.utils.data_utils import UndistortedFrameStore
from real_world.utils.h5_utils import H5Array, ensure_h5_path


def opening_to_umi_joints(opening_dist_m: float) -> dict:
    """Map real-world gripper opening distance (meters) to UMI URDF joints.

    This matches the mapping used in `postprocessor.py::get_control_pts()`:
    - real world open/close range: [MIN_OPEN, MAX_OPEN]
    - URDF prismatic joint range: [LIMIT_DOWN, LIMIT_UP] (close -> open)
    """
    # real world open - close
    MIN_OPEN, MAX_OPEN = 0.04, 0.112
    # urdf close - open
    LIMIT_UP, LIMIT_DOWN = 0.038, 0.005

    opening_dist_m = float(np.clip(opening_dist_m, MIN_OPEN, MAX_OPEN))
    normalized_opening = (opening_dist_m - MIN_OPEN) / (MAX_OPEN - MIN_OPEN)
    joint = LIMIT_UP - normalized_opening * (LIMIT_UP - LIMIT_DOWN)
    return {"joint_left": float(joint), "joint_right": float(-joint)}


class UrdfRenderer:
    """A reusable URDF renderer that loads the model once and allows multiple renderings."""

    def __init__(self, urdf_path: str, width: int, height: int):
        """
        Initialize the URDF renderer.

        Args:
            urdf_path: Path to the .urdf file.
            width: Image width for rendering.
            height: Image height for rendering.
        """
        self.urdf_path = urdf_path
        self.width = width
        self.height = height

        # Load URDF once
        self.urdf = yourdfpy.URDF.load(urdf_path, load_meshes=True)

        # Create pyrender scene once (will be updated with joint positions)
        self.scene = pyrender.Scene.from_trimesh_scene(self.urdf.scene)

        # Initialize camera and light nodes (will be updated with poses)
        self.camera = pyrender.IntrinsicsCamera(fx=1.0, fy=1.0, cx=0.0, cy=0.0)  # Placeholder values
        self.node_camera = pyrender.Node(camera=self.camera, matrix=np.eye(4))
        self.scene.add_node(self.node_camera)

        self.light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
        self.node_light = pyrender.Node(light=self.light, matrix=np.eye(4))
        self.scene.add_node(self.node_light)

        # Create renderer once
        self.renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)

    def render(
        self,
        joint_positions: dict,
        camera_pose: np.ndarray,
        K: np.ndarray
    ) -> np.ndarray:
        """
        Render the URDF to a binary mask from a specific camera viewpoint.

        Args:
            joint_positions: Dict of joint names to angles (e.g. {'joint_left': 0.04}).
            camera_pose: 4x4 Camera-to-World matrix (OpenCV convention: +Z forward, +Y down).
            K: 3x3 Intrinsic matrix.

        Returns:
            (H, W) binary mask numpy array (uint8, 0 or 255).
        """
        # Update joint configuration
        self.urdf.update_cfg(joint_positions)

        # Recreate the scene with updated joint positions
        # We need to recreate the scene since yourdfpy updates the trimesh scene
        self.scene = pyrender.Scene.from_trimesh_scene(self.urdf.scene)

        # Update camera intrinsics
        self.camera.fx = K[0, 0]
        self.camera.fy = K[1, 1]
        self.camera.cx = K[0, 2]
        self.camera.cy = K[1, 2]

        # Convert OpenCV Camera Pose (Right-Down-Forward) to OpenGL (Right-Up-Back)
        # Pyrender expects OpenGL camera coordinates (looking down -Z)
        # Transformation: Rotate 180 degrees around X-axis
        pose_gl = camera_pose.copy()
        pose_gl[0:3, 1] *= -1  # Flip Y axis vector
        pose_gl[0:3, 2] *= -1  # Flip Z axis vector

        # Update camera and light nodes
        self.node_camera = pyrender.Node(camera=self.camera, matrix=pose_gl)
        self.node_light = pyrender.Node(light=self.light, matrix=pose_gl)
        self.scene.add_node(self.node_camera)
        self.scene.add_node(self.node_light)

        # Render offscreen with alpha channel (silhouette/mask)
        color, depth = self.renderer.render(self.scene)

        # Extract alpha channel as binary mask
        mask = (depth > 0).astype(bool)

        return mask

    def close(self):
        """Clean up the renderer."""
        if hasattr(self, 'renderer'):
            self.renderer.delete()


def render_urdf(output_dir: Path, ep_idx: int, cam_indices: list, calib_path: str, cameras: list, urdf_path: str = None):
    """Render URDF as binary masks from camera viewpoints and save as HDF5/MP4.

    Args:
        output_dir: Base output directory
        ep_idx: Episode index
        cam_indices: List of camera indices to render
        calib_path: Path to calibration parameters file
        cameras: List of camera names
        urdf_path: Path to URDF file (optional, defaults to umi/umi.urdf)
    """
    import logging
    log = logging.getLogger(__name__)

    episode_id = ep_idx
    episode_data_dir = output_dir / f"episode_{episode_id}"

    # Load robot pose data from .npy file
    robot_npy_path = episode_data_dir / "robot" / "robot.npy"
    if not robot_npy_path.exists():
        log.warning(f"No robot.npy file found at {robot_npy_path}")
        return

    # Load robot data
    robot_data = np.load(robot_npy_path, allow_pickle=True).item()
    poses = robot_data['T_worlds']
    openings = robot_data['openings']
    bimanual = robot_data.get('bimanual', poses.ndim == 4)

    if len(poses) == 0:
        log.warning("No valid robot poses found.")
        return

    # Load camera calibration parameters
    metric_params_undistorted_arr = read_params(str(calib_path))
    metric_params_undistorted = {p["cam_name"]: p for p in metric_params_undistorted_arr}

    # Initialize URDF renderer path
    if urdf_path is None:
        urdf_path = str(Path(__file__).parent.parent.parent / "umi" / "umi.urdf")

    for cam_idx in cam_indices:
        cam_name = cameras[cam_idx]
        log.info(f"Rendering URDF mask for camera {cam_name}")

        # Get image dimensions for this camera
        cam_dir = episode_data_dir / cam_name
        frame_store = UndistortedFrameStore(cam_dir)
        height, width = frame_store.H, frame_store.W
        frame_store.close()

        # Initialize URDF renderer for this camera's dimensions
        renderer = UrdfRenderer(urdf_path, width, height)

        # Get camera intrinsics and extrinsics
        param = metric_params_undistorted[cam_name]
        K = np.eye(3, dtype=float)
        K[0, 0] = param["fx"]
        K[1, 1] = param["fy"]
        K[0, 2] = param["cx"]
        K[1, 2] = param["cy"]
        c2w = get_extr(param)

        # Prepare output files
        h5_path = ensure_h5_path(cam_dir / "rendered_urdf.h5")
        mp4_path = cam_dir / "rendered_urdf.mp4"

        # Initialize HDF5 array for frames
        h5_array = H5Array(
            str(h5_path),
            mode="w",
            shape=(len(poses), height, width),
            chunks=(1, height, width),
            dtype=bool,
            compression="gzip",
            compression_level=4,
        )

        # Collect frames for video
        video_frames = []

        for frame_idx in range(len(poses)):
            if bimanual:
                T_world_frame = poses[frame_idx]  # (2, 4, 4)
                opening_frame = openings[frame_idx]  # (2,)
            else:
                T_world_frame = [poses[frame_idx]]  # List of one (4, 4)
                opening_frame = [openings[frame_idx]]  # List of one float

            # Accumulate mask for all grippers
            final_mask = np.zeros((height, width), dtype=bool)
            
            for g_idx in range(len(T_world_frame)):
                T_world = T_world_frame[g_idx]
                opening = opening_frame[g_idx]
                
                # Calculate relative camera pose: inv(T_world) @ c2w
                rel_cam_pose = np.linalg.inv(T_world) @ c2w

                # Calculate joint positions from gripper opening
                if opening is not None:
                    joint_positions = opening_to_umi_joints(opening)
                else:
                    joint_positions = opening_to_umi_joints(0.04)

                # Render this gripper
                gripper_mask = renderer.render(joint_positions, rel_cam_pose, K)
                final_mask = np.logical_or(final_mask, gripper_mask)

            # Save to HDF5
            h5_array[frame_idx] = final_mask

            video_frames.append(np.stack([final_mask, final_mask, final_mask], axis=-1) * 255)

            if frame_idx % 50 == 0:
                log.info(f"Rendered mask frame {frame_idx}/{len(poses)} for camera {cam_name}")

        # Save video
        if video_frames:
            clip = mpy.ImageSequenceClip(video_frames, fps=30)
            clip.write_videofile(str(mp4_path), codec='libx264', logger=None)
            log.info(f"Saved URDF mask video to {mp4_path}")

        # Close HDF5
        h5_array.flush()
        h5_array.close()
        log.info(f"Saved URDF mask HDF5 to {h5_path}")

        # Close renderer
        renderer.close()


# Backward compatibility function
def render_urdf_mask(
    urdf_path: str,
    joint_positions: dict,
    camera_pose: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int
) -> np.ndarray:
    """
    Renders the URDF to a binary mask from a specific camera viewpoint.

    Args:
        urdf_path: Path to the .urdf file.
        joint_positions: Dict of joint names to angles (e.g. {'joint_left': 0.04}).
        camera_pose: 4x4 Camera-to-World matrix (OpenCV convention: +Z forward, +Y down).
        K: 3x3 Intrinsic matrix.
        width: Image width.
        height: Image height.

    Returns:
        (H, W) binary mask numpy array (uint8, 0 or 255).
    """
    renderer = UrdfRenderer(urdf_path, width, height)
    try:
        return renderer.render(joint_positions, camera_pose, K)
    finally:
        renderer.close()
