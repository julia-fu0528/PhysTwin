
import os
import glob
import pickle
import json
import cv2
import torch
import numpy as np
import wandb
from typing import Dict, List, Optional
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation as R
from pytorch3d.loss import chamfer_distance
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.render_utils import (
    create_camera_view, render_gaussians_lbs, load_mask_h5, 
    apply_mask_to_frame, calculate_metrics, save_comparison_video,
    read_video_frame
)
from gs_render import remove_gaussians_with_low_opacity
from qqtt.utils.visualize import visualize_pc
from qqtt.utils.config import cfg as qqtt_cfg

class BaseMetric:
    """Base class for evaluation metrics."""
    def __init__(self, output_dir: str):
        self.output_dir = output_dir

    def evaluate(self, episode_path: str, pred_positions: torch.Tensor, current_epoch: int) -> Dict[str, float]:
        """
        Evaluate metric for a single episode.
        
        Args:
            episode_path: Path to the episode directory.
            pred_positions: Predicted positions (T, N, 3) tensor.
            current_epoch: Current training epoch.
            
        Returns:
            Dictionary of metric names and values.
        """
        raise NotImplementedError

class ChamferMetric(BaseMetric):
    """Computes Chamfer Distance between prediction and ground truth object points."""
    
    def evaluate(self, episode_path: str, pred_positions: torch.Tensor, current_epoch: int, **kwargs) -> Dict[str, float]:
        # Load GT data
        gt_path = os.path.join(episode_path, "final_data.pkl")
        split_path = os.path.join(episode_path, "split.json")
        metadata_path = os.path.join(episode_path, "metadata.json")

        with open(gt_path, "rb") as f:
            data = pickle.load(f)
        with open(split_path, "r") as f:
            split_data = json.load(f)
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        object_points = data["object_points"]  # (T, N_obj, 3)
        object_visibilities = data.get("object_visibilities", None)
        train_range = split_data.get("train", [0, 0])
        test_range = split_data.get("test", [0, 0])
        offset = metadata.get("start_frame", 0)
        
        # Transform GT to marker space
        T_marker2world = np.array([[ 9.92500579e-01, -1.22225711e-01,  1.86443478e-03,  1.36186366e-01],
                                  [ 5.43975403e-04, -1.08359291e-02, -9.99941142e-01, -1.88119571e-02],
                                  [ 1.22238720e-01,  9.92443176e-01, -1.06881781e-02,  7.19721945e-02],
                                  [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
        T_world2marker = np.linalg.inv(T_marker2world)
        
        # Transform pred_positions to marker space
        orig_pred_shape = pred_positions.shape
        pred_positions_homog = torch.cat([pred_positions.reshape(-1, 3), torch.ones((pred_positions.numel() // 3, 1), device=pred_positions.device)], dim=-1)
        T_w2m_torch = torch.from_numpy(T_world2marker).float().to(pred_positions.device)
        pred_positions = (T_w2m_torch @ pred_positions_homog.T).T[:, :3].reshape(orig_pred_shape)
        
        # Transform object_points
        orig_shape = object_points.shape
        object_points_homog = np.concatenate([object_points.reshape(-1, 3), np.ones((object_points.size // 3, 1))], axis=-1)
        object_points = (T_world2marker @ object_points_homog.T).T[:, :3].reshape(orig_shape)
        
        # Prepare tensors
        object_points_torch = torch.from_numpy(object_points).float().to(pred_positions.device)
        if object_visibilities is not None:
             object_visibilities_torch = torch.from_numpy(object_visibilities).bool().to(pred_positions.device)
        else:
             object_visibilities_torch = torch.ones_like(object_points_torch[..., 0], dtype=torch.bool)

        n_frames = len(pred_positions)
        
        # Calculate Frame Indices
        train_start_rel = max(0, train_range[0] - offset)
        train_end_rel = min(n_frames, train_range[1] - offset)
        test_start_rel = max(0, test_range[0] - offset)
        test_end_rel = min(n_frames, test_range[1] - offset)

        def compute_avg_chamfer(start, end):
            errors = []
            for t in range(start, end):
                if t == 0: continue # Skip first frame usually
                pred_t = pred_positions[t]
                gt_t = object_points_torch[t]
                vis_t = object_visibilities_torch[t]
                gt_points_vis = gt_t[vis_t]
                if len(gt_points_vis) > 0:
                    dist1, _ = chamfer_distance(gt_points_vis.unsqueeze(0), pred_t.unsqueeze(0), single_directional=True, norm=1)
                    errors.append(dist1.item())
            return np.mean(errors) if errors else 0.0, len(errors)

        train_err, train_num = compute_avg_chamfer(train_start_rel, train_end_rel)
        test_err, test_num = compute_avg_chamfer(test_start_rel, test_end_rel)

        return {
            "train/chamfer_error": train_err,
            "test/chamfer_error": test_err,
            "train/chamfer_frame_num": train_num,
            "test/chamfer_frame_num": test_num
        }


class TrackMetric(BaseMetric):
    """Computes Tracking Error (L2 distance) on tracked points."""
    
    def evaluate(self, episode_path: str, pred_positions: torch.Tensor, current_epoch: int, **kwargs) -> Dict[str, float]:
        gt_path = os.path.join(episode_path, "final_data.pkl")
        split_path = os.path.join(episode_path, "split.json")
        metadata_path = os.path.join(episode_path, "metadata.json")

        with open(gt_path, "rb") as f:
            data = pickle.load(f)
        with open(split_path, "r") as f:
            split_data = json.load(f)
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        gt_track_3d = data["object_points"] # (T, N, 3)
        train_range = split_data.get("train", [0, 0])
        test_range = split_data.get("test", [0, 0])
        offset = metadata.get("start_frame", 0)
        
        # Transform to marker space
        T_marker2world = np.array([[ 9.92500579e-01, -1.22225711e-01,  1.86443478e-03,  1.36186366e-01],
                                  [ 5.43975403e-04, -1.08359291e-02, -9.99941142e-01, -1.88119571e-02],
                                  [ 1.22238720e-01,  9.92443176e-01, -1.06881781e-02,  7.19721945e-02],
                                  [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
        T_world2marker = np.linalg.inv(T_marker2world)
        
        # Transform pred_positions to marker space
        orig_pred_shape = pred_positions.shape
        pred_positions_homog = torch.cat([pred_positions.reshape(-1, 3), torch.ones((pred_positions.numel() // 3, 1), device=pred_positions.device)], dim=-1)
        T_w2m_torch = torch.from_numpy(T_world2marker).float().to(pred_positions.device)
        pred_positions = (T_w2m_torch @ pred_positions_homog.T).T[:, :3].reshape(orig_pred_shape)
        
        orig_shape = gt_track_3d.shape
        gt_points_homog = np.concatenate([gt_track_3d.reshape(-1, 3), np.ones((gt_track_3d.size // 3, 1))], axis=-1)
        gt_track_3d = (T_world2marker @ gt_points_homog.T).T[:, :3].reshape(orig_shape)
        
        pred_0 = pred_positions[0].cpu().numpy()
        gt_0 = gt_track_3d[0]
        mask = ~np.isnan(gt_0).any(axis=1)
        if not mask.any():
            return {"train/track_error": 0.0, "test/track_error": 0.0}
            
        kdtree = KDTree(pred_0)
        dis, idx = kdtree.query(gt_0[mask])
        
        n_frames = len(pred_positions)
        train_start_rel = max(0, train_range[0] - offset)
        train_end_rel = min(n_frames, train_range[1] - offset)
        test_start_rel = max(0, test_range[0] - offset)
        test_end_rel = min(n_frames, test_range[1] - offset)

        def compute_avg_track(start, end):
            errors = []
            for t in range(start, end):
                if t == 0: continue
                gt_subset_t = gt_track_3d[t][mask]
                new_mask_t = ~np.isnan(gt_subset_t).any(axis=1)
                gt_track_points = gt_subset_t[new_mask_t]
                pred_points = pred_positions[t].cpu().numpy()[idx][new_mask_t]
                if len(pred_points) > 0:
                    err = np.mean(np.linalg.norm(pred_points - gt_track_points, axis=1))
                    errors.append(err)
            return np.mean(errors) if errors else 0.0

        return {
            "train/track_error": compute_avg_track(train_start_rel, train_end_rel),
            "test/track_error": compute_avg_track(test_start_rel, test_end_rel)
        }


class RenderMetric(BaseMetric):
    """Renders predicted trajectory using LBS and computes visual metrics."""
    
    def __init__(self, output_dir: str, skip_render: bool = False):
        super().__init__(output_dir)
        self.skip_render = skip_render
        
    def evaluate(self, episode_path: str, pred_positions: torch.Tensor, current_epoch: int, **kwargs) -> Dict[str, float]:
        # Using camera 0 by default
        cam_idx = 0
        
        # Load calibration and metadata
        calibrate_path = os.path.join(episode_path, "calibrate.pkl")
        metadata_path = os.path.join(episode_path, "metadata.json")
        split_path = os.path.join(episode_path, "split.json")
        
        if not os.path.exists(calibrate_path) or not os.path.exists(metadata_path):
            print("Missing calibration/metadata for rendering")
            return {}
            
        with open(calibrate_path, 'rb') as f:
            c2ws = pickle.load(f)
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        with open(split_path, 'r') as f:
            split_data = json.load(f)
            
        # Determine frame range (test split)
        train_range = split_data.get("train", [0, 0])
        test_range = split_data.get("test", [0, 0])
        
        # Predicted frames start from the frame specified in metadata["start_frame"]
        offset = metadata.get("start_frame", 0) 
        
        # We rendered the full rollout (pred_positions) which typically starts from frame 0 relative to logical start
        # pred_positions corresponds to frames [offset, offset + T] in the original video
        
        # Camera setup (World to Marker)
        T_marker2world = np.array([[ 9.92500579e-01, -1.22225711e-01,  1.86443478e-03,  1.36186366e-01],
                                  [ 5.43975403e-04, -1.08359291e-02, -9.99941142e-01, -1.88119571e-02],
                                  [ 1.22238720e-01,  9.92443176e-01, -1.06881781e-02,  7.19721945e-02],
                                  [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
        T_world2marker = np.linalg.inv(T_marker2world)
        
        # Transform pred_positions to marker space
        orig_pred_shape = pred_positions.shape
        pred_positions_homog = torch.cat([pred_positions.reshape(-1, 3), torch.ones((pred_positions.numel() // 3, 1), device=pred_positions.device)], dim=-1)
        T_w2m_torch = torch.from_numpy(T_world2marker).float().to(pred_positions.device)
        pred_positions = (T_w2m_torch @ pred_positions_homog.T).T[:, :3].reshape(orig_pred_shape)
        c2ws = [T_world2marker @ c2w for c2w in c2ws]
        
        intrinsics = np.array(metadata["intrinsics"])
        width, height = metadata["WH"]
        
        cam_folder = sorted(glob.glob(os.path.join(episode_path, "*cam*")))[cam_idx]
        c2w = c2ws[cam_idx]
        w2c = np.linalg.inv(c2w)
        intrinsic = intrinsics[cam_idx]
        view = create_camera_view(w2c, intrinsic, height, width, cam_idx)
        background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        
        # Load Gaussian Model
        gs_paths = os.path.join(episode_path, "splatfacto", "splat_0.ply")
        if not gs_paths:
            print("No GS model found")
            return {}
        
        gaussians = GaussianModel(sh_degree=3)
        gaussians.load_ply(gs_paths)
        gaussians = remove_gaussians_with_low_opacity(gaussians, 0.1)
        
        # Transform Gaussians to element space/marker space
        R_motion = torch.tensor(T_world2marker[:3, :3], dtype=torch.float32, device="cuda")
        t_motion = torch.tensor(T_world2marker[:3, 3], dtype=torch.float32, device="cuda")
        
        xyz = gaussians.get_xyz
        new_xyz = (xyz @ R_motion.T) + t_motion
        gaussians._xyz = new_xyz
        
        curr_rot = gaussians.get_rotation
        rot_world2marker = R.from_matrix(T_world2marker[:3, :3])
        quats_world_scipy = np.roll(curr_rot.detach().cpu().numpy(), -1, axis=1) # wxyz -> xyzw
        rots_world = R.from_quat(quats_world_scipy)
        rots_marker = rot_world2marker * rots_world
        quats_marker_scipy = rots_marker.as_quat()
        quats_marker = np.roll(quats_marker_scipy, 1, axis=1) # xyzw -> wxyz
        gaussians._rotation = torch.tensor(quats_marker, dtype=curr_rot.dtype, device=curr_rot.device)
        
        # Render
        n_frames = len(pred_positions)
        pred_frames = render_gaussians_lbs(gaussians, pred_positions, view, background, n_frames)
        
        # Load GT frames for comparison
        gt_video_path = os.path.join(cam_folder, "undistorted.mp4")
        mask_path = os.path.join(cam_folder, "mask_refined.h5")
        
        # Apply mask to predictions
        for i, frame in enumerate(pred_frames):
            actual_frame_idx = i + offset
            mask = load_mask_h5(mask_path, actual_frame_idx)
            pred_frames[i] = apply_mask_to_frame(frame, mask)
        
        # Load GT frames
        gt_frames = []
        cap = cv2.VideoCapture(gt_video_path)
        for i in range(n_frames):
            actual_frame_idx = i + offset
            frame = read_video_frame(cap, actual_frame_idx)
            if frame is not None:
                mask = load_mask_h5(mask_path, actual_frame_idx)
                frame = apply_mask_to_frame(frame, mask)
                gt_frames.append(frame)
        cap.release()
        
        # Calculate metrics for both Train and Test splits
        metrics = {}
        
        # Train split
        train_start_abs = train_range[0]
        train_end_abs = train_range[1]
        
        train_start_rel = max(0, train_start_abs - offset)
        train_end_rel = min(n_frames, train_end_abs - offset)
        
        if train_end_rel > train_start_rel:
            gt_train = gt_frames[train_start_rel:train_end_rel]
            pred_train = pred_frames[train_start_rel:train_end_rel]
            
            # Additional safety check for length mismatch (though slicing handles it)
            min_len = min(len(gt_train), len(pred_train))
            gt_train = gt_train[:min_len]
            pred_train = pred_train[:min_len]
            
            if len(gt_train) > 0:
                res_train = calculate_metrics(gt_train, pred_train)
                metrics["train/psnr"] = res_train["psnr"]
                metrics["train/ssim"] = res_train["ssim"]
                metrics["train/lpips"] = res_train["lpips"]

        # Test split
        test_start_abs = test_range[0]
        test_end_abs = test_range[1]
        
        test_start_rel = max(0, test_start_abs - offset)
        test_end_rel = min(n_frames, test_end_abs - offset)
        
        if test_end_rel > test_start_rel:
            gt_test = gt_frames[test_start_rel:test_end_rel]
            pred_test = pred_frames[test_start_rel:test_end_rel]
            
            min_len = min(len(gt_test), len(pred_test))
            gt_test = gt_test[:min_len]
            pred_test = pred_test[:min_len]
            
            if len(gt_test) > 0:
                res = calculate_metrics(gt_test, pred_test)
                metrics["test/psnr"] = res["psnr"]
                metrics["test/ssim"] = res["ssim"]
                metrics["test/lpips"] = res["lpips"]
        
        # Save comparison video for visualization (Entire rollout)
        if not self.skip_render:
             save_name = f"epoch_{current_epoch}_{os.path.basename(episode_path)}.mp4"
             save_path = os.path.join(self.output_dir, "vis", save_name)
             os.makedirs(os.path.dirname(save_path), exist_ok=True)
             
             # Using full frames for visualization
             save_comparison_video(gt_frames, pred_frames, save_path)
             metrics["test/comparison_video"] = save_path
             
             # Visualize particles using Pyrender for debug
             qqtt_cfg.WH = (width, height)
             qqtt_cfg.intrinsics = intrinsics
             qqtt_cfg.c2ws = c2ws # These are already transformed to marker space
             qqtt_cfg.no_gui = True
             
             pc_save_name = f"epoch_{current_epoch}_{os.path.basename(episode_path)}_pc.mp4"
             pc_save_path = os.path.join(self.output_dir, "vis", pc_save_name)
             
             # Attempt to use EGL for headless rendering
             os.environ["PYOPENGL_PLATFORM"] = "egl"
             
             try:
                 visualize_pc(
                     object_points=pred_positions,
                     save_video=True,
                     save_path=pc_save_path,
                     vis_cam_idx=cam_idx
                 )
             except Exception as e:
                 print(f"Failed to visualize particles with Pyrender: {e}")
                 # Fallback or just ignore

        
        return metrics


