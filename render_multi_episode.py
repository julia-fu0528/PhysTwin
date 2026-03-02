#!/usr/bin/env python3
"""
Rendering script for ParticleFormer and PGND multi-episode checkpoints.

Renders Gaussian splatting blended with real background, then segments it.
Computes rendering metrics (PSNR/SSIM/LPIPS) on both blended and segmented videos.

Usage:
    # ParticleFormer
    pixi run python render_multi_episode.py \
        --method particleformer \
        --data_root /oscar/data/gdk/hli230/projects/vitac-particle/processed \
        --object 001-rope \
        --cam_name brics-odroid-022_cam1

    # PGND
    pixi run python render_multi_episode.py \
        --method pgnd \
        --data_root /oscar/data/gdk/hli230/projects/vitac-particle/processed \
        --object 001-rope \
        --cam_name brics-odroid-022_cam1
"""

import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["EGL_PLATFORM"] = "surfaceless"

import sys
import argparse
import glob
import json
import csv
import pickle
import numpy as np
import cv2
import torch
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.render_utils import (
    create_camera_view, render_gaussians_lbs, load_mask_h5,
    apply_mask_to_frame, calculate_metrics,
    read_video_frame
)
from gs_render import remove_gaussians_with_low_opacity
from camera_utils import find_camera_index

# Background image directory
BRICS_BG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "brics_background")

# World-to-marker transform (shared across codebase)
T_MARKER2WORLD = np.array([
    [ 9.92500579e-01, -1.22225711e-01,  1.86443478e-03,  1.36186366e-01],
    [ 5.43975403e-04, -1.08359291e-02, -9.99941142e-01, -1.88119571e-02],
    [ 1.22238720e-01,  9.92443176e-01, -1.06881781e-02,  7.19721945e-02],
    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]
])
T_WORLD2MARKER = np.linalg.inv(T_MARKER2WORLD)


# ============================================================================
#  Checkpoint Discovery
# ============================================================================

def discover_checkpoint(data_root, object_name, method):
    """Find the multi-episode checkpoint for the given method.
    
    Returns:
        (ckpt_path, train_episodes, test_episode)
    """
    obj_dir = os.path.join(data_root, object_name)
    
    if method == "particleformer":
        pattern = os.path.join(obj_dir, "train_ep_*.ckpt")
    elif method == "pgnd":
        pattern = os.path.join(obj_dir, "pgnd_ep_*.ckpt")
    else:
        raise ValueError(f"Unknown method: {method}")
    
    ckpts = sorted(glob.glob(pattern))
    if not ckpts:
        raise FileNotFoundError(f"No {method} checkpoint found matching {pattern}")
    
    # Prefer multi-episode checkpoints (those with multiple episode indices like train_ep_0_1_2_3.ckpt)
    prefix = "train_ep_" if method == "particleformer" else "pgnd_ep_"
    multi_ep_ckpts = [c for c in ckpts if os.path.basename(c).replace(".ckpt", "").replace(prefix, "").count("_") > 0]
    
    if multi_ep_ckpts:
        ckpt_path = multi_ep_ckpts[-1]  # Use the last multi-episode checkpoint
    else:
        ckpt_path = ckpts[-1]  # Fallback to last single-episode checkpoint
    
    # Parse training episodes from filename
    # e.g., train_ep_0_1_2.ckpt -> [0, 1, 2]
    basename = os.path.basename(ckpt_path).replace(".ckpt", "")
    if method == "particleformer":
        ep_str = basename.replace("train_ep_", "")
    else:
        ep_str = basename.replace("pgnd_ep_", "")
    
    train_episodes = [int(x) for x in ep_str.split("_")]
    
    # Find test episode: last episode in 0-4 range not in training set
    all_candidates = [0, 1, 2, 3, 4]
    existing_episodes = []
    for ep_id in all_candidates:
        ep_path = os.path.join(obj_dir, f"episode_{ep_id}")
        if os.path.isdir(ep_path):
            existing_episodes.append(ep_id)
    
    test_episode = existing_episodes[-1] if existing_episodes else train_episodes[-1]
    
    print(f"Checkpoint: {ckpt_path}")
    print(f"Train episodes: {train_episodes}")
    print(f"Test episode: {test_episode}")
    
    return ckpt_path, train_episodes, test_episode


# ============================================================================
#  ParticleFormer Rollout
# ============================================================================

def particleformer_rollout(ckpt_path, data_root, object_name, test_episode):
    """Load ParticleFormer checkpoint and do full rollout on test episode.
    
    Returns:
        pred_positions: (T, N_obj, 3) tensor in world coordinates
    """
    from particleformer.config import ParticleFormerConfig
    from particleformer.models import ParticleFormer as PFModel
    from particleformer.trainer import farthest_point_sample
    
    # Load checkpoint
    state_dict = torch.load(ckpt_path, map_location="cuda")
    
    # Create model with default config and load weights
    config = ParticleFormerConfig()
    model = PFModel.from_config(config)
    model.load_state_dict(state_dict)
    model = model.cuda().eval()
    
    # Load episode data
    ep_path = os.path.join(data_root, object_name, f"episode_{test_episode}")
    gt_path = os.path.join(ep_path, "final_data.pkl")
    
    with open(gt_path, "rb") as f:
        data = pickle.load(f)
    
    object_points = data["object_points"]   # (T, N, 3)
    controller_points = data["controller_points"]  # (T, M, 3)
    
    # Sub-sample (matches training)
    N_OBJ_TARGET = 32
    N_CTRL_TARGET = 2
    
    num_obj_total = object_points.shape[1]
    num_ctrl_total = controller_points.shape[1]
    
    curr_pos_obj = object_points[0]
    curr_pos_ctrl = controller_points[0]
    
    if num_obj_total > N_OBJ_TARGET:
        obj_xyz = torch.from_numpy(curr_pos_obj).float().unsqueeze(0).cuda()
        fps_indices = farthest_point_sample(obj_xyz, N_OBJ_TARGET)
        fps_indices = fps_indices[0].cpu().numpy()
        curr_pos_obj = curr_pos_obj[fps_indices]
        object_points = object_points[:, fps_indices]
    
    if num_ctrl_total > N_CTRL_TARGET:
        curr_pos_ctrl = curr_pos_ctrl[:N_CTRL_TARGET]
        controller_points = controller_points[:, :N_CTRL_TARGET]
    
    num_obj = len(curr_pos_obj)
    num_ctrl = len(curr_pos_ctrl)
    total_particles = num_obj + num_ctrl
    
    # Prepare tensors
    curr_pos = np.concatenate([curr_pos_obj, curr_pos_ctrl], axis=0)
    curr_pos = torch.from_numpy(curr_pos).float().unsqueeze(0).cuda()
    
    is_controller = torch.zeros(total_particles, dtype=torch.bool, device="cuda")
    is_controller[num_obj:] = True
    is_controller = is_controller.unsqueeze(0)
    
    object_ids = torch.zeros(1, dtype=torch.long, device="cuda")
    
    # Build actions from GT controller points
    num_frames = len(object_points)
    actions_full = np.zeros((num_frames, total_particles, 3), dtype=np.float32)
    actions_full[0:num_frames-1, num_obj:] = controller_points[1:] - controller_points[:-1]
    actions_full = torch.from_numpy(actions_full).float().cuda()
    
    # Rollout
    pred_positions_list = [curr_pos]
    current_positions = curr_pos
    
    print(f"Starting ParticleFormer rollout: {num_frames} frames")
    with torch.no_grad():
        for t in tqdm(range(num_frames - 1), desc="PF Rollout"):
            current_actions = actions_full[t].unsqueeze(0)
            
            next_positions, _ = model(
                positions=current_positions,
                actions=current_actions,
                object_ids=object_ids,
                is_controller=is_controller,
                attention_mask=None
            )
            
            # Force controller positions to GT
            gt_ctrl_next = torch.from_numpy(controller_points[t+1]).float().cuda()
            next_positions[0, num_obj:] = gt_ctrl_next
            
            pred_positions_list.append(next_positions)
            current_positions = next_positions
    
    # Stack: (T, N+M, 3)
    pred_positions = torch.cat(pred_positions_list, dim=0)
    
    # Return full predictions (object + controller particles)
    return pred_positions, ep_path


# ============================================================================
#  PGND Rollout
# ============================================================================

def pgnd_rollout(ckpt_path, data_root, object_name, test_episode):
    """Load PGND checkpoint and do full rollout on test episode.
    
    Returns:
        pred_positions: (T, N, 3) tensor in world coordinates
    """
    import warp as wp
    from omegaconf import OmegaConf
    from hydra import compose, initialize_config_dir
    from pgnd.sim import Friction, CacheDiffSimWithFrictionBatch, StaticsBatch, CollidersBatch
    from pgnd.models import PGNDModel
    from pgnd.data import RealTeleopBatchDataset
    from pgnd.utils import get_root
    
    wp.init()
    wp.ScopedTimer.enabled = False
    wp.set_module_options({'fast_math': False})
    
    pgnd_root = Path(__file__).parent / "pgnd"
    
    # Use hydra compose API to properly resolve nested defaults
    cfg_dir = str(pgnd_root / "cfg")
    with initialize_config_dir(config_dir=cfg_dir, version_base="1.2"):
        cfg = compose(config_name="default")
    
    # Override source dataset
    source_path = os.path.join(data_root, object_name)
    cfg.train.source_dataset_name = source_path
    cfg.train.name = f"{object_name}_render_eval"
    cfg.train.save = False
    cfg.debug = True  # Disable wandb logging
    
    torch_device = torch.device("cuda:0")
    wp_device = wp.get_device("cuda:0")
    
    # Load model
    material = PGNDModel(cfg)
    material.to(torch_device)
    
    ckpt_data = torch.load(ckpt_path, map_location=torch_device)
    material.load_state_dict(ckpt_data["material"])
    material.eval()
    
    friction = Friction(np.array([cfg.model.friction.value]))
    friction.to(torch_device)
    friction.eval()
    
    dt = eval(cfg.sim.dt) if isinstance(cfg.sim.dt, str) else cfg.sim.dt
    
    # Load dataset for the test episode
    ep_path = os.path.join(source_path, f"episode_{test_episode}")
    
    # Try both naming conventions
    episode_name = f"episode_{test_episode}"
    if not os.path.exists(os.path.join(source_path, episode_name)):
        episode_name = f"episode_{test_episode:04d}"
    
    log_root = pgnd_root / "log"
    dataset_name = f"{object_name}_render_eval/dataset"
    
    eval_dataset = RealTeleopBatchDataset(
        cfg,
        dataset_root=log_root / dataset_name / "state",
        source_data_root=Path(source_path),
        device=torch_device,
        num_steps=cfg.sim.num_steps,
        eval_episode_name=episode_name,
        save_dataset=False,
    )
    eval_dataloader = iter(torch.utils.data.DataLoader(
        eval_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True
    ))
    init_state, actions, gt_states, downsample_indices = next(eval_dataloader)
    
    x, v, x_his, v_his, clip_bound, enabled, episode_vec = init_state
    x = x.to(torch_device)
    v = v.to(torch_device)
    x_his = x_his.to(torch_device)
    v_his = v_his.to(torch_device)
    actions = actions.to(torch_device)
    
    gt_x, gt_v = gt_states
    gt_x = gt_x.to(torch_device)
    
    batch_size = gt_x.shape[0]
    num_steps_total = gt_x.shape[1]
    num_particles = gt_x.shape[2]
    
    sim = CacheDiffSimWithFrictionBatch(cfg, num_steps_total, batch_size, wp_device, requires_grad=False)
    
    statics = StaticsBatch()
    statics.init(shape=(batch_size, num_particles), device=wp_device)
    statics.update_clip_bound(clip_bound)
    statics.update_enabled(enabled)
    colliders = CollidersBatch()
    
    num_grippers = cfg.sim.num_grippers
    colliders.init(shape=(batch_size, num_grippers), device=wp_device)
    if num_grippers > 0:
        colliders.initialize_grippers(actions[:, 0])
    
    enabled = enabled.to(torch_device)
    enabled_mask = enabled.unsqueeze(-1).repeat(1, 1, 3)
    
    pred_positions_list = [x[0].clone()]
    
    print(f"Starting PGND rollout: {num_steps_total} steps")
    with torch.no_grad():
        for step in tqdm(range(num_steps_total), desc="PGND Rollout"):
            if num_grippers > 0:
                colliders.update_grippers(actions[:, step])
            if cfg.sim.gripper_forcing:
                x_in = x.clone()
            
            pred = material(x, v, x_his, v_his, enabled)
            x, v = sim(statics, colliders, step, x, v,
                       friction.mu[None].repeat(batch_size, 1), pred)
            
            if cfg.sim.gripper_forcing:
                gripper_xyz = actions[:, step, :, :3]
                gripper_v = actions[:, step, :, 3:6]
                x_from_gripper = x_in[:, None] - gripper_xyz[:, :, None]
                x_gripper_distance = torch.norm(x_from_gripper, dim=-1)
                x_gripper_distance_mask = x_gripper_distance < cfg.model.gripper_radius
                x_gripper_distance_mask = x_gripper_distance_mask.unsqueeze(-1).repeat(1, 1, 1, 3)
                gripper_v_expand = gripper_v[:, :, None].repeat(1, 1, num_particles, 1)
                
                gripper_closed = actions[:, step, :, -1] < 0.5
                x_gripper_distance_mask = torch.logical_and(
                    x_gripper_distance_mask,
                    gripper_closed[:, :, None, None].repeat(1, 1, num_particles, 3)
                )
                
                gripper_quat_vel = actions[:, step, :, 10:13]
                gripper_angular_vel = torch.linalg.norm(gripper_quat_vel, dim=-1, keepdims=True)
                gripper_quat_axis = gripper_quat_vel / (gripper_angular_vel + 1e-10)
                
                grid_from_gripper_axis = x_from_gripper - \
                    (gripper_quat_axis[:, :, None] * x_from_gripper).sum(dim=-1, keepdims=True) * gripper_quat_axis[:, :, None]
                gripper_v_expand = torch.cross(
                    gripper_quat_vel[:, :, None], grid_from_gripper_axis, dim=-1
                ) + gripper_v_expand
                
                for i in range(gripper_xyz.shape[1]):
                    mask_single = x_gripper_distance_mask[:, i]
                    x[mask_single] = x_in[mask_single] + dt * gripper_v_expand[:, i][mask_single]
                    v[mask_single] = gripper_v_expand[:, i][mask_single]
            
            if cfg.sim.n_history > 0:
                x_his = torch.cat([
                    x_his.reshape(batch_size, num_particles, -1, 3)[:, :, 1:],
                    x[:, :, None].detach()
                ], dim=2).reshape(batch_size, num_particles, -1)
                v_his = torch.cat([
                    v_his.reshape(batch_size, num_particles, -1, 3)[:, :, 1:],
                    v[:, :, None].detach()
                ], dim=2).reshape(batch_size, num_particles, -1)
            
            pred_positions_list.append(x[0].clone())
    
    # Stack: (T, N, 3) in normalized coordinates
    pred_positions_normalized = torch.stack(pred_positions_list, dim=0)
    
    # Transform from normalized to world coordinates
    final_data_path = os.path.join(ep_path, "final_data.pkl")
    with open(final_data_path, "rb") as f:
        original_data = pickle.load(f)
    
    gt_object_points = original_data["object_points"]
    scale = cfg.sim.preprocess_scale
    
    pred_frame0 = pred_positions_normalized[0].cpu().numpy()
    gt_frame0 = gt_object_points[0]
    
    pred_centroid = pred_frame0.mean(axis=0)
    gt_centroid = gt_frame0.mean(axis=0)
    global_translation = pred_centroid - gt_centroid * scale
    
    pred_positions_world = (pred_positions_normalized.cpu().numpy() - global_translation) / scale
    pred_positions_world = torch.from_numpy(pred_positions_world).float().cuda()
    
    return pred_positions_world, ep_path


# ============================================================================
#  Background Loading
# ============================================================================

def load_background_image(cam_name, height, width):
    """Load the static background image for the given camera from brics_background/.
    
    Args:
        cam_name: Camera name (e.g., 'brics-odroid-022_cam1')
        height: Target height
        width: Target width
    
    Returns:
        Background image as numpy array (H, W, 3) in RGB, uint8
    """
    bg_dir = os.path.join(BRICS_BG_DIR, cam_name, "undistorted")
    if not os.path.isdir(bg_dir):
        print(f"Warning: Background directory not found: {bg_dir}")
        return np.zeros((height, width, 3), dtype=np.uint8)
    
    bg_files = sorted(glob.glob(os.path.join(bg_dir, "*.jpg")) + 
                      glob.glob(os.path.join(bg_dir, "*.png")))
    if not bg_files:
        print(f"Warning: No background images found in {bg_dir}")
        return np.zeros((height, width, 3), dtype=np.uint8)
    
    bg_img = cv2.imread(bg_files[0])
    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
    
    if bg_img.shape[:2] != (height, width):
        bg_img = cv2.resize(bg_img, (width, height))
    
    print(f"Loaded background image: {bg_files[0]} ({bg_img.shape})")
    return bg_img


# ============================================================================
#  Rendering with Background Blending
# ============================================================================

def render_with_background(gaussians, pred_positions, view, bg_image, n_frames):
    """Render Gaussians blended with real background using dual-render alpha extraction.
    
    Renders twice per frame:
    1. Black background → gets Gaussian contribution only
    2. White background → used with black to compute alpha
    Then composites: final = gaussian_contribution + alpha_remaining * background
    
    Args:
        gaussians: GaussianModel
        pred_positions: (T, N, 3) predicted particle positions
        view: Camera view object
        bg_image: Background image (H, W, 3) numpy uint8
        n_frames: Number of frames to render
    
    Returns:
        blended_frames: List of frames with real background
        segmented_frames: List of frames with black background (from black-bg render)
    """
    bg_black = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    bg_white = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
    
    # Normalize background for compositing
    bg_float = bg_image.astype(np.float32) / 255.0  # (H, W, 3)
    
    blended_frames = []
    segmented_frames = []
    
    # Save initial Gaussian state so we can reset after first render pass
    saved_xyz = gaussians._xyz.clone()
    saved_rot = gaussians._rotation.clone()
    
    # Render with black background using LBS
    print("Rendering with black background (for segmented + alpha)...")
    black_frames = render_gaussians_lbs(gaussians, pred_positions, view, bg_black, n_frames)
    
    # Restore initial state for second render pass (LBS mutates _xyz and _rotation)
    gaussians._xyz = saved_xyz
    gaussians._rotation = saved_rot
    
    print("Rendering with white background (for alpha extraction)...")
    white_frames = render_gaussians_lbs(gaussians, pred_positions, view, bg_white, n_frames)
    
    print("Compositing with real background...")
    for i in tqdm(range(min(len(black_frames), len(white_frames))), desc="Compositing"):
        black_f = black_frames[i].astype(np.float32) / 255.0  # (H, W, 3)
        white_f = white_frames[i].astype(np.float32) / 255.0  # (H, W, 3)
        
        # Alpha = 1 - (white - black)
        # When Gaussians are fully opaque: white==black, so alpha=1
        # When nothing rendered: black=0, white=1, so alpha=0
        alpha = 1.0 - (white_f - black_f)
        alpha = np.clip(alpha, 0, 1)
        
        # Average across channels for a single alpha
        alpha_single = alpha.mean(axis=-1, keepdims=True)  # (H, W, 1)
        
        # Composite: final = black_render + (1 - alpha) * background
        blended = black_f + (1.0 - alpha_single) * bg_float
        blended = np.clip(blended, 0, 1)
        blended_uint8 = (blended * 255).astype(np.uint8)
        
        blended_frames.append(blended_uint8)
        segmented_frames.append(black_frames[i])  # Already uint8
    
    return blended_frames, segmented_frames


# ============================================================================
#  Video Saving
# ============================================================================

def save_vertical_video(gt_frames, pred_frames, output_path, fps=30):
    """Save vertically concatenated video: GT on top, prediction on bottom.
    
    Args:
        gt_frames: List of ground truth frames (RGB uint8).
        pred_frames: List of predicted frames (RGB uint8).
        output_path: Path to save the video.
        fps: Frames per second.
    """
    if len(gt_frames) == 0:
        return
    h, w, _ = gt_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h * 2))
    
    for gt, pred in zip(gt_frames, pred_frames):
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (w, h))
        gt_bgr = cv2.cvtColor(gt, cv2.COLOR_RGB2BGR)
        pred_bgr = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
        combined = np.vstack((gt_bgr, pred_bgr))
        out.write(combined)
    out.release()
    
    # Transcode with ffmpeg for compatibility
    import subprocess
    temp_output = output_path.replace(".mp4", "_temp.mp4")
    os.rename(output_path, temp_output)
    cmd = [
        "ffmpeg", "-y", "-i", temp_output,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
        output_path
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.remove(temp_output)
        print(f"Saved video to {output_path}")
    except Exception as e:
        print(f"Failed to transcode: {e}")
        os.rename(temp_output, output_path)


# ============================================================================
#  Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Render multi-episode checkpoints with background blending and metrics"
    )
    parser.add_argument("--method", type=str, required=True, choices=["particleformer", "pgnd"],
                        help="Method: particleformer or pgnd")
    parser.add_argument("--data_root", type=str, 
                        default="/oscar/data/gdk/hli230/projects/vitac-particle/processed",
                        help="Root directory containing object data folders")
    parser.add_argument("--object", type=str, required=True,
                        help="Object name (e.g., 001-rope)")
    parser.add_argument("--cam_name", type=str, default="brics-odroid-022_cam1",
                        help="Camera name for rendering")
    parser.add_argument("--output_dir", type=str, default="results/render_multi_ep",
                        help="Output directory for videos and metrics")
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="Explicit checkpoint path (auto-discovered if not provided)")
    parser.add_argument("--test_episode", type=int, default=None,
                        help="Explicit test episode index (auto-discovered if not provided)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Discover checkpoint
    if args.ckpt_path is not None:
        ckpt_path = args.ckpt_path
        train_episodes = []  # Unknown
        test_episode = args.test_episode if args.test_episode is not None else 4
    else:
        ckpt_path, train_episodes, test_episode = discover_checkpoint(
            args.data_root, args.object, args.method
        )
    
    if args.test_episode is not None:
        test_episode = args.test_episode
    
    ep_path = os.path.join(args.data_root, args.object, f"episode_{test_episode}")
    
    # 2. Full rollout
    print(f"\n{'='*60}")
    print(f"Running {args.method} rollout on episode {test_episode}")
    print(f"{'='*60}")
    
    if args.method == "particleformer":
        pred_positions, ep_path = particleformer_rollout(
            ckpt_path, args.data_root, args.object, test_episode
        )
    else:
        pred_positions, ep_path = pgnd_rollout(
            ckpt_path, args.data_root, args.object, test_episode
        )
    
    print(f"Predicted positions shape: {pred_positions.shape}")
    
    # 3. Setup rendering
    print(f"\n{'='*60}")
    print(f"Setting up Gaussian rendering")
    print(f"{'='*60}")
    
    # Load calibration data
    calibrate_path = os.path.join(ep_path, "calibrate.pkl")
    metadata_path = os.path.join(ep_path, "metadata.json")
    split_path = os.path.join(ep_path, "split.json")
    
    with open(calibrate_path, "rb") as f:
        c2ws = pickle.load(f)
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    with open(split_path, "r") as f:
        split_data = json.load(f)
    
    # Camera setup
    cam_idx, actual_cam_name = find_camera_index(ep_path, args.cam_name)
    print(f"Using camera: {actual_cam_name} (index {cam_idx})")
    
    # Transform cameras to marker space
    c2ws = [T_WORLD2MARKER @ c2w for c2w in c2ws]
    
    intrinsics = np.array(metadata["intrinsics"])
    width, height = metadata["WH"]
    offset = metadata.get("start_frame", 0)
    
    c2w = c2ws[cam_idx]
    w2c = np.linalg.inv(c2w)
    intrinsic = intrinsics[cam_idx]
    view = create_camera_view(w2c, intrinsic, height, width, cam_idx)
    
    # Load Gaussian model
    gs_path = os.path.join(ep_path, "splatfacto", "splat_0.ply")
    if not os.path.exists(gs_path):
        gs_paths = sorted(glob.glob(os.path.join(ep_path, "splatfacto", "*.ply")))
        if not gs_paths:
            print(f"No Gaussian model found for episode {test_episode}")
            return
        gs_path = gs_paths[0]
    
    gaussians = GaussianModel(sh_degree=3)
    gaussians.load_ply(gs_path)
    gaussians = remove_gaussians_with_low_opacity(gaussians, 0.1)
    
    # Transform Gaussians to marker space
    R_motion = torch.tensor(T_WORLD2MARKER[:3, :3], dtype=torch.float32, device="cuda")
    t_motion = torch.tensor(T_WORLD2MARKER[:3, 3], dtype=torch.float32, device="cuda")
    
    xyz = gaussians.get_xyz
    new_xyz = (xyz @ R_motion.T) + t_motion
    gaussians._xyz = new_xyz
    
    curr_rot = gaussians.get_rotation
    rot_world2marker = R.from_matrix(T_WORLD2MARKER[:3, :3])
    quats_world_scipy = np.roll(curr_rot.detach().cpu().numpy(), -1, axis=1)
    rots_world = R.from_quat(quats_world_scipy)
    rots_marker = rot_world2marker * rots_world
    quats_marker_scipy = rots_marker.as_quat()
    quats_marker = np.roll(quats_marker_scipy, 1, axis=1)
    gaussians._rotation = torch.tensor(quats_marker, dtype=curr_rot.dtype, device=curr_rot.device)
    
    # Transform predictions to marker space  
    orig_pred_shape = pred_positions.shape
    pred_homog = torch.cat([
        pred_positions.reshape(-1, 3),
        torch.ones((pred_positions.numel() // 3, 1), device=pred_positions.device)
    ], dim=-1)
    T_w2m_torch = torch.from_numpy(T_WORLD2MARKER).float().to(pred_positions.device)
    pred_positions_marker = (T_w2m_torch @ pred_homog.T).T[:, :3].reshape(orig_pred_shape)
    
    n_frames = len(pred_positions_marker)
    
    # 4. Load background image
    bg_image = load_background_image(actual_cam_name, height, width)
    
    # 5. Render with blending
    print(f"\n{'='*60}")
    print(f"Rendering {n_frames} frames with background blending")
    print(f"{'='*60}")
    
    blended_frames, segmented_frames = render_with_background(
        gaussians, pred_positions_marker, view, bg_image, n_frames
    )
    
    # 6. Load GT frames
    print(f"\n{'='*60}")
    print(f"Loading GT frames and computing metrics")
    print(f"{'='*60}")
    
    cam_folders = sorted(glob.glob(os.path.join(ep_path, "*cam*")))
    cam_folder = cam_folders[cam_idx]
    gt_video_path = os.path.join(cam_folder, "undistorted.mp4")
    mask_path = os.path.join(cam_folder, "mask_refined.h5")
    
    train_range = split_data.get("train", [0, 0])
    test_range = split_data.get("test", [0, 0])
    
    # Load GT frames (raw, unmasked for blended comparison)
    gt_frames_raw = []
    gt_frames_masked = []
    cap = cv2.VideoCapture(gt_video_path)
    for i in range(n_frames):
        actual_frame_idx = i + offset
        frame = read_video_frame(cap, actual_frame_idx)
        if frame is not None:
            gt_frames_raw.append(frame.copy())
            mask = load_mask_h5(mask_path, actual_frame_idx)
            gt_frames_masked.append(apply_mask_to_frame(frame, mask))
        else:
            break
    cap.release()
    
    # Apply mask to blended predictions (segment the object from the blended rendering)
    segmented_frames_masked = []
    for i, frame in enumerate(blended_frames):
        actual_frame_idx = i + offset
        mask = load_mask_h5(mask_path, actual_frame_idx)
        segmented_frames_masked.append(apply_mask_to_frame(frame, mask))
    
    # 7. Save comparison videos first (so they're available even if metrics fail)
    obj_name = args.object
    ep_suffix = f"ep{test_episode}"
    
    train_start_rel = max(0, train_range[0] - offset)
    train_end_rel = min(n_frames, train_range[1] - offset)
    test_start_rel = max(0, test_range[0] - offset)
    test_end_rel = min(n_frames, test_range[1] - offset)
    
    # Clamp
    train_end_rel = min(train_end_rel, len(gt_frames_raw), len(blended_frames))
    test_end_rel = min(test_end_rel, len(gt_frames_raw), len(blended_frames))
    
    blended_video_path = os.path.join(args.output_dir, f"{obj_name}_{ep_suffix}_{args.method}_blended.mp4")
    segmented_video_path = os.path.join(args.output_dir, f"{obj_name}_{ep_suffix}_{args.method}_segmented.mp4")
    
    print(f"\nSaving blended video (GT top, pred bottom): {blended_video_path}")
    save_vertical_video(gt_frames_raw, blended_frames, blended_video_path)
    
    print(f"Saving segmented video (GT top, pred bottom): {segmented_video_path}")
    save_vertical_video(gt_frames_masked, segmented_frames_masked, segmented_video_path)
    
    # 8. Compute metrics
    print(f"\nSplit: train=[{train_start_rel}, {train_end_rel}), test=[{test_start_rel}, {test_end_rel})")
    
    metrics = {}
    
    # --- Blended metrics (pred with BG vs raw GT) ---
    print("\nComputing BLENDED metrics...")
    if train_end_rel > train_start_rel:
        gt_train = gt_frames_raw[train_start_rel:train_end_rel]
        pred_train = blended_frames[train_start_rel:train_end_rel]
        min_len = min(len(gt_train), len(pred_train))
        if min_len > 0:
            res = calculate_metrics(gt_train[:min_len], pred_train[:min_len])
            metrics["blended_train/psnr"] = res["psnr"]
            metrics["blended_train/ssim"] = res["ssim"]
            metrics["blended_train/lpips"] = res["lpips"]
            print(f"  Blended Train: PSNR={res['psnr']:.2f}, SSIM={res['ssim']:.4f}, LPIPS={res['lpips']:.4f}")
    
    if test_end_rel > test_start_rel:
        gt_test = gt_frames_raw[test_start_rel:test_end_rel]
        pred_test = blended_frames[test_start_rel:test_end_rel]
        min_len = min(len(gt_test), len(pred_test))
        if min_len > 0:
            res = calculate_metrics(gt_test[:min_len], pred_test[:min_len])
            metrics["blended_test/psnr"] = res["psnr"]
            metrics["blended_test/ssim"] = res["ssim"]
            metrics["blended_test/lpips"] = res["lpips"]
            print(f"  Blended Test:  PSNR={res['psnr']:.2f}, SSIM={res['ssim']:.4f}, LPIPS={res['lpips']:.4f}")
    
    # --- Segmented metrics (masked pred vs masked GT) ---
    print("\nComputing SEGMENTED metrics...")
    if train_end_rel > train_start_rel:
        gt_train = gt_frames_masked[train_start_rel:train_end_rel]
        pred_train = segmented_frames_masked[train_start_rel:train_end_rel]
        min_len = min(len(gt_train), len(pred_train))
        if min_len > 0:
            res = calculate_metrics(gt_train[:min_len], pred_train[:min_len])
            metrics["segmented_train/psnr"] = res["psnr"]
            metrics["segmented_train/ssim"] = res["ssim"]
            metrics["segmented_train/lpips"] = res["lpips"]
            print(f"  Segmented Train: PSNR={res['psnr']:.2f}, SSIM={res['ssim']:.4f}, LPIPS={res['lpips']:.4f}")
    
    if test_end_rel > test_start_rel:
        gt_test = gt_frames_masked[test_start_rel:test_end_rel]
        pred_test = segmented_frames_masked[test_start_rel:test_end_rel]
        min_len = min(len(gt_test), len(pred_test))
        if min_len > 0:
            res = calculate_metrics(gt_test[:min_len], pred_test[:min_len])
            metrics["segmented_test/psnr"] = res["psnr"]
            metrics["segmented_test/ssim"] = res["ssim"]
            metrics["segmented_test/lpips"] = res["lpips"]
            print(f"  Segmented Test:  PSNR={res['psnr']:.2f}, SSIM={res['ssim']:.4f}, LPIPS={res['lpips']:.4f}")
    
    # 9. Save CSV metrics
    csv_path = os.path.join(args.output_dir, f"{obj_name}_{ep_suffix}_{args.method}_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Object", "Episode", "Method",
            "Blended Train PSNR", "Blended Train SSIM", "Blended Train LPIPS",
            "Blended Test PSNR", "Blended Test SSIM", "Blended Test LPIPS",
            "Segmented Train PSNR", "Segmented Train SSIM", "Segmented Train LPIPS",
            "Segmented Test PSNR", "Segmented Test SSIM", "Segmented Test LPIPS",
        ])
        writer.writerow([
            obj_name, test_episode, args.method,
            f"{metrics.get('blended_train/psnr', 0):.4f}",
            f"{metrics.get('blended_train/ssim', 0):.4f}",
            f"{metrics.get('blended_train/lpips', 0):.4f}",
            f"{metrics.get('blended_test/psnr', 0):.4f}",
            f"{metrics.get('blended_test/ssim', 0):.4f}",
            f"{metrics.get('blended_test/lpips', 0):.4f}",
            f"{metrics.get('segmented_train/psnr', 0):.4f}",
            f"{metrics.get('segmented_train/ssim', 0):.4f}",
            f"{metrics.get('segmented_train/lpips', 0):.4f}",
            f"{metrics.get('segmented_test/psnr', 0):.4f}",
            f"{metrics.get('segmented_test/ssim', 0):.4f}",
            f"{metrics.get('segmented_test/lpips', 0):.4f}",
        ])
    
    print(f"\n{'='*60}")
    print(f"Results saved to {csv_path}")
    print(f"Blended video: {blended_video_path}")
    print(f"Segmented video: {segmented_video_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
