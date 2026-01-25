"""
Evaluate Rendering Metrics (PSNR, SSIM, LPIPS) with proper:
1. Frame alignment using split.json
2. GT masking using mask_refined.h5
3. Prediction rendering using 3DGS + LBS deformation
4. Prediction masking using mask_refined.h5
"""

import os
# Set environment variables for headless rendering
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["EGL_PLATFORM"] = "surfaceless"

import sys
import argparse
import glob
import json
import cv2
import csv
import torch
import wandb
import numpy as np
import subprocess
import pickle
import h5py
import copy
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

# Add PhysTwin root to path for imports
# This script should be run from PhysTwin root or with PYTHONPATH set
_this_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.dirname(_this_dir)
sys.path.insert(0, _root_dir)
sys.path.insert(0, _this_dir)

# Import from gaussian_splatting submodules (using absolute paths from root)
from gaussian_splatting.utils.loss_utils import ssim
from gaussian_splatting.lpipsPyTorch import lpips
from gaussian_splatting.utils.image_utils import psnr
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.scene.cameras import Camera
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.dynamic_utils import (
    get_topk_indices, knn_weights_sparse, interpolate_motions_speedup, 
    calc_weights_vals_from_indices, compute_bone_transforms, apply_bone_transforms_speedup
)
from gaussian_splatting.utils.graphics_utils import focal2fov, getWorld2View2, getProjectionMatrix

# Import from root-level modules
from gs_render import remove_gaussians_with_low_opacity


def img2tensor(img):
    """Convert numpy image (H, W, C) to tensor (1, C, H, W) on GPU."""
    img = np.array(img, dtype=np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    return torch.from_numpy(img).unsqueeze(0).cuda()


def read_video_frame(cap, frame_idx):
    """Read a specific frame from a video capture object."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
    return None


def load_mask_h5(mask_path, frame_idx):
    """Load a single frame mask from H5 file."""
    try:
        with h5py.File(mask_path, 'r') as f:
            mask = f['data'][frame_idx]
            # Normalize to 0-255 if needed
            if mask.max() <= 1.0:
                mask = (mask * 255).astype(np.uint8)
            else:
                mask = mask.astype(np.uint8)
            return mask
    except Exception as e:
        print(f"Error loading mask: {e}")
        return None


def apply_mask_to_frame(frame, mask):
    """Apply binary mask to frame (set background to black)."""
    if mask is None:
        return frame
    masked_frame = frame.copy()
    # Expand mask to 3 channels if needed
    if len(mask.shape) == 2:
        mask_3ch = np.stack([mask, mask, mask], axis=-1)
    else:
        mask_3ch = mask
    # Set background (mask == 0) to black
    masked_frame[mask_3ch == 0] = 0
    return masked_frame


def create_camera_view(w2c, intrinsic, height, width, cam_id=0):
    """Create a camera view object for Gaussian rendering."""
    R = np.transpose(w2c[:3, :3])  # R is stored transposed for CUDA
    T = w2c[:3, 3]
    
    FovY = focal2fov(intrinsic[1, 1], height)
    FovX = focal2fov(intrinsic[0, 0], width)
    
    view = Camera(
        colmap_id=cam_id,
        R=R,
        T=T,
        FoVx=FovX,
        FoVy=FovY,
        image_path="",
        image_name=f"eval_{cam_id}",
        uid=cam_id,
        K=intrinsic,
        width=width,
        height=height,
    )
    return view


def render_gaussians_lbs(gaussians, ctrl_pts, view, background, n_frames):
    """
    Render Gaussians with LBS deformation based on control point trajectories.
    Uses chunked processing to avoid OOM and precomputes weights for efficiency.
    
    Args:
        gaussians: GaussianModel with initial state
        ctrl_pts: (n_frames, n_ctrl_pts, 3) tensor of control point positions
        view: Camera view for rendering
        background: Background color tensor
        n_frames: Number of frames to render
    
    Returns:
        List of rendered frames (H, W, 3) as numpy arrays
    """
    rendered_frames = []
    
    # Ensure we don't exceed trajectory length
    n_frames = min(n_frames, len(ctrl_pts))
    
    # Get initial Gaussian state
    xyz_0 = gaussians.get_xyz.clone()
    quat_0 = gaussians.get_rotation.clone()
    
    # Current state (will be updated each frame)
    current_pos = xyz_0.clone()
    current_rot = quat_0.clone()
    
    # Initialize LBS relations (computed once from first control points)
    # Optimized get_topk_indices avoids OOM on large particle sets
    init_particle_pos = ctrl_pts[0]
    relations = get_topk_indices(init_particle_pos, K=16)
    
    # Chunk size for processing
    chunk_size = 20_000
    n_gaussians = len(current_pos)
    num_chunks = (n_gaussians + chunk_size - 1) // chunk_size
    
    # PRECOMPUTE weights and indices for all Gaussian chunks
    # This saves massive amounts of memory and time during the frame loop
    all_weights = []
    all_weights_indices = []
    
    print("Precomputing LBS weights for Gaussians...")
    for j in tqdm(range(num_chunks), desc="Weight precomputation"):
        start = j * chunk_size
        end = min((j + 1) * chunk_size, n_gaussians)
        pos_chunk = xyz_0[start:end]
        
        # Compute weights relative to initial pose
        weights, weights_indices = knn_weights_sparse(init_particle_pos, pos_chunk, K=16)
        all_weights.append(weights.cpu())
        all_weights_indices.append(weights_indices.cpu())
        
        del weights, weights_indices
        torch.cuda.empty_cache()

    for frame_idx in tqdm(range(n_frames), desc="Rendering with LBS"):
        if frame_idx > 0:
            # Apply LBS deformation
            prev_particle_pos = ctrl_pts[frame_idx - 1]
            cur_particle_pos = ctrl_pts[frame_idx]
            
            # 1. Compute bone transformations ONCE per frame
            motions = cur_particle_pos - prev_particle_pos
            bone_transforms = compute_bone_transforms(prev_particle_pos, motions, relations)
            
            for j in range(num_chunks):
                start = j * chunk_size
                end = min((j + 1) * chunk_size, n_gaussians)
                
                pos_chunk = current_pos[start:end]
                rot_chunk = current_rot[start:end]
                
                # Use precomputed weights for this chunk
                weights = all_weights[j].to("cuda")
                weights_indices = all_weights_indices[j].to("cuda")
                
                # Apply transformations to this chunk
                new_pos, new_rot = apply_bone_transforms_speedup(
                    pos_chunk, rot_chunk, prev_particle_pos, bone_transforms, weights, weights_indices
                )
                
                current_pos[start:end] = new_pos
                current_rot[start:end] = new_rot
                
                # Intermediate cleanup
                del weights, weights_indices, new_pos, new_rot
        
        # Update Gaussian positions
        gaussians._xyz = current_pos
        gaussians._rotation = current_rot
        
        # Render
        with torch.no_grad():
            results = render(view, gaussians, None, background)
            rendering = results["render"]
            
            # Convert to numpy immediately to free GPU memory
            if rendering.shape[0] == 4:
                image = rendering[:3].permute(1, 2, 0).cpu().numpy()
            else:
                image = rendering.permute(1, 2, 0).cpu().numpy()
            
            image = (image.clip(0, 1) * 255).astype(np.uint8)
            rendered_frames.append(image)
            
            del results, rendering
        
        # Periodically clear GPU cache
        if frame_idx % 10 == 0:
            torch.cuda.empty_cache()
    
    return rendered_frames


def save_comparison_video(gt_frames, pred_frames, output_path, fps=30):
    """Save side-by-side comparison video with ffmpeg transcoding."""
    if len(gt_frames) == 0:
        return
    h, w, _ = gt_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w * 2, h))
    
    for gt, pred in zip(gt_frames, pred_frames):
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (w, h))
        
        gt_bgr = cv2.cvtColor(gt, cv2.COLOR_RGB2BGR)
        pred_bgr = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
        
        cv2.putText(gt_bgr, "Ground Truth", (w // 20, h // 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(pred_bgr, "Prediction", (w // 20, h // 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        combined = np.hstack((gt_bgr, pred_bgr))
        out.write(combined)
    out.release()
    
    # Transcode with ffmpeg
    final_output = output_path
    temp_output = output_path.replace(".mp4", "_temp.mp4")
    os.rename(output_path, temp_output)
    
    cmd = [
        "ffmpeg", "-y", "-i", temp_output,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
        final_output
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.remove(temp_output)
        print(f"Saved comparison video to {final_output}")
    except Exception as e:
        print(f"Failed to transcode: {e}")
        os.rename(temp_output, final_output)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Rendering Metrics (PSNR, SSIM, LPIPS)")
    parser.add_argument("--base_path", type=str, required=True, 
                        help="Path to ground truth data")
    parser.add_argument("--prediction_dir", type=str, required=True, 
                        help="Path to experiment outputs")
    parser.add_argument("--output_file", type=str, default="results/render_results.csv", 
                        help="Path to output CSV file")
    parser.add_argument("--cam_idx", type=int, default=0,
                        help="Camera index to use for evaluation (default: 0, first camera)")
    parser.add_argument("--skip_render", action="store_true",
                        help="Skip LBS rendering and use pre-rendered inference.mp4")
    parser.add_argument("--ep_idx", type=int, default=None,
                        help="Specific episode index to evaluate")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Skip WandB logging")
    return parser.parse_args()


def calculate_metrics(gt_frames, pred_frames):
    """Calculate PSNR, SSIM, LPIPS metrics between GT and predicted frames."""
    if len(gt_frames) != len(pred_frames):
        print(f"Warning: Frame count mismatch: {len(gt_frames)} vs {len(pred_frames)}")
        min_len = min(len(gt_frames), len(pred_frames))
        gt_frames = gt_frames[:min_len]
        pred_frames = pred_frames[:min_len]
    
    if len(gt_frames) == 0:
        return {"psnr": 0, "ssim": 0, "lpips": 0, "count": 0}
    
    psnrs, ssims, lpipss = [], [], []
    
    for i in range(len(gt_frames)):
        gt = gt_frames[i]
        pred = pred_frames[i]
        
        gt_tensor = img2tensor(gt)
        if gt.shape != pred.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
        pred_tensor = img2tensor(pred)
        
        psnrs.append(psnr(pred_tensor, gt_tensor).item())
        ssims.append(ssim(pred_tensor, gt_tensor).item())
        lpipss.append(lpips(pred_tensor, gt_tensor, net_type='vgg').item())
    
    return {
        "psnr": np.mean(psnrs),
        "ssim": np.mean(ssims),
        "lpips": np.mean(lpipss),
        "count": len(psnrs)
    }


def main():
    args = parse_args()
    base_path = args.base_path
    prediction_dir = args.prediction_dir
    output_file = args.output_file
    
    if args.ep_idx is not None and output_file == "results/render_results.csv":
        obj_name = os.path.basename(args.base_path.rstrip("/"))
        output_file = f"results/{obj_name}_ep_{args.ep_idx}_render.csv"

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    
    with open(output_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Case Name",
            "Train PSNR", "Train SSIM", "Train LPIPS",
            "Test PSNR", "Test SSIM", "Test LPIPS"
        ])
        
        if args.ep_idx is not None:
            case_name = f"episode_{args.ep_idx}"
            episodes = [os.path.join(prediction_dir, case_name)]
        else:
            episodes = sorted(glob.glob(f"{prediction_dir}/episode_*"))
        
        for ep_dir in episodes:
            case_name = os.path.basename(ep_dir)
            print(f"\n{'='*60}")
            print(f"Processing {case_name}")
            print(f"{'='*60}")
            
            # ===== Load Split =====
            split_path = f"{base_path}/{case_name}/split.json"
            if not os.path.exists(split_path):
                print(f"Split file not found: {split_path}")
                continue
            
            with open(split_path, 'r') as f:
                split_data = json.load(f)
            
            train_range = split_data.get("train", [0, 0])
            test_range = split_data.get("test", [0, 0])
            offset = train_range[0]  # Start frame of contact interval
            n_frames = split_data.get("frame_len", test_range[1] - offset)
            
            print(f"Split: train={train_range}, test={test_range}, offset={offset}, n_frames={n_frames}")
            
            # ===== Find Camera Folder =====
            cam_folders = sorted(glob.glob(f"{base_path}/{case_name}/*cam*"))
            if not cam_folders:
                print(f"No camera folders found for {case_name}")
                continue
            
            # Select camera (default: first one or specified index)
            cam_idx = min(args.cam_idx, len(cam_folders) - 1)
            gt_cam_folder = cam_folders[cam_idx]
            cam_name = os.path.basename(gt_cam_folder)
            print(f"Using camera: {cam_name}")
            
            # ===== Load GT Video and Mask =====
            gt_video_path = os.path.join(gt_cam_folder, "undistorted.mp4")
            mask_path = os.path.join(gt_cam_folder, "mask_refined.h5")
            
            if not os.path.exists(gt_video_path):
                print(f"GT video not found: {gt_video_path}")
                continue
            
            if not os.path.exists(mask_path):
                print(f"Warning: Mask not found, will use raw frames: {mask_path}")
                mask_path = None
            
            # Read GT frames with mask applied
            print("Loading GT frames...")
            gt_frames = []
            cap = cv2.VideoCapture(gt_video_path)
            
            for frame_idx in tqdm(range(n_frames), desc="Loading GT"):
                actual_frame_idx = frame_idx + offset
                frame = read_video_frame(cap, actual_frame_idx)
                if frame is None:
                    print(f"Failed to read GT frame {actual_frame_idx}")
                    break
                
                # Apply mask
                if mask_path:
                    mask = load_mask_h5(mask_path, actual_frame_idx)
                    frame = apply_mask_to_frame(frame, mask)
                
                gt_frames.append(frame)
            
            cap.release()
            print(f"Loaded {len(gt_frames)} GT frames")
            
            # ===== Load or Render Predictions =====
            if args.skip_render:
                # Use pre-rendered video (fallback mode)
                pred_video_path = os.path.join(ep_dir, "inference.mp4")
                if not os.path.exists(pred_video_path):
                    print(f"Prediction video not found: {pred_video_path}")
                    continue
                
                print("Loading pre-rendered predictions...")
                pred_frames = []
                cap = cv2.VideoCapture(pred_video_path)
                for i in range(n_frames):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pred_frames.append(frame)
                cap.release()
            else:
                # ===== LBS Rendering =====
                print("Setting up 3DGS + LBS rendering...")
                
                # Load 3DGS model
                gs_path_pattern = f"{base_path}/{case_name}/splatfacto/*.ply"
                gs_paths = sorted(glob.glob(gs_path_pattern))
                if not gs_paths:
                    print(f"3DGS model not found: {gs_path_pattern}")
                    continue
                gs_path = gs_paths[0]
                print(f"Loading 3DGS from: {gs_path}")
                
                gaussians = GaussianModel(sh_degree=3)
                gaussians.load_ply(gs_path)
                gaussians = remove_gaussians_with_low_opacity(gaussians, 0.1)
                
                # Transform Gaussians from World Space to Marker Space
                T_marker2world = np.array([[ 9.92500579e-01, -1.22225711e-01,  1.86443478e-03,  1.36186366e-01],
                                          [ 5.43975403e-04, -1.08359291e-02, -9.99941142e-01, -1.88119571e-02],
                                          [ 1.22238720e-01,  9.92443176e-01, -1.06881781e-02,  7.19721945e-02],
                                          [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
                T_world2marker = np.linalg.inv(T_marker2world)
                
                R_motion = torch.tensor(T_world2marker[:3, :3], dtype=torch.float32, device="cuda")
                t_motion = torch.tensor(T_world2marker[:3, 3], dtype=torch.float32, device="cuda")
                
                # Transform positions
                xyz = gaussians.get_xyz
                new_xyz = (xyz @ R_motion.T) + t_motion
                gaussians._xyz = new_xyz
                
                # Transform rotations
                curr_rot = gaussians.get_rotation  # (N, 4)
                rot_world2marker = R.from_matrix(T_world2marker[:3, :3])
                
                # Scipy uses [x,y,z,w], GS uses [w,x,y,z]
                quats_world_scipy = np.roll(curr_rot.detach().cpu().numpy(), -1, axis=1)
                rots_world = R.from_quat(quats_world_scipy)
                rots_marker = rot_world2marker * rots_world
                quats_marker_scipy = rots_marker.as_quat()
                quats_marker = np.roll(quats_marker_scipy, 1, axis=1)
                gaussians._rotation = torch.tensor(quats_marker, dtype=curr_rot.dtype, device=curr_rot.device)
                
                # Load inference trajectory
                inference_path = os.path.join(ep_dir, "inference.pkl")
                if not os.path.exists(inference_path):
                    print(f"Inference trajectory not found: {inference_path}")
                    continue
                
                with open(inference_path, 'rb') as f:
                    ctrl_pts = pickle.load(f)
                ctrl_pts = torch.tensor(ctrl_pts, dtype=torch.float32, device="cuda")
                print(f"Loaded trajectory: {ctrl_pts.shape}")
                
                # Load camera calibration
                calibrate_path = f"{base_path}/{case_name}/calibrate.pkl"
                metadata_path = f"{base_path}/{case_name}/metadata.json"
                
                if not os.path.exists(calibrate_path) or not os.path.exists(metadata_path):
                    print(f"Camera calibration not found")
                    continue
                
                with open(calibrate_path, 'rb') as f:
                    c2ws = pickle.load(f)
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Apply world-to-marker transformation to cameras (matches train_warp.py)
                T_marker2world = np.array([[ 9.92500579e-01, -1.22225711e-01,  1.86443478e-03,  1.36186366e-01],
                                          [ 5.43975403e-04, -1.08359291e-02, -9.99941142e-01, -1.88119571e-02],
                                          [ 1.22238720e-01,  9.92443176e-01, -1.06881781e-02,  7.19721945e-02],
                                          [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
                T_world2marker = np.linalg.inv(T_marker2world)
                
                # Transform cameras to marker space
                c2ws = [T_world2marker @ c2w for c2w in c2ws]
                
                intrinsics = np.array(metadata["intrinsics"])
                WH = metadata["WH"]
                width, height = WH
                
                # Get camera for selected view
                c2w = c2ws[cam_idx]
                w2c = np.linalg.inv(c2w)
                intrinsic = intrinsics[cam_idx]
                
                view = create_camera_view(w2c, intrinsic, height, width, cam_idx)
                background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
                
                # Render with LBS
                print("Rendering predictions with LBS...")
                pred_frames = render_gaussians_lbs(
                    gaussians, ctrl_pts, view, background, n_frames
                )

                # Applying same mask applied to GT to rendered predictions
                if mask_path:
                    print("Applying mask to predictions...")
                    for i, frame in enumerate(tqdm(pred_frames, desc="Masking predictions")):
                        actual_frame_idx = i + offset
                        mask = load_mask_h5(mask_path, actual_frame_idx)
                        pred_frames[i] = apply_mask_to_frame(frame, mask)
            
            print(f"Loaded {len(pred_frames)} prediction frames")
            
            # ===== Calculate Metrics =====
            # Train/Test split (relative to loaded frames, not offset)
            train_start = 0
            train_end = train_range[1] - offset
            test_start = test_range[0] - offset
            test_end = min(test_range[1] - offset, len(gt_frames))
            
            # Clamp to available frames
            train_end = min(train_end, len(gt_frames), len(pred_frames))
            test_end = min(test_end, len(gt_frames), len(pred_frames))
            
            print(f"Eval ranges: Train: {train_start}-{train_end}, Test: {test_start}-{test_end}")
            
            gt_train = gt_frames[train_start:train_end]
            pred_train = pred_frames[train_start:train_end]
            gt_test = gt_frames[test_start:test_end]
            pred_test = pred_frames[test_start:test_end]
            
            # Save comparison video
            output_dir = os.path.dirname(output_file) or "."
            obj_name = os.path.basename(args.base_path.rstrip("/"))
            debug_video_path = os.path.join(output_dir, f"{obj_name}_ep_{args.ep_idx}_comparison.mp4")
            save_comparison_video(gt_frames, pred_frames, debug_video_path)
            
            # Calculate metrics
            print("Calculating metrics...")
            train_metrics = calculate_metrics(gt_train, pred_train)
            test_metrics = calculate_metrics(gt_test, pred_test)
            
            print(f"Train: PSNR={train_metrics['psnr']:.2f}, SSIM={train_metrics['ssim']:.4f}, LPIPS={train_metrics['lpips']:.4f}")
            print(f"Test:  PSNR={test_metrics['psnr']:.2f}, SSIM={test_metrics['ssim']:.4f}, LPIPS={test_metrics['lpips']:.4f}")
            
            writer.writerow([
                case_name,
                f"{train_metrics['psnr']:.4f}", f"{train_metrics['ssim']:.4f}", f"{train_metrics['lpips']:.4f}",
                f"{test_metrics['psnr']:.4f}", f"{test_metrics['ssim']:.4f}", f"{test_metrics['lpips']:.4f}"
            ])
            file.flush()

            # WandB logging
            if args.ep_idx is not None and not args.no_wandb:
                # Infer object name from base_path
                obj_name = os.path.basename(base_path)
                run_name = f"{obj_name}_ep_{args.ep_idx}"
                
                # Check if there is already an active run
                if wandb.run is None:
                    wandb.init(project="deformable_dynamics", name=run_name, resume="allow", config={"method": "PhysTwin"})
                
                wandb.log({
                    "train/psnr": train_metrics['psnr'],
                    "train/ssim": train_metrics['ssim'],
                    "train/lpips": train_metrics['lpips'],
                    "test/psnr": test_metrics['psnr'],
                    "test/ssim": test_metrics['ssim'],
                    "test/lpips": test_metrics['lpips'],
                    "test/comparison_video": wandb.Video(debug_video_path, fps=30, format="mp4")
                })
    
    if wandb.run is not None:
        wandb.finish()
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
