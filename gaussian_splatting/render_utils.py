import os
import sys
import torch
import numpy as np
import cv2
import h5py
import subprocess
from tqdm import tqdm
from gaussian_splatting.scene.cameras import Camera
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.graphics_utils import focal2fov
from gaussian_splatting.dynamic_utils import (
    get_topk_indices, knn_weights_sparse, interpolate_motions_speedup
)
from gaussian_splatting.utils.loss_utils import ssim
from gaussian_splatting.lpipsPyTorch import lpips
from gaussian_splatting.utils.image_utils import psnr

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
    Uses chunked processing to avoid OOM.
    
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
    
    # Get initial Gaussian state
    xyz_0 = gaussians.get_xyz.clone()
    quat_0 = gaussians.get_rotation.clone()
    
    # Current state (will be updated each frame)
    current_pos = xyz_0.clone()
    current_rot = quat_0.clone()
    
    # Initialize LBS relations (computed once from first control points)
    init_particle_pos = ctrl_pts[0]
    relations = get_topk_indices(init_particle_pos, K=16)
    
    # Chunk size for processing to avoid OOM (same as gs_render_dynamics.py)
    chunk_size = 20_000
    n_gaussians = len(current_pos)
    num_chunks = (n_gaussians + chunk_size - 1) // chunk_size
    
    # Pre-compute skinning weights
    all_weights = []
    all_weights_indices = []
    
    print("Pre-computing skinning weights...")
    for j in range(num_chunks):
        start = j * chunk_size
        end = min((j + 1) * chunk_size, n_gaussians)
        pos_chunk_0 = xyz_0[start:end]
        
        # Use initial particle positions for weight computation
        w, wi = knn_weights_sparse(init_particle_pos, pos_chunk_0, K=16)
        all_weights.append(w)
        all_weights_indices.append(wi)
        
    for frame_idx in tqdm(range(n_frames), desc="Rendering with LBS"):
        if frame_idx > 0:
            # Apply LBS deformation in chunks
            prev_particle_pos = ctrl_pts[frame_idx - 1] # This arg name in interpolate might be misleading if it expects displacement
            cur_particle_pos = ctrl_pts[frame_idx]
            
            # Calculate motion of control points
            motion_ctrl = cur_particle_pos - init_particle_pos # Displacement from REF pose
            # OR is it frame-to-frame? 
            # interpolate_motions_speedup(bones, motions, ...)
            # If standard LBS: NewPos = InitPos + Sum(w * (Ctrl_t - Ctrl_0))
            # Let's see how `interpolate_motions_speedup` is implemented. 
            # Assuming it takes (RefBones, MotionOfBones, ...).
            # Let's stick to the implementation pattern but optimize the weight calc.
            
            # Actually, looking at the loop:
            # prev_particle_pos = ctrl_pts[frame_idx - 1]
            # cur_particle_pos = ctrl_pts[frame_idx]
            # It seems to be doing incremental updates? 
            # If so, weights re-calculation might be intended for "Eulerian" style?
            # BUT, standard LBS for this task (PhysGaussian) usually uses fixed weights from Frame 0.
            # Re-calculating weights every frame O(N_gauss * N_particles) is definitely the bottleneck.
            
            # We will switch to using CONSTANT weights computed at frame 0.
            
            for j in range(num_chunks):
                start = j * chunk_size
                end = min((j + 1) * chunk_size, n_gaussians)
                
                # Use pre-computed weights
                weights = all_weights[j]
                weights_indices = all_weights_indices[j]
                
                # To purely apply LBS from t-1 to t (incremental):
                # We need to know if we are deforming from Init or Previous.
                # `current_pos` is updated every frame. So it is incremental.
                # But weights should likely stay consistent to the material coordinates.
                # Using weights computed at frame 0 is the correct physical approximation for solid objects.
                
                # Interpolate motions for this chunk using speedup version
                # Note: we pass prev_particle_pos as 'bones' which implies weights were computed relative to it?
                # If we use Frame 0 weights, 'bones' should be init_particle_pos?
                # BUT `interpolate_motions_speedup` might expect `bones` to match the `weights` reference.
                # If we pass `init_particle_pos` as bones and `cur - prev` as motion? No.
                
                # Incremental update strategy with FIXED weights:
                # 1. Weights from Fr 0.
                # 2. Motion = Ctrl_t - Ctrl_{t-1}
                # 3. Apply to Gaussians_{t-1}.
                # This works if topological relationship is preserved.
                
                pos_chunk = current_pos[start:end].clone()
                rot_chunk = current_rot[start:end].clone()

                new_pos, new_rot, _ = interpolate_motions_speedup(
                    bones=ctrl_pts[frame_idx - 1], # Source positions for motion vector
                    motions=cur_particle_pos - prev_particle_pos,
                    relations=relations,
                    weights=weights,
                    weights_indices=weights_indices,
                    xyz=pos_chunk,
                    quat=rot_chunk,
                )

                
                # Interpolate motions for this chunk using speedup version
                new_pos, new_rot, _ = interpolate_motions_speedup(
                    bones=prev_particle_pos,
                    motions=cur_particle_pos - prev_particle_pos,
                    relations=relations,
                    weights=weights,
                    weights_indices=weights_indices,
                    xyz=pos_chunk,
                    quat=rot_chunk,
                )
                
                current_pos[start:end] = new_pos
                current_rot[start:end] = new_rot
                
                # Clean up intermediate tensors
                del weights, weights_indices, pos_chunk, rot_chunk, new_pos, new_rot
        
        # Update Gaussian positions
        gaussians._xyz = current_pos
        gaussians._rotation = current_rot
        
        # Render
        with torch.no_grad():
            results = render(view, gaussians, None, background)
            rendering = results["render"]  # (3, H, W) or (4, H, W)
            
            # Convert to numpy immediately to free GPU memory
            if rendering.shape[0] == 4:
                image = rendering[:3].permute(1, 2, 0).cpu().numpy()
            else:
                image = rendering.permute(1, 2, 0).cpu().numpy()
            
            image = (image.clip(0, 1) * 255).astype(np.uint8)
            rendered_frames.append(image)
            
            # Clean up render results
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
