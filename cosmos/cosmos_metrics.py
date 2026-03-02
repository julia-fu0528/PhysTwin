#!/usr/bin/env python3
"""Calculate SSIM, PSNR, LPIPS metrics for cosmos-generated videos.

Each video in the input folder is a vertically concatenated video where:
  - Top half = ground-truth
  - Bottom half = generated (cosmos prediction)

Filename format: {obj_name}_{episode_number}_{camera_name}.mp4

The script:
1. Splits each video into GT and generated halves
2. Loads train/test frame ranges from split.json
3. Applies segmentation masks from mask_refined.h5
4. Calculates SSIM, PSNR, LPIPS separately for train and test frames
5. Saves masked debug videos and a CSV of all results

Usage:
    pixi run python cosmos/cosmos_metrics.py
    pixi run python cosmos/cosmos_metrics.py --video_dir cosmos/epsplit_60obj_4train_1test
    pixi run python cosmos/cosmos_metrics.py --videos 001-rope_4_brics-odroid-022_cam1.mp4
"""

import os
import sys
import argparse
import csv
import json
import glob

import cv2
import h5py
import numpy as np
import torch
from tqdm import tqdm

# Add project root to path so we can import GS utilities
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from gaussian_splatting.utils.loss_utils import ssim
from gaussian_splatting.lpipsPyTorch.modules.lpips import LPIPS
from gaussian_splatting.utils.image_utils import psnr

_lpips_vgg = None

def get_lpips_vgg():
    global _lpips_vgg
    if _lpips_vgg is None:
        _lpips_vgg = LPIPS(net_type='vgg').cuda()
    return _lpips_vgg


def img2tensor(img):
    """Convert numpy image (H, W, C) uint8 to tensor (1, C, H, W) on GPU."""
    img = np.array(img, dtype=np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    return torch.from_numpy(img).unsqueeze(0).cuda()


def load_mask_h5(mask_path, frame_idx):
    """Load a single frame mask from H5 file."""
    try:
        with h5py.File(mask_path, 'r') as f:
            mask = f['data'][frame_idx]
            if mask.max() <= 1.0:
                mask = (mask * 255).astype(np.uint8)
            else:
                mask = mask.astype(np.uint8)
            return mask
    except Exception as e:
        print(f"Error loading mask from {mask_path} at frame {frame_idx}: {e}")
        return None


def apply_mask_to_frame(frame, mask):
    """Apply binary mask to frame (set background to black)."""
    if mask is None:
        return frame
    masked_frame = frame.copy()

    # Resize mask to match frame if needed
    if mask.shape[:2] != frame.shape[:2]:
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

    if len(mask.shape) == 2:
        mask_3ch = np.stack([mask, mask, mask], axis=-1)
    else:
        mask_3ch = mask
    masked_frame[mask_3ch == 0] = 0
    return masked_frame


def calculate_metrics(gt_frames, pred_frames):
    """Calculate PSNR, SSIM, LPIPS metrics between GT and predicted frames."""
    if len(gt_frames) != len(pred_frames):
        print(f"Warning: Frame count mismatch: {len(gt_frames)} vs {len(pred_frames)}")
        min_len = min(len(gt_frames), len(pred_frames))
        gt_frames = gt_frames[:min_len]
        pred_frames = pred_frames[:min_len]

    if len(gt_frames) == 0:
        return {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0, "count": 0}

    psnrs, ssims, lpipss = [], [], []

    lpips_vgg = get_lpips_vgg()

    for i in range(len(gt_frames)):
        gt = gt_frames[i]
        pred = pred_frames[i]

        gt_tensor = img2tensor(gt)
        if gt.shape != pred.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
        pred_tensor = img2tensor(pred)

        psnrs.append(psnr(pred_tensor, gt_tensor).item())
        ssims.append(ssim(pred_tensor, gt_tensor).item())
        lpipss.append(lpips_vgg(pred_tensor, gt_tensor).item())

    return {
        "psnr": np.mean(psnrs),
        "ssim": np.mean(ssims),
        "lpips": np.mean(lpipss),
        "count": len(psnrs),
    }


def parse_video_filename(filename):
    """Parse {obj_name}_{episode_number}_{camera_name}.mp4 into components.

    The camera_name itself contains underscores (e.g. brics-odroid-022_cam1),
    so we parse from the right: last part is camera suffix (cam0/cam1),
    second-to-last is camera device, third-to-last is episode number.
    Everything before that is obj_name.
    """
    stem = os.path.splitext(filename)[0]  # remove .mp4
    parts = stem.split("_")

    # Camera name is last two parts joined: e.g. "brics-odroid-022_cam1"
    cam_suffix = parts[-1]       # "cam1"
    cam_device = parts[-2]       # "brics-odroid-022"
    camera_name = f"{cam_device}_{cam_suffix}"

    # Episode number is third from last
    episode_number = int(parts[-3])

    # Object name is everything before that
    obj_name = "_".join(parts[:-3])

    return obj_name, episode_number, camera_name


def save_masked_video(gt_masked, gen_masked, output_path, fps=30):
    """Save vertically concatenated masked GT (top) + masked generated (bottom)."""
    if len(gt_masked) == 0:
        return
    h, w, _ = gt_masked[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h * 2))

    for gt_frame, gen_frame in zip(gt_masked, gen_masked):
        gt_bgr = cv2.cvtColor(gt_frame, cv2.COLOR_RGB2BGR)
        gen_bgr = cv2.cvtColor(gen_frame, cv2.COLOR_RGB2BGR)
        combined = np.vstack((gt_bgr, gen_bgr))
        out.write(combined)
    out.release()

    # Transcode with ffmpeg for better compatibility
    temp_path = output_path.replace(".mp4", "_temp.mp4")
    os.rename(output_path, temp_path)
    cmd = f'ffmpeg -y -i "{temp_path}" -c:v libx264 -pix_fmt yuv420p -crf 23 "{output_path}" 2>/dev/null'
    ret = os.system(cmd)
    if ret == 0:
        os.remove(temp_path)
    else:
        os.rename(temp_path, output_path)


def process_video(video_path, data_root, masked_output_dir):
    """Process a single cosmos video and return metrics dict."""
    filename = os.path.basename(video_path)
    obj_name, episode_number, camera_name = parse_video_filename(filename)

    # Clamp episode number to 4 if greater
    ep_num = min(episode_number, 4)

    episode_dir = os.path.join(data_root, obj_name, f"episode_{ep_num}")
    split_path = os.path.join(episode_dir, "split.json")
    metadata_path = os.path.join(episode_dir, "metadata.json")
    mask_path = os.path.join(episode_dir, camera_name, "mask_refined.h5")

    # Check required files exist
    for p, name in [(split_path, "split.json"), (metadata_path, "metadata.json")]:
        if not os.path.exists(p):
            print(f"  WARNING: {name} not found at {p}, skipping {filename}")
            return None

    # Load split and metadata
    with open(split_path, 'r') as f:
        split_data = json.load(f)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    train_range = split_data.get("train", [0, 0])
    test_range = split_data.get("test", [0, 0])
    offset = metadata.get("start_frame", 0)

    # Read video frames
    cap = cv2.VideoCapture(video_path)
    total_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    total_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    half_h = total_height // 2

    gt_frames = []
    gen_frames = []

    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gt_frames.append(frame_rgb[:half_h, :, :])
        gen_frames.append(frame_rgb[half_h:, :, :])
    cap.release()

    actual_n_frames = len(gt_frames)
    print(f"  Video: {total_width}x{total_height}, {actual_n_frames} frames, "
          f"split into {total_width}x{half_h} halves")

    # Check if mask file exists
    has_masks = os.path.exists(mask_path)
    if not has_masks:
        print(f"  WARNING: mask not found at {mask_path}, computing metrics without masks")

    # Apply masks
    gt_masked = []
    gen_masked = []
    for i in range(actual_n_frames):
        actual_frame_idx = i + offset
        if has_masks:
            mask = load_mask_h5(mask_path, actual_frame_idx)
        else:
            mask = None
        gt_masked.append(apply_mask_to_frame(gt_frames[i], mask))
        gen_masked.append(apply_mask_to_frame(gen_frames[i], mask))

    # Save masked debug video
    os.makedirs(masked_output_dir, exist_ok=True)
    masked_video_path = os.path.join(masked_output_dir, filename)
    save_masked_video(gt_masked, gen_masked, masked_video_path, fps=fps)
    print(f"  Saved masked video to {masked_video_path}")

    # Compute relative frame indices for train/test splits
    train_start_rel = max(0, train_range[0] - offset)
    train_end_rel = min(actual_n_frames, train_range[1] - offset)
    test_start_rel = max(0, test_range[0] - offset)
    test_end_rel = min(actual_n_frames, test_range[1] - offset)

    print(f"  Split: train frames [{train_start_rel}:{train_end_rel}], "
          f"test frames [{test_start_rel}:{test_end_rel}] (relative)")

    result = {
        "obj_name": obj_name,
        "episode": episode_number,
        "camera": camera_name,
    }

    # Train metrics
    if train_end_rel > train_start_rel:
        gt_train = gt_masked[train_start_rel:train_end_rel]
        gen_train = gen_masked[train_start_rel:train_end_rel]
        train_metrics = calculate_metrics(gt_train, gen_train)
        result["train_psnr"] = train_metrics["psnr"]
        result["train_ssim"] = train_metrics["ssim"]
        result["train_lpips"] = train_metrics["lpips"]
        print(f"  Train ({len(gt_train)} frames): "
              f"PSNR={train_metrics['psnr']:.4f}, "
              f"SSIM={train_metrics['ssim']:.4f}, "
              f"LPIPS={train_metrics['lpips']:.4f}")
    else:
        result["train_psnr"] = float("nan")
        result["train_ssim"] = float("nan")
        result["train_lpips"] = float("nan")
        print(f"  Train: no frames in range")

    # Test metrics
    if test_end_rel > test_start_rel:
        gt_test = gt_masked[test_start_rel:test_end_rel]
        gen_test = gen_masked[test_start_rel:test_end_rel]
        test_metrics = calculate_metrics(gt_test, gen_test)
        result["test_psnr"] = test_metrics["psnr"]
        result["test_ssim"] = test_metrics["ssim"]
        result["test_lpips"] = test_metrics["lpips"]
        print(f"  Test ({len(gt_test)} frames): "
              f"PSNR={test_metrics['psnr']:.4f}, "
              f"SSIM={test_metrics['ssim']:.4f}, "
              f"LPIPS={test_metrics['lpips']:.4f}")
    else:
        result["test_psnr"] = float("nan")
        result["test_ssim"] = float("nan")
        result["test_lpips"] = float("nan")
        print(f"  Test: no frames in range")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Calculate metrics for cosmos-generated videos"
    )
    parser.add_argument(
        "--video_dir", type=str,
        default=os.path.join(SCRIPT_DIR, "epsplit_60obj_4train_1test"),
        help="Directory containing cosmos videos",
    )
    parser.add_argument(
        "--data_root", type=str,
        default="/oscar/data/gdk/hli230/projects/vitac-particle/processed",
        help="Root directory containing object/episode data",
    )
    parser.add_argument(
        "--output_csv", type=str, default=None,
        help="Output CSV path (default: {video_dir}_metrics.csv)",
    )
    parser.add_argument(
        "--masked_dir", type=str, default=None,
        help="Output dir for masked debug videos (default: {video_dir}_masked)",
    )
    parser.add_argument(
        "--videos", type=str, nargs="*", default=None,
        help="Specific video filenames to process (default: all .mp4 in video_dir)",
    )
    args = parser.parse_args()

    video_dir = args.video_dir
    data_root = args.data_root
    output_csv = args.output_csv or (video_dir.rstrip("/") + "_metrics.csv")
    masked_dir = args.masked_dir or (video_dir.rstrip("/") + "_masked")

    # Collect video files
    if args.videos:
        video_files = [os.path.join(video_dir, v) for v in args.videos]
    else:
        video_files = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))

    print(f"Found {len(video_files)} videos in {video_dir}")
    print(f"Data root: {data_root}")
    print(f"Output CSV: {output_csv}")
    print(f"Masked videos dir: {masked_dir}")
    print("=" * 60)

    results = []
    for i, vf in enumerate(video_files):
        print(f"\n[{i+1}/{len(video_files)}] Processing: {os.path.basename(vf)}")
        try:
            result = process_video(vf, data_root, masked_dir)
            if result is not None:
                results.append(result)
        except Exception as e:
            print(f"  ERROR processing {os.path.basename(vf)}: {e}")
            import traceback
            traceback.print_exc()

    # Write CSV
    if results:
        fieldnames = [
            "obj_name", "episode", "camera",
            "train_psnr", "train_ssim", "train_lpips",
            "test_psnr", "test_ssim", "test_lpips",
        ]
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\n{'=' * 60}")
        print(f"Saved metrics for {len(results)} videos to {output_csv}")

        # Print summary averages
        for split in ["train", "test"]:
            valid = [r for r in results if not np.isnan(r[f"{split}_psnr"])]
            if valid:
                avg_psnr = np.mean([r[f"{split}_psnr"] for r in valid])
                avg_ssim = np.mean([r[f"{split}_ssim"] for r in valid])
                avg_lpips = np.mean([r[f"{split}_lpips"] for r in valid])
                print(f"  {split.upper()} avg ({len(valid)} videos): "
                      f"PSNR={avg_psnr:.4f}, SSIM={avg_ssim:.4f}, LPIPS={avg_lpips:.4f}")
    else:
        print("\nNo results to save.")


if __name__ == "__main__":
    main()
