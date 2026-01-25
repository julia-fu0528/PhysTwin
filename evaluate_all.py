import argparse
import os
import subprocess
import csv
import wandb
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Run all evaluation scripts and log to WandB once")
    parser.add_argument("--base_path", type=str, required=True, help="Path to ground truth data")
    parser.add_argument("--prediction_dir", type=str, required=True, help="Path to experiment outputs")
    parser.add_argument("--ep_idx", type=int, required=True, help="Specific episode index to evaluate")
    return parser.parse_args()

def run_script(script_path, args_list):
    print(f"Running {script_path}...")
    cmd = [sys.executable, script_path] + args_list
    result = subprocess.run(cmd, capture_output=False) # Let it print to stdout
    if result.returncode != 0:
        print(f"Error running {script_path}")
        return False
    return True

def read_csv(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    with open(file_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if rows:
            return rows[0]
    return None

def main():
    args = parse_args()
    
    # Define script paths
    root_dir = os.path.dirname(os.path.abspath(__file__))
    chamfer_script = os.path.join(root_dir, "evaluate_chamfer.py")
    track_script = os.path.join(root_dir, "evaluate_track.py")
    render_script = os.path.join(root_dir, "gaussian_splatting/evaluate_render.py")
    
    # Common arguments
    common_args = [
        "--base_path", args.base_path,
        "--prediction_dir", args.prediction_dir,
        "--ep_idx", str(args.ep_idx),
        "--no_wandb"
    ]
    
    # Run scripts
    success = True
    success &= run_script(chamfer_script, common_args)
    success &= run_script(track_script, common_args)
    success &= run_script(render_script, common_args)
    
    if not success:
        print("Some evaluation scripts failed.")
    
    # Results files
    results_dir = "results"
    obj_name = os.path.basename(args.base_path.rstrip("/"))
    chamfer_csv = os.path.join(results_dir, f"{obj_name}_ep_{args.ep_idx}_chamfer.csv")
    track_csv = os.path.join(results_dir, f"{obj_name}_ep_{args.ep_idx}_track.csv")
    render_csv = os.path.join(results_dir, f"{obj_name}_ep_{args.ep_idx}_render.csv")
    
    # Comparison video path
    comparison_video = os.path.join(results_dir, f"{obj_name}_ep_{args.ep_idx}_comparison.mp4")
    # Wait, check evaluate_render.py for comparison video filename
    # Line 556: debug_video_path = os.path.join(output_dir, f"{case_name}_comparison.mp4")
    # case_name = os.path.basename(ep_dir) which is episode_{args.ep_idx}
    # Let's also update evaluate_render.py to use the more concrete name for comparison video
    
    # Log to WandB
    run_name = f"{obj_name}_ep_{args.ep_idx}"
    
    print(f"Logging to WandB run: {run_name}")
    wandb.init(
        project="deformable_dynamics", 
        name=run_name, 
        resume="allow", 
        config={
            "method": "PhysTwin",
            "object_name": obj_name,
            "ep_idx": args.ep_idx
        }
    )
    
    metrics = {}
    
    # Read Chamfer
    chamfer_res = read_csv(chamfer_csv)
    if chamfer_res:
        metrics.update({
            "train/chamfer_error": float(chamfer_res["Train Chamfer Error"]),
            "test/chamfer_error": float(chamfer_res["Test Chamfer Error"]),
            "train/chamfer_frame_num": int(chamfer_res["Train Frame Num"]),
            "test/chamfer_frame_num": int(chamfer_res["Test Frame Num"]),
        })
        
    # Read Track
    track_res = read_csv(track_csv)
    if track_res:
        metrics.update({
            "train/track_error": float(track_res["Train Track Error"]),
            "test/track_error": float(track_res["Test Track Error"]),
        })
        
    # Read Render
    render_res = read_csv(render_csv)
    if render_res:
        metrics.update({
            "train/psnr": float(render_res["Train PSNR"]),
            "train/ssim": float(render_res["Train SSIM"]),
            "train/lpips": float(render_res["Train LPIPS"]),
            "test/psnr": float(render_res["Test PSNR"]),
            "test/ssim": float(render_res["Test SSIM"]),
            "test/lpips": float(render_res["Test LPIPS"]),
        })
    
    # Log Comparison Video
    if os.path.exists(comparison_video):
        metrics["test/comparison_video"] = wandb.Video(comparison_video, fps=30, format="mp4")
    
    if metrics:
        wandb.log(metrics)
        print("Successfully logged all metrics to WandB.")
    else:
        print("No metrics found to log.")
        
    wandb.finish()

    # Delete files after uploading
    print("Cleaning up temporary evaluation files...")
    for f in [chamfer_csv, track_csv, render_csv, comparison_video]:
        if os.path.exists(f):
            os.remove(f)
            print(f"Deleted {f}")

if __name__ == "__main__":
    main()
