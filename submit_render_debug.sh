#!/bin/bash
#SBATCH --job-name=rnd_debug
#SBATCH --nodes=1
#SBATCH --time=2:00:00
#SBATCH --partition=3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --output=hpc_render_log/debug_render.%j.out
#SBATCH --error=hpc_render_log/debug_render.%j.err

set -euo pipefail

module load cuda ffmpeg
cd /oscar/data/gdk/hli230/projects/PhysTwin
mkdir -p hpc_render_log

# Debug: render 001-rope, episode 4, particleformer
pixi run python render_multi_episode.py \
    --method particleformer \
    --data_root /oscar/data/gdk/hli230/projects/vitac-particle/processed \
    --object 001-rope \
    --test_episode 4 \
    --cam_name brics-odroid-022_cam1 \
    --output_dir results/render_debug

echo "=== ParticleFormer done ==="

# Debug: render 001-rope, episode 4, pgnd
pixi run python render_multi_episode.py \
    --method pgnd \
    --data_root /oscar/data/gdk/hli230/projects/vitac-particle/processed \
    --object 001-rope \
    --test_episode 4 \
    --cam_name brics-odroid-022_cam1 \
    --output_dir results/render_debug

echo "=== PGND done ==="
echo "Results in results/render_debug/"
