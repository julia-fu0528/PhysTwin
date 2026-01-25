#!/bin/bash
# Usage: ./evaluate.sh [--base_path PATH] [--prediction_dir PATH]
# If no arguments provided, uses DATA_PATH below

DATA_PATH="/oscar/data/gdk/hli230/projects/vitac-particle/processed/001-rope"

# The base_path is where GT data lives (e.g., {DATA_PATH}/episode_0/final_data.pkl)
# The prediction_dir is where experiments are (e.g., {DATA_PATH}/experiments/episode_0)

# If ep_idx is provided in arguments, use the consolidated evaluate_all.py
if [[ "$*" == *"--ep_idx"* ]]; then
    python evaluate_all.py --base_path "$DATA_PATH" --prediction_dir "$DATA_PATH/experiments" "$@"
else
    # Fallback to individual scripts for batch processing
    python evaluate_chamfer.py --base_path "$DATA_PATH" --prediction_dir "$DATA_PATH/experiments" "$@"
    python evaluate_track.py --base_path "$DATA_PATH" --prediction_dir "$DATA_PATH/experiments" "$@"
    python gaussian_splatting/evaluate_render.py --base_path "$DATA_PATH" --prediction_dir "$DATA_PATH/experiments" "$@"
fi