#!/bin/bash
# Debug run script for Contact Prediction.
# Run directly on a GPU node for development/debugging.
#
# Usage:
#   # Quick debug with 1 object, 2 epochs
#   bash contact_prediction/run_debug.sh
#
#   # Or source it to set up the env first
#   cd /oscar/data/gdk/hli230/projects/PhysTwin
#   bash contact_prediction/run_debug.sh

set -euo pipefail

cd /oscar/data/gdk/hli230/projects/PhysTwin

module load cuda ffmpeg 2>/dev/null || true

echo "=== Contact Prediction Debug Run ==="
echo "Using all 55 objects."
echo ""

pixi run python -m contact_prediction.train \
    --data_root /oscar/data/gdk/hli230/projects/vitac-particle/processed \
    --train_episodes 0 1 2 3 \
    --test_episodes 4 \
    --num_epochs 10 \
    --batch_size 512 \
    --num_workers 8 \
    --log_interval 5 \
    --eval_interval 1 \
    --output_dir outputs/contact_prediction_debug \
    --mixed_precision bf16 \
    --use_wandb


echo ""
echo "=== Debug run complete ==="
