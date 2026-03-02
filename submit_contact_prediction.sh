#!/usr/bin/env bash
# Submit Contact Prediction training job via SLURM.
# Job array training on ALL 55 objects (eps 0-3 train, ep 4 test) across all cameras.

set -euo pipefail
umask 002

# Create log directory
LOG_DIR="hpc_contact_log"
mkdir -p "$LOG_DIR"

job_script="${LOG_DIR}/contact_prediction.sh"
cat > "$job_script" <<'EOT'
#!/bin/bash
#SBATCH --job-name=contact_pred
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --partition=3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=125GB
#SBATCH --output=hpc_contact_log/contact_pred_%A_%a.out
#SBATCH --error=hpc_contact_log/contact_pred_%A_%a.err
#SBATCH --array=0-35

set -euo pipefail

# Array of all 36 RGB cameras
CAMERAS=(
    "brics-odroid-001_cam0" "brics-odroid-001_cam1" "brics-odroid-002_cam0"
    "brics-odroid-006_cam0" "brics-odroid-007_cam0" "brics-odroid-007_cam1"
    "brics-odroid-008_cam0" "brics-odroid-008_cam1" "brics-odroid-009_cam0"
    "brics-odroid-009_cam1" "brics-odroid-010_cam0" "brics-odroid-010_cam1"
    "brics-odroid-011_cam0" "brics-odroid-012_cam0" "brics-odroid-012_cam1"
    "brics-odroid-013_cam0" "brics-odroid-013_cam1" "brics-odroid-014_cam1"
    "brics-odroid-015_cam0" "brics-odroid-015_cam1" "brics-odroid-016_cam0"
    "brics-odroid-017_cam0" "brics-odroid-017_cam1" "brics-odroid-019_cam1"
    "brics-odroid-021_cam0" "brics-odroid-021_cam1" "brics-odroid-022_cam0"
    "brics-odroid-022_cam1" "brics-odroid-023_cam0" "brics-odroid-024_cam0"
    "brics-odroid-024_cam1" "brics-odroid-025_cam0" "brics-odroid-025_cam1"
    "brics-odroid-027_cam0" "brics-odroid-027_cam1" "brics-odroid-028_cam0"
)

# Get the camera for this task
CAM_NAME="${CAMERAS[$SLURM_ARRAY_TASK_ID]}"
echo "Running task ID ${SLURM_ARRAY_TASK_ID} with camera: ${CAM_NAME}"

module load cuda ffmpeg
cd /oscar/data/gdk/hli230/projects/PhysTwin

pixi run python -m contact_prediction.train \
    --data_root /oscar/data/gdk/hli230/projects/vitac-particle/processed \
    --train_episodes 0 1 2 3 \
    --test_episodes 4 \
    --num_epochs 10 \
    --batch_size 512 \
    --num_workers 8 \
    --cam_name "${CAM_NAME}" \
    --use_wandb \
    --wandb_project contact_prediction \
    --output_dir outputs/contact_prediction_"${CAM_NAME}" \
    --mixed_precision bf16
EOT

chmod +x "$job_script"
jid=$(sbatch --parsable "$job_script")
echo "Submitted contact prediction job array → $jid (Tasks 0-35)"
echo "Logs: ${LOG_DIR}/contact_pred_${jid}_<task_id>.out"
