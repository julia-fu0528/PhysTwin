#!/usr/bin/env bash
set -euo pipefail
umask 002

# Batch rendering script for multi-episode checkpoints
# Submits SLURM jobs for all objects with both ParticleFormer and PGND

LOG_DIR="hpc_render_log"
mkdir -p "$LOG_DIR"

PROCESSED_DIR="/oscar/data/gdk/hli230/projects/vitac-particle/processed"
CAM_NAME="brics-odroid-022_cam1"
OUTPUT_DIR="results/render_multi_ep"

# Only submit for specified methods (default: both)
METHODS="${1:-particleformer pgnd}"

for OBJ_DIR in "${PROCESSED_DIR}"/*/; do
    OBJ=$(basename "$OBJ_DIR")
    
    # Skip non-directories
    if [ ! -d "${PROCESSED_DIR}/${OBJ}" ]; then
        continue
    fi

    # Check that at least one episode exists
    HAS_EPISODES=false
    for ep_dir in "${PROCESSED_DIR}/${OBJ}"/episode_*; do
        if [ -d "$ep_dir" ]; then
            HAS_EPISODES=true
            break
        fi
    done
    if [ "$HAS_EPISODES" = false ]; then
        echo "No episodes found for ${OBJ}, skipping."
        continue
    fi

    # Require BOTH particleformer and pgnd multi-episode checkpoints
    PF_CKPTS=( $(ls ${PROCESSED_DIR}/${OBJ}/train_ep_*_*.ckpt 2>/dev/null || true) )
    PGND_CKPTS=( $(ls ${PROCESSED_DIR}/${OBJ}/pgnd_ep_*_*.ckpt 2>/dev/null || true) )
    if [ ${#PF_CKPTS[@]} -eq 0 ] || [ ${#PGND_CKPTS[@]} -eq 0 ]; then
        echo "Missing checkpoint for ${OBJ} (PF: ${#PF_CKPTS[@]}, PGND: ${#PGND_CKPTS[@]}), skipping."
        continue
    fi

    for METHOD in $METHODS; do

        job_script="${LOG_DIR}/render_${METHOD}_${OBJ}.sh"
        cat > "$job_script" <<EOT
#!/bin/bash
#SBATCH --job-name=rnd_${METHOD:0:2}_${OBJ}
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --partition=3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --output=${LOG_DIR}/${OBJ}_${METHOD}.%j.out
#SBATCH --error=${LOG_DIR}/${OBJ}_${METHOD}.%j.err

set -euo pipefail

module load cuda ffmpeg
cd /oscar/data/gdk/hli230/projects/PhysTwin

pixi run python render_multi_episode.py \\
    --method ${METHOD} \\
    --data_root ${PROCESSED_DIR} \\
    --object ${OBJ} \\
    --cam_name ${CAM_NAME} \\
    --output_dir ${OUTPUT_DIR}/${OBJ}
EOT

        chmod +x "$job_script"
        jid=$(sbatch --parsable "$job_script")
        echo "Submitted ${METHOD} render for ${OBJ} → $jid"
    done
done
