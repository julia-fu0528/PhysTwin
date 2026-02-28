#!/usr/bin/env bash
set -euo pipefail
umask 002

# if hpc_log folder does not exist, create it
if [ ! -d "hpc_particleformer_log" ]; then
    mkdir -p hpc_particleformer_log
fi

PROCESSED_DIR="/oscar/data/gdk/hli230/projects/vitac-particle/processed"
OBJ_NAMES=$(ls "${PROCESSED_DIR}")

for OBJ in $OBJ_NAMES; do
    # Skip non-directory files if any
    if [ ! -d "${PROCESSED_DIR}/${OBJ}" ]; then
        continue
    fi

    # Make sure at least one episode exists
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

    job_script="hpc_particleformer_log/particleformer_multi_${OBJ}.sh"
    cat > "$job_script" <<EOT
#!/bin/bash
#SBATCH --job-name=pfm_${OBJ}
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --partition=3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --output=hpc_particleformer_log/${OBJ}_multi.%j.out
#SBATCH --error=hpc_particleformer_log/${OBJ}_multi.%j.err

set -euo pipefail

module load cuda ffmpeg
cd /oscar/data/gdk/hli230/projects/PhysTwin

# Run ParticleFormer training in multi-episode mode
pixi run python -m particleformer.train --use_wandb --object "${OBJ}" --mode multi-episode --cam_name "brics-odroid-022_cam1"
EOT

    chmod +x "$job_script"
    jid=$(sbatch --parsable "$job_script")
    echo "Submitted multi-episode job for ${OBJ} → $jid"
done
