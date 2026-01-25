#!/usr/bin/env bash
set -euo pipefail
umask 002

# if hpc_log folder does not exist, create it
if [ ! -d "hpc_log" ]; then
    mkdir -p hpc_log
fi

PROCESSED_DIR="/oscar/data/gdk/hli230/projects/vitac-particle/processed"
# OBJ_NAMES=$(ls "${PROCESSED_DIR}")
# hard code the object name
OBJ_NAMES="001-rope"

for OBJ in $OBJ_NAMES; do
    # Skip non-directory files if any
    if [ ! -d "${PROCESSED_DIR}/${OBJ}" ]; then
        continue
    fi
    
    # Find all episode directories and extract indices
    # This assumes episode directories are named 'episode_X'
    EP_INsDICES=$(ls -d "${PROCESSED_DIR}/${OBJ}"/episode_* 2>/dev/null | sed 's/.*_//' | sort -n | uniq | tr '\n' ',' | sed 's/,$//')
    
    if [ -z "$EP_INDICES" ]; then
        echo "No episodes found for ${OBJ}, skipping."
        continue
    fi

    job_script="hpc_dynamics_log/phystwin_${OBJ}.sh"
    cat > "$job_script" <<EOT
#!/bin/bash
#SBATCH --job-name=phystwin_${OBJ}
#SBATCH --array=${EP_INDICES}
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --partition=3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=125GB
#SBATCH --output=hpc_dynamics_log/${OBJ}_ep_%a.%A.out
#SBATCH --error=hpc_dynamics_log/${OBJ}_ep_%a.%A.err

set -euo pipefail

EP_IDX=\${SLURM_ARRAY_TASK_ID}
DATA_PATH="${PROCESSED_DIR}/${OBJ}"

module load cuda ffmpeg
cd /oscar/data/gdk/hli230/projects/PhysTwin

# Run optimization, training, inference and evaluation
pixi run python script_optimize.py --base_path "\$DATA_PATH" --ep_idx \$EP_IDX --no-gui
pixi run python script_train.py --base_path "\$DATA_PATH" --ep_idx \$EP_IDX --no-gui
pixi run python script_inference.py --base_path "\$DATA_PATH" --ep_idx \$EP_IDX
pixi run python evaluate_chamfer.py --base_path "\$DATA_PATH" --prediction_dir "\$DATA_PATH/experiments" --ep_idx \$EP_IDX
pixi run python evaluate_track.py --base_path "\$DATA_PATH" --prediction_dir "\$DATA_PATH/experiments" --ep_idx \$EP_IDX
pixi run python gaussian_splatting/evaluate_render.py --base_path "\$DATA_PATH" --prediction_dir "\$DATA_PATH/experiments" --ep_idx \$EP_IDX
EOT

    chmod +x "$job_script"
    jid=$(sbatch --parsable "$job_script")
    echo "Submitted job for ${OBJ} episodes [${EP_INDICES}] → \$jid"
done
