#!/usr/bin/env bash
set -euo pipefail
umask 002

if [ ! -d "hpc_particleformer_log" ]; then
    mkdir -p hpc_particleformer_log
fi

PROCESSED_DIR="/oscar/data/gdk/hli230/projects/vitac-particle/processed"

job_script="hpc_particleformer_log/particleformer_multi_object.sh"
cat > "$job_script" <<EOT
#!/bin/bash
#SBATCH --job-name=pf_multi_obj
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --partition=3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --output=hpc_particleformer_log/particleformer_multi_obj.%j.out
#SBATCH --error=hpc_particleformer_log/particleformer_multi_obj.%j.err

set -euo pipefail

module load cuda ffmpeg
cd /oscar/data/gdk/hli230/projects/PhysTwin

pixi run python -m particleformer.train \\
    --use_wandb \\
    --mode multi-object \\
    --data_root "${PROCESSED_DIR}" \\
    --object multi-object \\
    --output_dir outputs/particleformer/multi_object \\
    --cam_name brics-odroid-022_cam1
EOT

chmod +x "$job_script"
jid=$(sbatch --parsable "$job_script")
echo "Submitted ParticleFormer multi-object job -> $jid"
