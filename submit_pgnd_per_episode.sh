#!/usr/bin/env bash
set -euo pipefail
umask 002

# if hpc_log folder does not exist, create it
if [ ! -d "hpc_pgnd_log" ]; then
    mkdir -p hpc_pgnd_log
fi

PROCESSED_DIR="/oscar/data/gdk/hli230/projects/vitac-particle/processed"
OBJ_NAMES=(
    002-rope-silk 003-cable 004-rubber-band 001-rope
    006-fur 008-pink-cloth 010-orange-cloth 011-green-cloth
    012-hat-cloth 013-glove-cloth 016-shirt-cloth
    015-airbag-cloth 017-chessboard-cloth 018-trashbag-cloth
    019-trashbag-plastic-cloth 021-bag-cloth 022-handkerchief
    024-glass-cleaner-cloth 023-cleaning-cloth
    025-bag-small-cloth 027-umbrella-bag-cloth 026-sock-cloth
    030-foam-flat-cloth 029-foam-cloth 038-mat-cloth
    040-paper-cloth 043-dog 045-cat 046-sponge
    048-butter-sponge 059-shoe 062-banana 063-flower
    068-nylon-rope 082-curtain-cloth 088-snake 090-sloth
    092-squirrel 096-octopus 100-puppet 095-watermelon
    103-ice-pack-cloth 109-pouch-cloth 110-shower-cap-cloth
    113-collar 115-cotton-gauze-cloth 120-bread-plush
    118-envelope-cloth 117-bubble-wrap-cloth
    121-croissant-plush 125-rabbit 135-makeup-sponge
    147-baking-mold 148-crepe-paper-cloth
    156-mesh-produce-bag-cloth 150-shredded-packing-paper-cloth
    157-sack-cloth 159-purse 163-bear 164-sheep
)

for OBJ in "${OBJ_NAMES[@]}"; do
    # Skip non-directory files if any
    if [ ! -d "${PROCESSED_DIR}/${OBJ}" ]; then
        continue
    fi
    
    # Find all episode directories and extract indices
    EP_INDICES=""
    for ep_dir in "${PROCESSED_DIR}/${OBJ}"/episode_*; do
        if [ -d "$ep_dir" ]; then
            # Extract numbers from episode_XXXX or episode_X
            EP_IDX=$(echo "$(basename "$ep_dir")" | sed 's/episode_//' | sed 's/^0*//')
            # If EP_IDX is empty after stripping zeros, it was episode_0
            if [ -z "$EP_IDX" ]; then EP_IDX=0; fi

            if [ -z "$EP_INDICES" ]; then
                EP_INDICES="$EP_IDX"
            else
                EP_INDICES="${EP_INDICES},${EP_IDX}"
            fi
        fi
    done
    
    if [ -z "$EP_INDICES" ]; then
        echo "No episodes found for ${OBJ}, skipping."
        continue
    fi

    job_script="hpc_pgnd_log/pgnd_${OBJ}.sh"
    cat > "$job_script" <<EOT
#!/bin/bash
#SBATCH --job-name=pgnd_${OBJ}
#SBATCH --array=${EP_INDICES}
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --partition=3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=125GB
#SBATCH --output=hpc_pgnd_log/${OBJ}_ep_%a.%A.out
#SBATCH --error=hpc_pgnd_log/${OBJ}_ep_%a.%A.err

set -euo pipefail

EP_IDX=\${SLURM_ARRAY_TASK_ID}
NEXT_EP=\$((EP_IDX + 1))

module load cuda ffmpeg
cd /oscar/data/gdk/hli230/projects/PhysTwin

# Run PGND training for a single episode
pixi run python -m pgnd.train \\
    train.name="${OBJ}_episode_\${EP_IDX}" \\
    train.source_dataset_name="${PROCESSED_DIR}/${OBJ}" \\
    train.training_start_episode=\${EP_IDX} \\
    train.training_end_episode=\${NEXT_EP} \\
    train.eval_start_episode=\${EP_IDX} \\
    train.eval_end_episode=\${NEXT_EP} \\
    train.save_dataset=false \\
    train.save=true \\
    debug=False
EOT

    chmod +x "$job_script"
    jid=$(sbatch --parsable "$job_script")
    echo "Submitted job for ${OBJ} episodes [${EP_INDICES}] → $jid"
done
