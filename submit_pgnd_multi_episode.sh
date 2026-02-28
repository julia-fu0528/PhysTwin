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

    # Discover existing episodes (0-4)
    EXISTING_EPS=()
    for ep_id in 0 1 2 3 4; do
        if [ -d "${PROCESSED_DIR}/${OBJ}/episode_${ep_id}" ]; then
            EXISTING_EPS+=("$ep_id")
        fi
    done

    if [ ${#EXISTING_EPS[@]} -eq 0 ]; then
        echo "No episodes found for ${OBJ}, skipping."
        continue
    fi

    # Last existing episode for eval, preceding for training
    LAST_EP=${EXISTING_EPS[-1]}
    EVAL_START=${LAST_EP}
    EVAL_END=$((LAST_EP + 1))

    if [ ${#EXISTING_EPS[@]} -gt 1 ]; then
        # Training on all episodes before the last
        TRAIN_START=${EXISTING_EPS[0]}
        TRAIN_END=${LAST_EP}
    else
        # Only one episode: use it for both train and eval
        echo "Warning: Only one episode (${LAST_EP}) found for ${OBJ}. Using it for both training and testing."
        TRAIN_START=${LAST_EP}
        TRAIN_END=$((LAST_EP + 1))
    fi

    job_script="hpc_pgnd_log/pgnd_multi_${OBJ}.sh"
    cat > "$job_script" <<EOT
#!/bin/bash
#SBATCH --job-name=pgndm_${OBJ}
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --partition=3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=125GB
#SBATCH --output=hpc_pgnd_log/${OBJ}_multi.%j.out
#SBATCH --error=hpc_pgnd_log/${OBJ}_multi.%j.err

set -euo pipefail

module load cuda ffmpeg
cd /oscar/data/gdk/hli230/projects/PhysTwin

# Run PGND training in multi-episode mode
pixi run python -m pgnd.train \\
    train.name="${OBJ}_multi_episode" \\
    train.source_dataset_name="${PROCESSED_DIR}/${OBJ}" \\
    train.training_start_episode=${TRAIN_START} \\
    train.training_end_episode=${TRAIN_END} \\
    train.eval_start_episode=${EVAL_START} \\
    train.eval_end_episode=${EVAL_END} \\
    train.save_dataset=false \\
    train.save=true \\
    train.cam_name=brics-odroid-022_cam1 \\
    debug=False
EOT

    chmod +x "$job_script"
    jid=$(sbatch --parsable "$job_script")
    echo "Submitted multi-episode job for ${OBJ} (train eps ${TRAIN_START}-${TRAIN_END}, eval ep ${EVAL_START}) → $jid"
done
