#!/usr/bin/env bash
set -euo pipefail
umask 002

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

TEST_OBJS=(003-cable 045-cat 059-shoe 117-bubble-wrap-cloth 159-purse)

is_test_obj() {
    local candidate="$1"
    for test_obj in "${TEST_OBJS[@]}"; do
        if [ "$candidate" = "$test_obj" ]; then
            return 0
        fi
    done
    return 1
}

TRAIN_OBJS=()
for obj in "${OBJ_NAMES[@]}"; do
    if ! is_test_obj "$obj"; then
        TRAIN_OBJS+=("$obj")
    fi
done

join_by_comma() {
    local IFS=","
    echo "$*"
}

TRAIN_HYDRA="[$(join_by_comma "${TRAIN_OBJS[@]}")]"
TEST_HYDRA="[$(join_by_comma "${TEST_OBJS[@]}")]"

job_script="hpc_pgnd_log/pgnd_multi_object.sh"
cat > "$job_script" <<EOT
#!/bin/bash
#SBATCH --job-name=pgnd_multi_obj
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --partition=3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=125GB
#SBATCH --output=hpc_pgnd_log/pgnd_multi_obj.%j.out
#SBATCH --error=hpc_pgnd_log/pgnd_multi_obj.%j.err

set -euo pipefail

module load cuda ffmpeg
cd /oscar/data/gdk/hli230/projects/PhysTwin
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256

pixi run python -m pgnd.train \\
    train.name="pgnd_multi_object" \\
    train.mode=multi-object \\
    train.object_name=multi-object \\
    train.source_dataset_name="${PROCESSED_DIR}" \\
    train.batch_size=4 \\
    "train.train_objects=${TRAIN_HYDRA}" \\
    "train.test_objects=${TEST_HYDRA}" \\
    train.save_dataset=false \\
    train.save=true \\
    train.cam_name=brics-odroid-022_cam1 \\
    debug=False
EOT

chmod +x "$job_script"
jid=$(sbatch --parsable "$job_script")
echo "Submitted PGND multi-object job (train=${#TRAIN_OBJS[@]}, test=${#TEST_OBJS[@]}) -> $jid"
