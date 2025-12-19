#!/bin/bash
set -euo pipefail


DATA_DIR="/oscar/data/gdk/hli230/projects/vitac-particle/163-bear"
session="163-bear"
MAX_ITER=20
EP_NUM=10

mkdir -p brics_log/${session}

optimize_jobids=()

# Optimize episodes - adjust the range as needed
for ((ep_idx=3; ep_idx< 4; ep_idx++)); do
  d="episode_${ep_idx}"
  mkdir -p "brics_log/${session}/$d"

  job_script="brics_log/${session}/$d/optimize_${ep_idx}.sh"
  cat > "$job_script" <<EOT
#!/bin/bash
#SBATCH --job-name=opt_${ep_idx}
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --partition=3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --output=brics_log/${session}/$d/optimize_${ep_idx}.%j.out
#SBATCH --error=brics_log/${session}/$d/optimize_${ep_idx}.%j.err

source /oscar/data/ssrinath/users/wfu16/PhysTwin/phystwin_env/bin/activate
python script_optimize.py \
  --base_path "$DATA_DIR" \
  --ep_idx ${ep_idx} \
  --max_iter ${MAX_ITER}

  
python script_train.py \
  --base_path "$DATA_DIR" \
  --ep_idx ${ep_idx}
EOT

  chmod +x "$job_script"
  jid=$(sbatch --parsable "$job_script")
  echo "Submitted optimization job ep=${ep_idx} → $jid"
  optimize_jobids+=("$jid")
  sleep 0.2
done

echo "Submitted ${#optimize_jobids[@]} jobs"
