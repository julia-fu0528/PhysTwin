DATA_PATH="/mnt/data/ParticleData/processed/096-octopus"
EP_IDX=0

python script_optimize.py --base_path $DATA_PATH --ep_idx $EP_IDX --no-gui
python script_train.py --base_path $DATA_PATH --ep_idx $EP_IDX --no-gui

python interactive_playground.py \
  --base_path $DATA_PATH \
  --case_name episode_$EP_IDX --n_ctrl_parts 1 \
  --bg_img_path data/bg.jpg