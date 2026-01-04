root="/mnt/data/ParticleData/processed/"


# scenes=("double_lift_cloth_1" "double_lift_cloth_3" "double_lift_sloth" "double_lift_zebra"
#         "double_stretch_sloth" "double_stretch_zebra"
#         "rope_double_hand"
#         "single_clift_cloth_1" "single_clift_cloth_3"
#         "single_lift_cloth" "single_lift_cloth_1" "single_lift_cloth_3" "single_lift_cloth_4"
#         "single_lift_dinosor" "single_lift_rope" "single_lift_sloth" "single_lift_zebra"
#         "single_push_rope" "single_push_rope_1" "single_push_rope_4"
#         "single_push_sloth"
#         "weird_package")

scenes=("2025-12-15-163-bear")

exp_name="init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0"
EP_NUM=10

# Iterate over each scene and episode
for scene_name in "${scenes[@]}"; do
    echo "Processing scene: $scene_name"

    output_dir="${root}/${scene_name}/gaussian_output"
    output_video_dir="${root}/${scene_name}/gaussian_output_video"
    
    for ((ep_idx=0; ep_idx< 1; ep_idx++)); do
        episode_name=$(printf "episode_%d" $ep_idx)
        echo "Processing episode: $episode_name"
        
        # Training
        python gs_train.py \
            -s ${root}/${scene_name}/${episode_name} \
            -m ${output_dir}/${episode_name} \
            --iterations 30000 \
            --lambda_depth 0.001 \
            --lambda_normal 0.0 \
            --lambda_anisotropic 0.0 \
            --lambda_seg 1.0 \
            --use_masks \
            --isotropic \
            --gs_init_opt 'hybrid' \
            --use_wandb \
            --wandb_project 'phystwin-3dgs'

        # # Rendering
        # python gs_render.py \
        #     -s ${root}/${scene_name}/${episode_name} \
        #     -m ${output_dir}/${exp_name}

        # # Convert images to video
        # python gaussian_splatting/img2video.py \
        #     --image_folder ${output_dir}/${exp_name}/test/ours_10000/renders \
        #     --video_path ${output_video_dir}/${exp_name}.mp4
    done
done
