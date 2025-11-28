root="/oscar/data/gdk/hli230/projects/vitac-particle"
scene_name="008-pink-cloth"

output_dir="${root}/${scene_name}/gaussian_output"
output_video_dir="${root}/${scene_name}/gaussian_output_video"
# scenes=("double_lift_cloth_1" "double_lift_cloth_3" "double_lift_sloth" "double_lift_zebra"
#         "double_stretch_sloth" "double_stretch_zebra"
#         "rope_double_hand"
#         "single_clift_cloth_1" "single_clift_cloth_3"
#         "single_lift_cloth" "single_lift_cloth_1" "single_lift_cloth_3" "single_lift_cloth_4"
#         "single_lift_dinosor" "single_lift_rope" "single_lift_sloth" "single_lift_zebra"
#         "single_push_rope" "single_push_rope_1" "single_push_rope_4"
#         "single_push_sloth"
#         "weird_package")


exp_name="init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0"

# interp poses
# echo "Generating interpolated poses for: $scene_name"
# python ./gaussian_splatting/generate_interp_poses.py \
#     --base_path ${root} \
#     --case_name ${scene_name} \
#     --n_interp 50 \
#     --use_all_cameras \
#     --close_loop

# train
python gs_train.py \
    -s ${root}/${scene_name}/episode_0000 \
    -m ${output_dir}/${exp_name} \
    --iterations 10000 \
    --lambda_depth 0.001 \
    --lambda_normal 0.0 \
    --lambda_anisotropic 0.0 \
    --lambda_seg 1.0 \
    --use_masks \
    --isotropic \
    --gs_init_opt 'hybrid'