root="/oscar/data/gdk/hli230/projects/vitac-particle"
scene_name="043-dog"
output_dir="${root}/${scene_name}/gaussian_output_dynamic/episode_0000"

# views=("0" "1" "2")
views=("0")

# scenes=("double_lift_cloth_1" "double_lift_cloth_3" "double_lift_sloth" "double_lift_zebra"
#         "double_stretch_sloth" "double_stretch_zebra"
#         "rope_double_hand"
#         "single_clift_cloth_1" "single_clift_cloth_3"
#         "single_lift_cloth" "single_lift_cloth_1" "single_lift_cloth_3" "single_lift_cloth_4"
#         "single_lift_dinosor" "single_lift_rope" "single_lift_sloth" "single_lift_zebra"
#         "single_push_rope" "single_push_rope_1" "single_push_rope_4"
#         "single_push_sloth"
#         "weird_package")

exp_name='init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0'

python gs_render_dynamics.py \
    -s ${root}/${scene_name}/episode_0000  \
    -m ${root}/${scene_name}/episode_0000 \
    --output_dir ${output_dir} \
    --iteration 30000 \
    --name ${scene_name} \
    --start_frame 55 \
    --end_frame 261 \
    --num_frames 261 \
    --white_background \
    # --remove_gaussians
    # --render_all_frames

# python gaussian_splatting/img2video.py \
#             --image_folder ${output_dir}/${scene_name}/3 \
#             --video_path ${output_dir}/${scene_name}/3.mp4