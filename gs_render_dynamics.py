#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from gaussian_splatting.scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_splatting.gaussian_renderer import render
import torchvision
from gaussian_splatting.utils.general_utils import safe_state
from argparse import ArgumentParser
from gaussian_splatting.arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_splatting.gaussian_renderer import GaussianModel
import sys
from PIL import Image
import torch.nn.functional as F
try:
    from diff_gaussian_rasterization import SparseGaussianAdam

    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

import numpy as np
from kornia import create_meshgrid
import copy
from gs_render import (
    remove_gaussians_with_mask,
    remove_gaussians_with_low_opacity,
    remove_gaussians_with_point_mesh_distance,
)
from gaussian_splatting.dynamic_utils import (
    interpolate_motions,
    create_relation_matrix,
    knn_weights,
    get_topk_indices,
    quat2mat,
    mat2quat,
)
import pickle
import imageio
import decord


def create_grid_frame(images, grid_shape=None):
    """
    Create a grid image from a list of images (H, W, 3) uint8.
    """
    if not images:
        return None

    n = len(images)
    if grid_shape is None:
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
    else:
        rows, cols = grid_shape

    h, w = images[0].shape[:2]
    grid_h = rows * h
    grid_w = cols * w
    grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

    for idx, img in enumerate(images):
        r = idx // cols
        c = idx % cols
        if r < rows and c < cols:
            # Handle RGBA images by taking only RGB channels
            if img.shape[2] == 4:
                img = img[:, :, :3]
            grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = img

    return grid


def render_set(
    output_path,
    name,
    views,
    gaussians_list,
    pipeline,
    background,
    train_test_exp,
    separate_sh,
    disable_sh=False,
    start_frame=0,
):
    video_readers = {}
    
    print(f"views shape: {len(views)}")
    render_path = os.path.join(output_path, name)
    makedirs(render_path, exist_ok=True)

    # Render all views
    view_indices = list(range(len(views)))
    selected_views = [views[i] for i in view_indices]

    # store per-view frames for later grid video
    all_view_frames = []

    for idx, view in enumerate(tqdm(selected_views, desc="Rendering progress")):
        if idx != 23:
            continue
        print(f"Processing view {idx}")
        # Path for this view's video
        view_video_path = os.path.join(render_path, f"{idx}.mp4")

        # Collect frames for this view to create a video
        video_frames = []

        for frame_idx, gaussians in enumerate(gaussians_list):
            if disable_sh:
                override_color = gaussians.get_features_dc.squeeze()
                results = render(
                    view,
                    gaussians,
                    pipeline,
                    background,
                    override_color=override_color,
                    # use_trained_exp=train_test_exp,
                    use_trained_exp=False,
                    separate_sh=separate_sh,
                )
            else:
                results = render(
                    view,
                    gaussians,
                    pipeline,
                    background,
                    # use_trained_exp=train_test_exp,
                    use_trained_exp=False,
                    separate_sh=separate_sh,
                )

            # RGB(A) rendering
            rendering = results["render"]  # (3 or 4, H, W) in range [0, 1]

            # Load original image from camera if available
            if hasattr(view, 'image_path') and view.image_path is not None and os.path.exists(view.image_path):
                try:
                    if view.image_path.endswith(".mp4"):
                        if view.image_path not in video_readers:
                            video_readers[view.image_path] = decord.VideoReader(view.image_path, ctx=decord.cpu(0))
                        reader = video_readers[view.image_path]
                        # Use frame_idx + start_frame as the frame index in the video
                        gt_image = reader[frame_idx + start_frame].asnumpy()
                        gt_image = Image.fromarray(gt_image)
                    else:
                        image_path = view.image_path.replace("000000", f"{frame_idx + start_frame:06d}")
                        gt_image = Image.open(image_path).convert("RGB")
                    
                    gt_image = gt_image.resize((view.image_width, view.image_height))
                    gt_image = torch.from_numpy(np.array(gt_image)).float() / 255.0  # [0, 1]
                    gt_image = gt_image.permute(2, 0, 1)  # (3, H, W)
                    gt_image = gt_image.to(rendering.device)
                except Exception as e:
                    print(f"Warning: Could not load image from {view.image_path}: {e}")
                    # Fallback: use background color
                    bg_color = background if isinstance(background, torch.Tensor) else torch.tensor(background, device=rendering.device)
                    gt_image = bg_color.unsqueeze(1).unsqueeze(2).expand(3, rendering.shape[1], rendering.shape[2])  # (3, H, W)
            else:
                # Fallback: use background color
                bg_color = background if isinstance(background, torch.Tensor) else torch.tensor(background, device=rendering.device)
                gt_image = bg_color.unsqueeze(1).unsqueeze(2).expand(3, rendering.shape[1], rendering.shape[2])  # (3, H, W)
            
            # Resize gt_image to match rendering resolution if needed
            if gt_image.shape[1] != rendering.shape[1] or gt_image.shape[2] != rendering.shape[2]:
                gt_image = F.interpolate(
                    gt_image.unsqueeze(0),
                    size=(rendering.shape[1], rendering.shape[2]),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
            
            # Alpha blending: foreground (GS) at 0.6 alpha, background (original) at 0.4 alpha
            # Formula: composite = foreground * alpha + background * (1 - alpha)
            # rendered_image is (3, H, W), gt_image is (3, H, W)
            alpha_gs = 0.6  # Alpha for GS rendering
            alpha_bg = 0.4  # Alpha for background (original image)
            
            # Create alpha channel: 0.6 where GS is visible, 0.4 elsewhere
            # Check if GS is visible (non-zero pixels)
            gs_visible = (rendering.sum(dim=0) > 0.01).float()  # (H, W) - 1 where GS visible, 0 elsewhere
            alpha_channel = alpha_gs * gs_visible + alpha_bg * (1 - gs_visible)  # (H, W)
            alpha_channel = alpha_channel.unsqueeze(0)  # (1, H, W)
            
            # Proper alpha blending: composite = foreground * alpha + background * (1 - alpha)
            # But we want GS at 0.6 and background at 0.4, so:
            # composite = rendering * 0.6 + gt_image * 0.4
            # give rendering an alpha channel of 0.6
            rendering_alpha = torch.ones_like(rendering[0, :, :]) 
            rendering = torch.cat([rendering[0:3, :, :], rendering_alpha.unsqueeze(0)], dim=0)
            # give gt_image an alpha channel of 0.4
            gt_image_alpha = torch.ones_like(gt_image[0, :, :])
            gt_image = torch.cat([gt_image[0:3, :, :], gt_image_alpha.unsqueeze(0)], dim=0)
            composite_rgb = rendering * alpha_gs + gt_image * alpha_bg  # (3, H, W)
            # Combine RGB with alpha channel to create RGBA
            composite_rgba = torch.cat([composite_rgb, alpha_channel], dim=0)  # (4, H, W)

            # Convert composite RGB to uint8 frame and store for video
            frame_np = (
                composite_rgb.clamp(0, 1)
                # rendering.clamp(0, 1)
                .detach()
                .cpu()
                .permute(1, 2, 0)  # (H, W, 3)
                .numpy()
            )
            frame_np = (frame_np * 255).astype("uint8")
            video_frames.append(frame_np)

        # After all frames for this view, write out a video
        if len(video_frames) > 0:
            imageio.mimwrite(view_video_path, video_frames, fps=30)
            print(f"Saved video to {view_video_path}")
            all_view_frames.append(video_frames)
        else:
            print(f"No frames to save for view {idx}")

    # After all individual view videos, create a grid video across views
    if all_view_frames:
        min_len = min(len(v) for v in all_view_frames if v)
        if min_len > 0:
            grid_frames = []
            for frame_idx in range(min_len):
                frame_imgs = [v[frame_idx] for v in all_view_frames]
                grid_img = create_grid_frame(frame_imgs)
                if grid_img is not None:
                    grid_frames.append(grid_img)
            if grid_frames:
                grid_video_path = os.path.join(render_path, "grid_video.mp4")
                imageio.mimwrite(grid_video_path, grid_frames, fps=30)
                print(f"Saved grid video to {grid_video_path}")


def render_sets(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    skip_train: bool,
    skip_test: bool,
    separate_sh: bool,
    remove_gaussians: bool = False,
    name: str = "dynamic",
    output_dir: str = "./gaussian_output_dynamic",
    start_frame: int = 0,
    end_frame: int = 60000,
    num_frames: int = 60000,
    render_all_frames: bool = True,
):
    print(f"output_dir: {output_dir}")
    with torch.no_grad():
        output_path = output_dir
        print(f"dataset.white_background: {dataset.white_background}")

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        print(f"bg_color: {bg_color}")
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        print(f"dataset.sh_degree: {dataset.sh_degree}")
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, start_frame=start_frame, end_frame=end_frame, num_frames=num_frames)

        # remove gaussians that are outside the mask
        if remove_gaussians:
            gaussians = remove_gaussians_with_mask(gaussians, scene.getTrainCameras())

        # remove gaussians that are low opacity
        gaussians = remove_gaussians_with_low_opacity(gaussians)

        # remove gaussians that are far from the mesh
        # gaussians = remove_gaussians_with_point_mesh_distance(gaussians, scene.mesh_sampled_points, dist_threshold=0.01)

        # rollout
        exp_name = dataset.source_path.split("/")[-1]
        ctrl_pts_path = f"{'/'.join(dataset.source_path.split('/')[:-1])}/experiments/{exp_name}/inference.pkl"
        print(f"ctrl_pts_path: {ctrl_pts_path}")
        with open(ctrl_pts_path, "rb") as f:
            ctrl_pts = pickle.load(f)  # (n_frames, n_ctrl_pts, 3) ndarray
        ctrl_pts = torch.tensor(ctrl_pts, dtype=torch.float32, device="cuda")
        
        # CRITICAL: Transform control points to match Gaussians coordinate system
        # Control points are in world space, Gaussians are in marker space
        # To transform control points from world to marker space, we need T_world2marker = inv(T_marker2world)
        # Load T_marker2world from ArUco calibration (same as in optimize_cma.py)
        aruco_results_path = '/users/wfu16/data/users/wfu16/datasets/2025-10-23_snapshot_julia_aruco/aruco_results/avg_marker2world.npy'
        # if os.path.exists(aruco_results_path):
        #     T_marker2world = np.load(aruco_results_path)
            # T_marker2world = np.array([[ 9.92457290e-01, -1.22580045e-01,  1.63125912e-03,  3.31059452e-01],
            #                   [ 2.70205336e-04, -1.11191912e-02, -9.99938143e-01,  1.90897759e-01],
            #                   [ 1.22590601e-01,  9.92396340e-01, -1.10022006e-02,  2.75183546e-01],
            #                   [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
            # print(f"Loaded T_marker2world from ArUco calibration: {aruco_results_path}")
        # else:
        #     # Fallback to hardcoded value if file doesn't exist
        #     T_marker2world = np.array([[ 9.92457290e-01, -1.22580045e-01,  1.63125912e-03,  3.31059452e-01],
        #                       [ 2.70205336e-04, -1.11191912e-02, -9.99938143e-01,  1.90897759e-01],
        #                       [ 1.22590601e-01,  9.92396340e-01, -1.10022006e-02,  2.75183546e-01],
        #                       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
        #     print(f"Using hardcoded T_marker2world (ArUco calibration file not found)")
        T_marker2world = np.array([[ 9.92500579e-01, -1.22225711e-01,  1.86443478e-03,  1.36186366e-01],
                              [ 5.43975403e-04, -1.08359291e-02, -9.99941142e-01, -1.88119571e-02],
                              [ 1.22238720e-01,  9.92443176e-01, -1.06881781e-02,  7.19721945e-02],
                              [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
        # T_marker2world = np.array([[ 9.92457290e-01, -1.22580045e-01,  1.63125912e-03,  3.31059452e-01],
        #                       [ 2.70205336e-04, -1.11191912e-02, -9.99938143e-01,  1.90897759e-01],
        #                       [ 1.22590601e-01,  9.92396340e-01, -1.10022006e-02,  2.75183546e-01],
        #                       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
        # Transform control points from world space to marker space: ctrl_pts_marker = T_world2marker @ ctrl_pts_world
        # where T_world2marker = inv(T_marker2world)
        # T_world2marker = np.linalg.inv(T_marker2world)
        T_marker2world = torch.tensor(T_marker2world, dtype=torch.float32, device="cuda")
        ctrl_pts_homogeneous = torch.cat([ctrl_pts, torch.ones(ctrl_pts.shape[0], ctrl_pts.shape[1], 1, device="cuda")], dim=-1)  # (n_frames, n_ctrl_pts, 4)
        ctrl_pts_transformed = torch.einsum('ij,npj->npi', T_marker2world, ctrl_pts_homogeneous)[:, :, :3]  # (n_frames, n_ctrl_pts, 3)
        ctrl_pts = ctrl_pts_transformed
        print(f"✓ Transformed control points from world to marker space using T_marker2world (inv of T_marker2world)")

        xyz_0 = gaussians.get_xyz
        print(f"xyz_0 shape: {xyz_0.shape}")
        rgb_0 = gaussians.get_features_dc.squeeze(1)
        print(f"rgb_0 shape: {rgb_0.shape}")
        quat_0 = gaussians.get_rotation
        print(f"quat_0 shape: {quat_0.shape}")
        opa_0 = gaussians.get_opacity
        print(f"opa_0 shape: {opa_0.shape}")
        scale_0 = gaussians.get_scaling
        print(f"scale_0 shape: {scale_0}")
        print(f"unique scales: {torch.unique(scale_0)}")
        print(f"gaussians._scaling shape: {gaussians._scaling.shape}")
        print(f"unique scales: {torch.unique(gaussians._scaling)}")

        # print(gaussians.get_features_dc.shape)   # (N, 1, 3)
        # print(gaussians.get_features_rest.shape) # (N, 15, 3)

        print("===== Number of steps: ", ctrl_pts.shape[0])
        print("===== Number of control points: ", ctrl_pts.shape[1])
        print("===== Number of gaussians: ", gaussians.get_xyz.shape[0])

        n_steps = ctrl_pts.shape[0]

        # rollout
        xyz, rgb, quat, opa = rollout(xyz_0, rgb_0, quat_0, opa_0, ctrl_pts, n_steps)

        # interpolate smoothly
        change_points = (
            (xyz - torch.cat([xyz[0:1], xyz[:-1]], dim=0))
            .norm(dim=-1)
            .sum(dim=-1)
            .nonzero()
            .squeeze(1)
        )
        change_points = torch.cat([torch.tensor([0]), change_points])
        for i in range(1, len(change_points)):
            start = change_points[i - 1]
            end = change_points[i]
            if end - start < 2:  # 0 or 1
                continue
            xyz[start:end] = torch.lerp(
                xyz[start][None],
                xyz[end][None],
                torch.linspace(0, 1, end - start + 1).to(xyz.device)[:, None, None],
            )[:-1]
            rgb[start:end] = torch.lerp(
                rgb[start][None],
                rgb[end][None],
                torch.linspace(0, 1, end - start + 1).to(rgb.device)[:, None, None],
            )[:-1]
            quat[start:end] = torch.lerp(
                quat[start][None],
                quat[end][None],
                torch.linspace(0, 1, end - start + 1).to(quat.device)[:, None, None],
            )[:-1]
            opa[start:end] = torch.lerp(
                opa[start][None],
                opa[end][None],
                torch.linspace(0, 1, end - start + 1).to(opa.device)[:, None, None],
            )[:-1]
        quat = torch.nn.functional.normalize(quat, dim=-1)

        gaussians_list = []
        for i in range(n_steps):
            gaussians_i = copy.deepcopy(gaussians)
            gaussians_i._xyz = xyz[i].to("cuda")
            gaussians_i._features_dc = rgb[i].unsqueeze(1).to("cuda")
            gaussians_i._rotation = quat[i].to("cuda")
            gaussians_i._opacity = gaussians_i.inverse_opacity_activation(opa[i]).to(
                "cuda"
            )
            gaussians_i._scaling = gaussians._scaling
            print(f"gaussians_i._scaling shape: {gaussians_i._scaling.shape}")
            print(f"unique scales: {torch.unique(gaussians_i._scaling)}")
            gaussians_list.append(gaussians_i)

        # Option to render both training and test cameras, or just test cameras
        # views = scene.getTestCameras()
        # print(f"test views shape: {len(views)}")
        views = scene.getTrainCameras()
        print(f"train views shape: {len(views)}")
        
        # if render_all_frames:
        #     # Combine training and test cameras to render all frames
        #     all_views = train_views + views
        #     print(f"Rendering {len(train_views)} training cameras and {len(views)} test cameras (total: {len(all_views)})")
        #     views_to_render = all_views
        # else:
        # Only render test cameras (original behavior)
        print(f"Rendering {len(views)} test cameras only")
        views_to_render = views
        
        render_set(
            output_path,
            name,
            views_to_render,
            gaussians_list,
            pipeline,
            background,
            # dataset.train_test_exp,
            False,
            separate_sh,
            # disable_sh=dataset.disable_sh,
            disable_sh=False,
            start_frame=args.start_frame,
        )


def rollout(xyz_0, rgb_0, quat_0, opa_0, ctrl_pts, n_steps, device="cuda"):
    # store results
    xyz = xyz_0.cpu()[None].repeat(n_steps, 1, 1)  # (n_steps, n_gaussians, 3)
    rgb = rgb_0.cpu()[None].repeat(n_steps, 1, 1)  # (n_steps, n_gaussians, 3)
    quat = quat_0.cpu()[None].repeat(n_steps, 1, 1)  # (n_steps, n_gaussians, 4)
    opa = opa_0.cpu()[None].repeat(n_steps, 1, 1)  # (n_steps, n_gaussians, 1)

    # init relation matrix
    init_particle_pos = ctrl_pts[0]
    relations = get_topk_indices(init_particle_pos, K=16)

    all_pos = xyz_0
    all_rot = quat_0

    for i in tqdm(range(1, n_steps), desc="Rollout progress", dynamic_ncols=True):

        prev_particle_pos = ctrl_pts[i - 1]
        cur_particle_pos = ctrl_pts[i]

        # relations = get_topk_indices(prev_particle_pos, K=16)

        # interpolate all_pos and particle_pos
        chunk_size = 20_000
        # chunk_size = 100
        num_chunks = (len(all_pos) + chunk_size - 1) // chunk_size
        for j in range(num_chunks):
            start = j * chunk_size
            end = min((j + 1) * chunk_size, len(all_pos))
            all_pos_chunk = all_pos[start:end]
            all_rot_chunk = all_rot[start:end]
            weights = knn_weights(prev_particle_pos, all_pos_chunk, K=16)
            all_pos_chunk, all_rot_chunk, _ = interpolate_motions(
                bones=prev_particle_pos,
                motions=cur_particle_pos - prev_particle_pos,
                relations=relations,
                weights=weights,
                xyz=all_pos_chunk,
                quat=all_rot_chunk,
            )
            all_pos[start:end] = all_pos_chunk
            all_rot[start:end] = all_rot_chunk

        quat[i] = all_rot.cpu()
        xyz[i] = all_pos.cpu()
        # if i > 200:
        #     print(f"delta between xyz[{i}] and xyz[{i-1}]: {xyz[i] - xyz[i-1]}")
        #     print(f"sum of delta between xyz[{i}] and xyz[0]: {(xyz[i] - xyz[0]).sum()}")
        #     print(f"delta between xzy[{i}] and xyz[0]: {xyz[i] - xyz[0]}")
        #     print(f"sum of delta between xzy[{i}] and xyz[0]: {(xyz[i] - xyz[0]).sum()}")
        rgb[i] = rgb[i - 1].clone()
        opa[i] = opa[i - 1].clone()

    return xyz, rgb, quat, opa


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--remove_gaussians", action="store_true")
    parser.add_argument("--name", default="sceneA", type=str)
    parser.add_argument("--output_dir", default="./gaussian_output_dynamic", type=str)
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=60000)
    parser.add_argument("--num_frames", type=int, default=60000)
    parser.add_argument("--render_all_frames", action="store_true", help="Render both training and test frames (default: only test frames)")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
        SPARSE_ADAM_AVAILABLE,
        args.remove_gaussians,
        args.name,
        args.output_dir,
        args.start_frame,
        args.end_frame,
        args.num_frames,
        args.render_all_frames,
    )

    with open("./rendering_finished_dynamic.txt", "a") as f:
        f.write("Rendering finished of " + args.name + "\n")
