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
from gs_render import (
    remove_gaussians_with_mask,
    remove_gaussians_with_low_opacity,
    remove_gaussians_with_point_mesh_distance,
)


def render_set(
    output_path,
    name,
    views,
    gaussians,
    pipeline,
    background,
    train_test_exp,
    separate_sh,
    disable_sh=False,
    start_frame=0,
):
    print(f"views shape: {len(views)}")
    render_path = os.path.join(output_path, name)
    makedirs(render_path, exist_ok=True)

    # view_indices = [0, 25, 50, 75, 100, 125]
    # view_indices = [0, 1, 23]
    view_indices = [0]
    selected_views = [views[i] for i in view_indices]

    for idx, view in enumerate(tqdm(selected_views, desc="Rendering progress")):

        view_render_path = os.path.join(render_path, f"{idx}")
        makedirs(view_render_path, exist_ok=True)

        # Render the static Gaussians (no frame loop needed since they don't change)
        if disable_sh:
            override_color = gaussians.get_features_dc.squeeze()
            results = render(
                view,
                gaussians,
                pipeline,
                background,
                override_color=override_color,
                use_trained_exp=False,
                separate_sh=separate_sh,
            )
        else:
            results = render(
                view,
                gaussians,
                pipeline,
                background,
                use_trained_exp=False,
                separate_sh=separate_sh,
            )

        rendering = results["render"]  # (3, H, W) in range [0, 1]

        # # Load original image from camera if available
        # if hasattr(view, 'image_path') and view.image_path is not None and os.path.exists(view.image_path):
        #     image_path = view.image_path.replace("000000", f"{start_frame:06d}")
        #     try:
        #         # Load original image as RGB
        #         gt_image = Image.open(image_path).convert("RGB")
        #         gt_image = gt_image.resize((view.image_width, view.image_height))
        #         gt_image = torch.from_numpy(np.array(gt_image)).float() / 255.0  # [0, 1]
        #         gt_image = gt_image.permute(2, 0, 1)  # (3, H, W)
        #         gt_image = gt_image.to(rendering.device)
        #     except Exception as e:
        #         print(f"Warning: Could not load image from {image_path}: {e}")
        #         # Fallback: use background color
        #         bg_color = background if isinstance(background, torch.Tensor) else torch.tensor(background, device=rendering.device)
        #         gt_image = bg_color.unsqueeze(1).unsqueeze(2).expand(3, rendering.shape[1], rendering.shape[2])  # (3, H, W)
        # else:
        #     # Fallback: use background color
        #     bg_color = background if isinstance(background, torch.Tensor) else torch.tensor(background, device=rendering.device)
        #     gt_image = bg_color.unsqueeze(1).unsqueeze(2).expand(3, rendering.shape[1], rendering.shape[2])  # (3, H, W)
        
        # # Resize gt_image to match rendering resolution if needed
        # if gt_image.shape[1] != rendering.shape[1] or gt_image.shape[2] != rendering.shape[2]:
        #     gt_image = F.interpolate(
        #         gt_image.unsqueeze(0),
        #         size=(rendering.shape[1], rendering.shape[2]),
        #         mode='bilinear',
        #         align_corners=False
        #     ).squeeze(0)
        
        # # Alpha blending: foreground (GS) at 0.6 alpha, background (original) at 0.4 alpha
        # alpha_gs = 0.6  # Alpha for GS rendering
        # alpha_bg = 0.4  # Alpha for background (original image)
        
        # # Create alpha channel: 0.6 where GS is visible, 0.4 elsewhere
        # gs_visible = (rendering.sum(dim=0) > 0.01).float()  # (H, W) - 1 where GS visible, 0 elsewhere
        # alpha_channel = alpha_gs * gs_visible + alpha_bg * (1 - gs_visible)  # (H, W)
        # alpha_channel = alpha_channel.unsqueeze(0)  # (1, H, W)
        
        # # Composite: rendering * 0.6 + gt_image * 0.4
        # composite_rgb = rendering * alpha_gs + gt_image * alpha_bg  # (3, H, W)
        
        # Save the rendering
        torchvision.utils.save_image(
            rendering,
            os.path.join(view_render_path, "00000.png"),
        )
        # Optionally save overlaid version
        # torchvision.utils.save_image(
        #     composite_rgb,
        #     os.path.join(view_render_path, "00000_overlaid.png"),
        # )


def render_sets(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    skip_train: bool,
    skip_test: bool,
    separate_sh: bool,
    remove_gaussians: bool = False,
    name: str = "static",
    output_dir: str = "./gaussian_output_static",
    start_frame: int = 0,
    end_frame: int = 60000,
    num_frames: int = 60000,
):
    print(f"output_dir: {output_dir}")
    with torch.no_grad():
        output_path = output_dir
        print(f"dataset.white_background: {dataset.white_background}")

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        print(f"bg_color: {bg_color}")
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, start_frame=start_frame, end_frame=end_frame, num_frames=num_frames)

        # remove gaussians that are outside the mask
        if remove_gaussians:
            gaussians = remove_gaussians_with_mask(gaussians, scene.getTrainCameras())

        # remove gaussians that are low opacity
        gaussians = remove_gaussians_with_low_opacity(gaussians)

        # remove gaussians that are far from the mesh
        # gaussians = remove_gaussians_with_point_mesh_distance(gaussians, scene.mesh_sampled_points, dist_threshold=0.01)

        print(f"Number of gaussians: {gaussians.get_xyz.shape[0]}")

        # Get views to render
        views = scene.getTrainCameras()
        print(f"train views shape: {len(views)}")
        
        print(f"Rendering {len(views)} cameras (static Gaussians, no deformation)")
        views_to_render = views
        
        render_set(
            output_path,
            name,
            views_to_render,
            gaussians,  # Single static gaussians object, not a list
            pipeline,
            background,
            False,
            separate_sh,
            disable_sh=False,
            start_frame=start_frame,
        )


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Static Gaussian rendering (no deformation)")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--remove_gaussians", action="store_true")
    parser.add_argument("--name", default="static", type=str)
    parser.add_argument("--output_dir", default="./gaussian_output_static", type=str)
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=60000)
    parser.add_argument("--num_frames", type=int, default=60000)
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
    )

    with open("./rendering_finished_static.txt", "a") as f:
        f.write("Rendering finished of " + args.name + "\n")

