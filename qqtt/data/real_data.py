import numpy as np
import torch
import pickle
from qqtt.utils import logger, visualize_pc, visualize_pc_grid, cfg
import matplotlib.pyplot as plt
import sys


class RealData:
    def __init__(self, visualize=True, save_gt=True, use_grid=None, vis_cam_indices=None):
        logger.info(f"[DATA]: loading data from {cfg.data_path}")
        self.data_path = cfg.data_path
        self.base_dir = cfg.base_dir
        # Use config values if not explicitly provided
        self.use_grid = use_grid if use_grid is not None else cfg.use_grid_gt
        self.vis_cam_indices = vis_cam_indices if vis_cam_indices is not None else cfg.vis_cam_indices
        with open(self.data_path, "rb") as f:
            data = pickle.load(f)

        object_points = data["object_points"]
        original_shape = object_points.shape
        points_flat = object_points.reshape(-1, 3)
        points_homogeneous = np.hstack((points_flat, np.ones((points_flat.shape[0], 1))))
        object_points = (cfg.T_world2marker @ points_homogeneous.T).T[:, :3].reshape(original_shape)
        # get the z range of object_points
        z_min, z_max = np.min(object_points[:, :, 2]), np.max(object_points[:, :, 2])
        # log the z range
        logger.info(f"z range: {z_min:.3f} to {z_max:.3f}")
        logger.info(f"y range: {object_points[:, :, 1].min():.3f} to {object_points[:, :, 1].max():.3f}")
        logger.info(f"x range: {object_points[:, :, 0].min():.3f} to {object_points[:, :, 0].max():.3f}")
        object_colors = data["object_colors"]
        object_visibilities = data["object_visibilities"]
        object_motions_valid = data["object_motions_valid"]
        
        controller_points = data["controller_points"]
        points_flat = controller_points.reshape(-1, 3)
        points_homogeneous = np.hstack((points_flat, np.ones((points_flat.shape[0], 1))))
        controller_points = (cfg.T_world2marker @ points_homogeneous.T).T[:, :3].reshape(controller_points.shape)
        # add a point (0, 0, 0) to the controller points
        # controller_points = np.concatenate([controller_points, np.zeros((controller_points.shape[0], 1, 3))], axis=1)
        # new_point = np.tile(np.array([[1.0, 0.0, 0.0]]), (controller_points.shape[0], 1, 1))  # shape (T, 1, 3)

        # controller_points = np.concatenate([controller_points, new_point], axis=1)
        logger.info(f"controller points z range: {controller_points[:, :, 2].min():.3f} to {controller_points[:, :, 2].max():.3f}")
        logger.info(f"controller points y range: {controller_points[:, :, 1].min():.3f} to {controller_points[:, :, 1].max():.3f}")
        logger.info(f"controller points x range: {controller_points[:, :, 0].min():.3f} to {controller_points[:, :, 0].max():.3f}")
       
        
        other_surface_points = data["surface_points"]
        points_flat = other_surface_points.reshape(-1, 3)
        points_homogeneous = np.hstack((points_flat, np.ones((points_flat.shape[0], 1))))
        other_surface_points = (cfg.T_world2marker @ points_homogeneous.T).T[:, :3].reshape(other_surface_points.shape)
        
        interior_points = data["interior_points"]
        points_flat = interior_points.reshape(-1, 3)
        points_homogeneous = np.hstack((points_flat, np.ones((points_flat.shape[0], 1))))
        interior_points = (cfg.T_world2marker @ points_homogeneous.T).T[:, :3].reshape(interior_points.shape)

        # Get the rainbow color for the object_colors
        y_min, y_max = np.min(object_points[0, :, 1]), np.max(object_points[0, :, 1])
        y_normalized = (object_points[0, :, 1] - y_min) / (y_max - y_min)
        rainbow_colors = plt.cm.rainbow(y_normalized)[:, :3]

        self.num_original_points = object_points.shape[1]
        self.num_surface_points = (
            self.num_original_points + other_surface_points.shape[0]
        )
        print(f"other_surface_points.shape: {other_surface_points.shape}")
        print(f"interior_points.shape: {interior_points.shape}")
        print(f"object_points.shape: {object_points.shape}")
        self.num_all_points = self.num_surface_points + interior_points.shape[0]

        # Concatenate the surface points and interior points
        self.structure_points = np.concatenate(
            [object_points[0], other_surface_points, interior_points], axis=0
        )
        self.structure_points = torch.tensor(
            self.structure_points, dtype=torch.float32, device=cfg.device
        )
        self.object_points = torch.tensor(
            object_points, dtype=torch.float32, device=cfg.device
        )
        # self.object_colors = torch.tensor(
        #     object_colors, dtype=torch.float32, device=cfg.device
        # )
        self.original_object_colors = torch.tensor(
            object_colors, dtype=torch.float32, device=cfg.device
        )
        # Apply the rainbow color to the object_colors
        rainbow_colors = torch.tensor(
            rainbow_colors, dtype=torch.float32, device=cfg.device
        )
        # Make the same rainbow color for each frame
        self.object_colors = rainbow_colors.repeat(self.object_points.shape[0], 1, 1)

        # # Apply the first frame color to all frames
        # first_frame_colors = torch.tensor(
        #     object_colors[0], dtype=torch.float32, device=cfg.device
        # )
        # self.object_colors = first_frame_colors.repeat(self.object_points.shape[0], 1, 1)

        self.object_visibilities = torch.tensor(
            object_visibilities, dtype=torch.bool, device=cfg.device
        )
        self.object_motions_valid = torch.tensor(
            object_motions_valid, dtype=torch.bool, device=cfg.device
        )
        self.controller_points = torch.tensor(
            controller_points, dtype=torch.float32, device=cfg.device
        )

        self.frame_len = self.object_points.shape[0]
        # Visualize/save the GT frames
        self.visualize_data(visualize=visualize, save_gt=save_gt, use_grid=self.use_grid, vis_cam_indices=self.vis_cam_indices)

    def visualize_data(self, visualize=True, save_gt=True, use_grid=False, vis_cam_indices=None):
        if visualize:
            if use_grid:
                visualize_pc_grid(
                    self.object_points,
                    self.object_colors,
                    self.controller_points,
                    self.object_visibilities,
                    self.object_motions_valid,
                    save_video=True,
                    vis_cam_indices=vis_cam_indices,
                )
            else:
                visualize_pc(
                    self.object_points,
                    self.object_colors,
                    self.controller_points,
                    self.object_visibilities,
                    self.object_motions_valid,
                    visualize=True,
                    vis_cam_idx=23,
                )
        if save_gt:
            if use_grid:
                visualize_pc_grid(
                    self.object_points,
                    self.object_colors,
                    self.controller_points,
                    self.object_visibilities,
                    self.object_motions_valid,
                    save_video=True,
                    save_path=f"{self.base_dir}/gt_grid.mp4",
                    vis_cam_indices=vis_cam_indices,
                )
            else:
                visualize_pc(
                    self.object_points,
                    self.object_colors,
                    self.controller_points,
                    self.object_visibilities,
                    self.object_motions_valid,
                    visualize=False,
                    save_video=True,
                    save_path=f"{self.base_dir}/gt.mp4",
                    vis_cam_idx=23,
                )
