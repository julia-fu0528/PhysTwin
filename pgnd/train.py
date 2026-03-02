#!/usr/bin/env python3
"""Training script for PGND (Particle-Grid Neural Dynamics).

Usage:
    # Basic training with default config
    pixi run python -m pgnd.train train.name=my_experiment
    
    # Training with custom config overrides
    pixi run python -m pgnd.train train.name=my_experiment train.batch_size=16
"""

from pathlib import Path
import random
import time
import os
from typing import Dict, List
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm, trange
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
from PIL import Image
import warp as wp
import torch
import torch.backends.cudnn
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
import kornia

from pgnd.sim import Friction, CacheDiffSimWithFrictionBatch, StaticsBatch, CollidersBatch
from pgnd.models import PGNDModel
from pgnd.data import RealTeleopBatchDataset
from pgnd.utils import Logger, get_root, mkdir

# Import metrics for evaluation
import pickle
import json
from particleformer.metrics import ChamferMetric, TrackMetric, RenderMetric


root: Path = Path(__file__).parent.resolve()

ALL_OBJECTS = [
    "002-rope-silk", "003-cable", "004-rubber-band", "001-rope",
    "006-fur", "008-pink-cloth", "010-orange-cloth", "011-green-cloth",
    "012-hat-cloth", "013-glove-cloth", "016-shirt-cloth",
    "015-airbag-cloth", "017-chessboard-cloth", "018-trashbag-cloth",
    "019-trashbag-plastic-cloth", "021-bag-cloth", "022-handkerchief",
    "024-glass-cleaner-cloth", "023-cleaning-cloth",
    "025-bag-small-cloth", "027-umbrella-bag-cloth", "026-sock-cloth",
    "030-foam-flat-cloth", "029-foam-cloth", "038-mat-cloth",
    "040-paper-cloth", "043-dog", "045-cat", "046-sponge",
    "048-butter-sponge", "059-shoe", "062-banana", "063-flower",
    "068-nylon-rope", "082-curtain-cloth", "088-snake", "090-sloth",
    "092-squirrel", "096-octopus", "100-puppet", "095-watermelon",
    "103-ice-pack-cloth", "109-pouch-cloth", "110-shower-cap-cloth",
    "113-collar", "115-cotton-gauze-cloth", "120-bread-plush",
    "118-envelope-cloth", "117-bubble-wrap-cloth",
    "121-croissant-plush", "125-rabbit", "135-makeup-sponge",
    "147-baking-mold", "148-crepe-paper-cloth",
    "156-mesh-produce-bag-cloth", "150-shredded-packing-paper-cloth",
    "157-sack-cloth", "159-purse", "163-bear", "164-sheep",
]
TEST_OBJECTS = {"003-cable", "045-cat", "059-shoe", "117-bubble-wrap-cloth", "159-purse"}
TRAIN_OBJECTS = [obj for obj in ALL_OBJECTS if obj not in TEST_OBJECTS]


def dataloader_wrapper(dataloader, name):
    """Infinite dataloader wrapper."""
    cnt = 0
    while True:
        cnt += 1
        for data in dataloader:
            yield data


def transform_gripper_points(cfg, gripper_points, gripper):
    """Transform gripper points to world coordinates."""
    dx = cfg.sim.num_grids[-1]

    gripper_xyz = gripper[:, :, :, :3]  # (bsz, num_steps, num_grippers, 3)
    gripper_v = gripper[:, :, :, 3:6]  # (bsz, num_steps, num_grippers, 3)
    gripper_quat = gripper[:, :, :, 6:10]  # (bsz, num_steps, num_grippers, 4)
    num_steps = gripper_xyz.shape[1]
    num_grippers = gripper_xyz.shape[2]
    gripper_mat = kornia.geometry.conversions.quaternion_to_rotation_matrix(gripper_quat)
    gripper_points = gripper_points[:, None, None].repeat(1, num_steps, num_grippers, 1, 1)
    gripper_x = gripper_points @ gripper_mat + gripper_xyz[:, :, :, None]
    bsz = gripper_x.shape[0]
    num_points = gripper_x.shape[3]

    gripper_quat_vel = gripper[:, :, :, 10:13]
    gripper_angular_vel = torch.linalg.norm(gripper_quat_vel, dim=-1, keepdims=True)
    gripper_quat_axis = gripper_quat_vel / (gripper_angular_vel + 1e-10)

    gripper_v_expand = gripper_v[:, :, :, None].repeat(1, 1, 1, num_points, 1)
    gripper_points_from_axis = gripper_x - gripper_xyz[:, :, :, None]
    grid_from_gripper_axis = gripper_points_from_axis - \
        (gripper_quat_axis[:, :, :, None] * gripper_points_from_axis).sum(dim=-1, keepdims=True) * gripper_quat_axis[:, :, :, None]
    gripper_v_expand = torch.cross(gripper_quat_vel[:, :, :, None], grid_from_gripper_axis, dim=-1) + gripper_v_expand
    gripper_v = gripper_v_expand.reshape(bsz, num_steps, num_grippers * num_points, 3)
    gripper_x = gripper_x.reshape(bsz, num_steps, num_grippers * num_points, 3)

    gripper_x_mask = (gripper_x[:, :, :, 0] > dx * (cfg.model.clip_bound + 0.5)) \
                   & (gripper_x[:, :, :, 0] < 1 - (dx * (cfg.model.clip_bound + 0.5))) \
                   & (gripper_x[:, :, :, 1] > dx * (cfg.model.clip_bound + 0.5)) \
                   & (gripper_x[:, :, :, 1] < 1 - (dx * (cfg.model.clip_bound + 0.5))) \
                   & (gripper_x[:, :, :, 2] > dx * (cfg.model.clip_bound + 0.5)) \
                   & (gripper_x[:, :, :, 2] < 1 - (dx * (cfg.model.clip_bound + 0.5)))

    return gripper_x, gripper_v, gripper_x_mask


class Trainer:
    """PGND Trainer class.
    
    Handles training, evaluation, and logging for PGND model.
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        print(OmegaConf.to_yaml(cfg, resolve=True))

        wp.init()
        wp.ScopedTimer.enabled = False
        wp.set_module_options({'fast_math': False})
        wp.config.verify_autograd_array_access = True

        gpus = [int(gpu) for gpu in cfg.gpus]
        wp_devices = [wp.get_device(f'cuda:{gpu}') for gpu in gpus]
        torch_devices = [torch.device(f'cuda:{gpu}') for gpu in gpus]
        device_count = len(torch_devices)
    
        assert device_count == 1
        self.wp_device = wp_devices[0]
        self.torch_device = torch_devices[0]

        seed = cfg.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        torch.autograd.set_detect_anomaly(True)
        torch.backends.cudnn.benchmark = True

        # path
        log_root: Path = root / 'log'
        exp_root: Path = log_root / cfg.train.name
        self.save = getattr(cfg.train, 'save', True)
        
        if self.save:
            mkdir(exp_root, overwrite=cfg.overwrite, resume=cfg.resume)
            OmegaConf.save(cfg, exp_root / 'hydra.yaml', resolve=True)

        if self.save:
            ckpt_root: Path = exp_root / 'ckpt'
            ckpt_root.mkdir(parents=True, exist_ok=True)
            self.ckpt_root = ckpt_root
        else:
            self.ckpt_root = None
        
        self.log_root = log_root
        self.exp_root = exp_root  # Store for evaluation output

        self.use_pv = cfg.train.use_pv
        self.dataset_non_overwrite = cfg.train.dataset_non_overwrite
        if not self.use_pv:
            print('not using pv rendering...')
        
        assert self.cfg.train.source_dataset_name is not None
        self.use_gs = cfg.train.use_gs
        self.mode = getattr(cfg.train, "mode", "episode")
        self.train_episode_specs: List[Dict] = []
        self.eval_episode_specs: List[Dict] = []

        # logging
        self.verbose = False
        if not cfg.debug:
            logger = Logger(cfg)
            self.logger = logger
        else:
            self.logger = None
        
        # Evaluate dt if it's a string expression like "1/30"
        self.dt = eval(cfg.sim.dt) if isinstance(cfg.sim.dt, str) else cfg.sim.dt
        
        # Initialize evaluation metrics
        eval_output_dir = exp_root / 'eval_results'
        if self.save:
            eval_output_dir.mkdir(parents=True, exist_ok=True)
        self.chamfer_metric = ChamferMetric(str(eval_output_dir))
        self.track_metric = TrackMetric(str(eval_output_dir))
        self.render_metric = RenderMetric(str(eval_output_dir), skip_render=False)
        
        # Camera name for rendering evaluation (configurable via cfg.train.cam_name)
        self.cam_name = getattr(cfg.train, 'cam_name', None)
        self.save_dataset = getattr(cfg.train, 'save_dataset', True)

    @staticmethod
    def _parse_episode_id(episode_dir: Path) -> int:
        ep_name = episode_dir.name
        if not ep_name.startswith("episode_"):
            return -1
        ep_str = ep_name.replace("episode_", "")
        try:
            return int(ep_str)
        except ValueError:
            return -1

    def _discover_object_episode_specs(self, source_root: Path, object_names: List[str]) -> List[Dict]:
        specs: List[Dict] = []
        for obj_name in object_names:
            obj_root = source_root / obj_name
            if not obj_root.is_dir():
                continue
            for ep_dir in sorted(obj_root.glob("episode_*")):
                if not ep_dir.is_dir():
                    continue
                ep_id = self._parse_episode_id(ep_dir)
                if ep_id < 0:
                    continue
                if ep_id > 4:
                    # Keep multi-object training aligned with 5-episode protocol.
                    continue
                if not (ep_dir / "final_data.pkl").exists():
                    continue
                specs.append({
                    "object_name": obj_name,
                    "episode_id": ep_id,
                    "episode_name": ep_dir.name,
                    "episode_path": ep_dir,
                })
        return specs

    def _build_multi_object_splits(self, source_root: Path) -> None:
        cfg = self.cfg
        train_objects = list(getattr(cfg.train, "train_objects", [])) or list(TRAIN_OBJECTS)
        test_objects = list(getattr(cfg.train, "test_objects", [])) or [obj for obj in ALL_OBJECTS if obj in TEST_OBJECTS]

        self.train_episode_specs = self._discover_object_episode_specs(source_root, train_objects)
        self.eval_episode_specs = self._discover_object_episode_specs(source_root, test_objects)

        available_train_objects = sorted({s["object_name"] for s in self.train_episode_specs})
        available_test_objects = sorted({s["object_name"] for s in self.eval_episode_specs})

        if len(self.train_episode_specs) == 0:
            raise ValueError(f"No training episodes found for multi-object mode at {source_root}")
        if len(self.eval_episode_specs) == 0:
            raise ValueError(f"No testing episodes found for multi-object mode at {source_root}")

        cfg.train.mode = "multi-object"
        cfg.train.object_name = "multi-object"
        cfg.train.train_objects = available_train_objects
        cfg.train.test_objects = available_test_objects
        # Keep legacy fields coherent for logging.
        cfg.train.training_start_episode = 0
        cfg.train.training_end_episode = len(self.train_episode_specs)
        cfg.train.eval_start_episode = 0
        cfg.train.eval_end_episode = len(self.eval_episode_specs)

        print(
            f"Multi-object split: {len(available_train_objects)} train objects "
            f"({len(self.train_episode_specs)} episodes), "
            f"{len(available_test_objects)} test objects "
            f"({len(self.eval_episode_specs)} episodes)"
        )
        print("Multi-object episode range: using episode IDs in [0, 4].")

    def load_train_dataset(self):
        """Load training dataset."""
        cfg = self.cfg
        if cfg.train.dataset_name is None:
            cfg.train.dataset_name = Path(cfg.train.name).parent / 'dataset'

        source_dataset_root = self.log_root / str(cfg.train.source_dataset_name)
        assert os.path.exists(source_dataset_root)

        if self.mode == "multi-object":
            self._build_multi_object_splits(source_dataset_root)
            train_episode_paths = [spec["episode_path"] for spec in self.train_episode_specs]
            # Multi-object in-memory tensors can trigger worker-side collation/storage issues.
            # Use single-process data loading for stability.
            if cfg.train.num_workers != 0:
                print(f"Multi-object mode: overriding train.num_workers from {cfg.train.num_workers} to 0 for stable collation.")
                cfg.train.num_workers = 0
        else:
            train_episode_paths = None

        dataset = RealTeleopBatchDataset(
            cfg, 
            dataset_root=self.log_root / cfg.train.dataset_name / 'state',
            source_data_root=source_dataset_root, 
            device=self.torch_device,
            num_steps=cfg.sim.num_steps_train,
            train=True,
            episode_paths=train_episode_paths,
            dataset_non_overwrite=self.dataset_non_overwrite,
            save_dataset=self.save_dataset,
        )
        self.dataset = dataset

    def init_train(self):
        """Initialize training components."""
        cfg = self.cfg

        dataloader = dataloader_wrapper(
            DataLoader(self.dataset, batch_size=cfg.train.batch_size, shuffle=True, 
                      num_workers=cfg.train.num_workers, pin_memory=True, drop_last=True),
            'dataset'
        )
        self.dataloader = dataloader

        # material model
        material_requires_grad = cfg.model.material.requires_grad
        material: nn.Module = PGNDModel(cfg)
        material.to(self.torch_device)
        material.requires_grad_(material_requires_grad)
        material.train(True)

        # friction
        friction: nn.Module = Friction(np.array([cfg.model.friction.value]))
        friction.to(self.torch_device)
        friction.requires_grad_(False)
        friction.train(False)

        if cfg.resume and cfg.train.resume_iteration > 0:
            assert (self.ckpt_root / f'{cfg.train.resume_iteration:06d}.pt').exists()
            ckpt = torch.load(self.ckpt_root / f'{cfg.train.resume_iteration:06d}.pt', map_location=self.torch_device)
            material.load_state_dict(ckpt['material'])

        elif cfg.model.ckpt:
            ckpt = torch.load(self.log_root / cfg.model.ckpt, map_location=self.torch_device)
            material.load_state_dict(ckpt['material'])

        if self.save and not (cfg.resume and cfg.train.resume_iteration > 0):
            torch.save({
                'material': material.state_dict(),
            }, self.ckpt_root / f'{cfg.train.resume_iteration:06d}.pt')

        if material_requires_grad:
            material_optimizer = torch.optim.Adam(material.parameters(), lr=cfg.train.material_lr, weight_decay=cfg.train.material_wd)
            material_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=material_optimizer, T_max=cfg.train.num_iterations)
            if cfg.train.resume_iteration > 0:
                material_lr_scheduler.last_epoch = cfg.train.resume_iteration - 1
                material_lr_scheduler.step()

        criterion = nn.MSELoss(reduction='mean')
        criterion.to(self.torch_device)

        total_step_count = 0
        if cfg.resume and cfg.train.resume_iteration > 0:
            total_step_count = cfg.train.resume_iteration * cfg.sim.num_steps_train
        losses_log = defaultdict(int)
        loss_factor_v = cfg.train.loss_factor_v
        loss_factor_x = cfg.train.loss_factor_x
    
        self.loss_factor_v = loss_factor_v
        self.loss_factor_x = loss_factor_x
        self.material_requires_grad = material_requires_grad
        self.material = material
        self.material_optimizer = material_optimizer if material_requires_grad else None
        self.material_lr_scheduler = material_lr_scheduler if material_requires_grad else None
        self.criterion = criterion
        self.total_step_count = total_step_count
        self.losses_log = losses_log
        self.friction = friction
    
    def train(self, start_iteration, end_iteration, save=True):
        """Run training loop."""
        cfg = self.cfg
        self.material.train(True)
        for iteration in trange(start_iteration, end_iteration, dynamic_ncols=True):
            if self.material_requires_grad:
                self.material_optimizer.zero_grad()

            losses = defaultdict(int)

            init_state, actions, gt_states = next(self.dataloader)
            x, v, x_his, v_his, clip_bound, enabled, episode_vec = init_state
            x = x.to(self.torch_device)
            v = v.to(self.torch_device)
            x_his = x_his.to(self.torch_device)
            v_his = v_his.to(self.torch_device)

            actions = actions.to(self.torch_device)

            gt_x, gt_v = gt_states
            gt_x = gt_x.to(self.torch_device)
            gt_v = gt_v.to(self.torch_device)

            batch_size = gt_x.shape[0]
            num_steps_total = gt_x.shape[1]
            num_particles = gt_x.shape[2]

            sim = CacheDiffSimWithFrictionBatch(cfg, num_steps_total, batch_size, self.wp_device, requires_grad=True)

            statics = StaticsBatch()
            statics.init(shape=(batch_size, num_particles), device=self.wp_device)
            statics.update_clip_bound(clip_bound)
            statics.update_enabled(enabled)
            colliders = CollidersBatch()

            num_grippers = cfg.sim.num_grippers

            colliders.init(shape=(batch_size, num_grippers), device=self.wp_device)
            if num_grippers > 0:
                assert len(actions.shape) > 2
                colliders.initialize_grippers(actions[:, 0])

            enabled = enabled.to(self.torch_device)
            enabled_mask = enabled.unsqueeze(-1).repeat(1, 1, 3)

            for step in range(num_steps_total):
                if num_grippers > 0:
                    colliders.update_grippers(actions[:, step])

                x_in = x.clone()
                if step == 0:
                    x_in_gt = x.clone()
                    v_in_gt = v.clone()
                else:
                    x_in_gt = x_in_gt + v_in_gt * self.dt * cfg.sim.interval

                pred = self.material(x, v, x_his, v_his, enabled)
                x, v = sim(statics, colliders, step, x, v, self.friction.mu.clone()[None].repeat(batch_size, 1), pred)

                if cfg.sim.gripper_forcing:
                    gripper_xyz = actions[:, step, :, :3]
                    gripper_v = actions[:, step, :, 3:6]
                    x_from_gripper = x_in[:, None] - gripper_xyz[:, :, None]
                    x_gripper_distance = torch.norm(x_from_gripper, dim=-1)
                    x_gripper_distance_mask = x_gripper_distance < cfg.model.gripper_radius
                    x_gripper_distance_mask = x_gripper_distance_mask.unsqueeze(-1).repeat(1, 1, 1, 3)
                    gripper_v_expand = gripper_v[:, :, None].repeat(1, 1, num_particles, 1)

                    gripper_closed = actions[:, step, :, -1] < 0.5
                    x_gripper_distance_mask = torch.logical_and(x_gripper_distance_mask, gripper_closed[:, :, None, None].repeat(1, 1, num_particles, 3))

                    gripper_quat_vel = actions[:, step, :, 10:13]
                    gripper_angular_vel = torch.linalg.norm(gripper_quat_vel, dim=-1, keepdims=True)
                    gripper_quat_axis = gripper_quat_vel / (gripper_angular_vel + 1e-10)

                    grid_from_gripper_axis = x_from_gripper - \
                        (gripper_quat_axis[:, :, None] * x_from_gripper).sum(dim=-1, keepdims=True) * gripper_quat_axis[:, :, None]
                    gripper_v_expand = torch.cross(gripper_quat_vel[:, :, None], grid_from_gripper_axis, dim=-1) + gripper_v_expand

                    for i in range(gripper_xyz.shape[1]):
                        x_gripper_distance_mask_single = x_gripper_distance_mask[:, i]
                        x[x_gripper_distance_mask_single] = x_in[x_gripper_distance_mask_single] + self.dt * gripper_v_expand[:, i][x_gripper_distance_mask_single]
                        v[x_gripper_distance_mask_single] = gripper_v_expand[:, i][x_gripper_distance_mask_single]

                if cfg.sim.n_history > 0:
                    x_his = torch.cat([x_his.reshape(batch_size, num_particles, -1, 3)[:, :, 1:], x[:, :, None].detach()], dim=2)
                    v_his = torch.cat([v_his.reshape(batch_size, num_particles, -1, 3)[:, :, 1:], v[:, :, None].detach()], dim=2)
                    x_his = x_his.reshape(batch_size, num_particles, -1)
                    v_his = v_his.reshape(batch_size, num_particles, -1)

                if self.verbose:
                    print('x', x.min().item(), x.max().item())
                    print('v', v.min().item(), v.max().item())

                if self.loss_factor_x > 0:
                    loss_x = self.criterion(x[enabled_mask > 0], gt_x[:, step][enabled_mask > 0]) * self.loss_factor_x
                    losses['loss_x'] += loss_x
                    self.losses_log['loss_x'] += loss_x.item()
                
                if self.loss_factor_v > 0:
                    loss_v = self.criterion(v[enabled_mask > 0], gt_v[:, step][enabled_mask > 0]) * self.loss_factor_v
                    losses['loss_v'] += loss_v
                    self.losses_log['loss_v'] += loss_v.item()

                with torch.no_grad():
                    if self.loss_factor_x > 0:
                        loss_x_trivial = self.criterion((x_in_gt + v_in_gt * self.dt * cfg.sim.interval)[enabled_mask > 0], gt_x[:, step][enabled_mask > 0]) * self.loss_factor_x
                        self.losses_log['loss_x_trivial'] += loss_x_trivial.item()

                    if self.loss_factor_v > 0:
                        loss_v_trivial = self.criterion(v_in_gt[enabled_mask > 0], gt_v[:, step][enabled_mask > 0]) * self.loss_factor_v
                        self.losses_log['loss_v_trivial'] += loss_v_trivial.item()

                    loss_x_sanity = self.criterion(x_in[enabled_mask > 0], (x - v * self.dt * cfg.sim.interval)[enabled_mask > 0]) * self.loss_factor_x
                    self.losses_log['loss_x_sanity'] += loss_x_sanity.item()

                    if step > 0:
                        loss_x_gt_sanity = self.criterion((gt_x[:, step - 1] + gt_v[:, step] * self.dt * cfg.sim.interval)[enabled_mask > 0], gt_x[:, step][enabled_mask > 0]) * self.loss_factor_x
                        self.losses_log['loss_x_gt_sanity'] += loss_x_gt_sanity.item()
                    else:
                        loss_x_gt_sanity = self.criterion((x_in + gt_v[:, step] * self.dt * cfg.sim.interval)[enabled_mask > 0], gt_x[:, step][enabled_mask > 0]) * self.loss_factor_x
                        self.losses_log['loss_x_gt_sanity'] += loss_x_gt_sanity.item()

                if not cfg.debug:
                    self.logger.add_scalar('main/iteration', iteration, step=self.total_step_count)
                    for loss_k, loss_v in losses.items():
                        self.logger.add_scalar(f'main/{loss_k}', loss_v.item(), step=self.total_step_count)
                self.total_step_count += 1

            loss = sum(losses.values())
            try:
                loss.backward()
            except Exception as e:
                print(f'loss.backward() failed: {e}')
                continue

            if self.material_requires_grad:
                material_grad_norm = clip_grad_norm_(
                    self.material.parameters(),
                    max_norm=cfg.train.material_grad_max_norm,
                    error_if_nonfinite=True)
                self.material_optimizer.step()

            if (iteration + 1) % cfg.train.iteration_log_interval == 0:
                msgs = [
                    cfg.train.name,
                    time.strftime('%H:%M:%S'),
                    'iteration {:{width}d}/{}'.format(iteration + 1, cfg.train.num_iterations, width=len(str(cfg.train.num_iterations))),
                ]

                msgs.extend([
                    'pred.norm {:.4f}'.format(pred.norm().item()),
                ])

                if self.material_requires_grad:
                    material_lr = self.material_optimizer.param_groups[0]['lr']
                    msgs.extend([
                        'e-lr {:.2e}'.format(material_lr),
                        'e-|grad| {:.4f}'.format(material_grad_norm),
                    ])

                for loss_k, loss_v in self.losses_log.items():
                    msgs.append('{} {:.8f}'.format(loss_k, loss_v / cfg.train.iteration_log_interval))
                    if not cfg.debug:
                        self.logger.add_scalar('stat/mean_{}'.format(loss_k), loss_v / cfg.train.iteration_log_interval, step=self.total_step_count)
                
                msg = ','.join(msgs)
                print('[{}]'.format(msg))
                self.losses_log = defaultdict(int)
            
            if not cfg.debug:
                self.logger.add_scalar('stat/pred_norm', pred.norm().item(), step=self.total_step_count)

            if self.material_requires_grad:
                material_lr = self.material_optimizer.param_groups[0]['lr']
                if not cfg.debug:
                    self.logger.add_scalar('stat/material_lr', material_lr, step=self.total_step_count)
                    self.logger.add_scalar('stat/material_grad_norm', material_grad_norm, step=self.total_step_count)

            if self.save and save and (iteration + 1) % cfg.train.iteration_save_interval == 0:
                torch.save({
                    'material': self.material.state_dict(),
                }, self.ckpt_root / '{:06d}.pt'.format(iteration + 1))

            if self.material_requires_grad:
                self.material_lr_scheduler.step()

    def eval_episode(self, iteration: int, episode_spec, save: bool = True):
        """Evaluate a single episode with Chamfer, Track, and Render metrics."""
        cfg = self.cfg

        log_root: Path = root / 'log'
        dataset_name = str(cfg.train.dataset_name) if cfg.train.dataset_name else ""
        if isinstance(episode_spec, dict):
            episode = episode_spec["episode_id"]
            episode_path = Path(episode_spec["episode_path"])
            episode_name_for_dataset = episode_spec["episode_name"]
            object_name = episode_spec.get("object_name", "")
            episode_label = f"{object_name}/{episode_name_for_dataset}" if object_name else episode_name_for_dataset
            episode_state_tag = f"{object_name}__{episode_name_for_dataset}" if object_name else episode_name_for_dataset
            source_data_root = episode_path.parent
        else:
            episode = int(episode_spec)
            source_dataset_root = self.log_root / str(cfg.train.source_dataset_name)
            # Get the episode path for metrics evaluation - try both naming conventions
            episode_path = source_dataset_root / f'episode_{episode}'
            episode_name_for_dataset = f'episode_{episode}'
            if not episode_path.exists():
                episode_path = source_dataset_root / f'episode_{episode:04d}'
                episode_name_for_dataset = f'episode_{episode:04d}'
            object_name = source_dataset_root.name
            episode_label = f"{object_name}/{episode_name_for_dataset}"
            episode_state_tag = f"episode_{episode:04d}"
            source_data_root = source_dataset_root

        if self.save and save:
            state_root: Path = self.exp_root / 'state'
            mkdir(state_root, overwrite=cfg.overwrite, resume=cfg.resume)
            episode_state_root = state_root / episode_state_tag
            mkdir(episode_state_root, overwrite=cfg.overwrite, resume=cfg.resume)
            OmegaConf.save(cfg, self.exp_root / 'hydra.yaml', resolve=True)

        if cfg.train.dataset_name is None:
            cfg.train.dataset_name = Path(cfg.train.name).parent / 'dataset'
        assert cfg.train.source_dataset_name is not None

        if not episode_path.exists():
            print(f"Episode {episode_label} not found at {source_data_root}")
            return {}
        
        eval_dataset = RealTeleopBatchDataset(
            cfg, 
            dataset_root=self.log_root / cfg.train.dataset_name / 'state',
            source_data_root=source_data_root,
            device=self.torch_device,
            num_steps=self.cfg.sim.num_steps,
            eval_episode_name=episode_name_for_dataset,
            episode_paths=[episode_path],
            save_dataset=False,
        )
        eval_dataloader = dataloader_wrapper(
            DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=cfg.train.num_workers, pin_memory=True),
            'dataset'
        )
        init_state, actions, gt_states, downsample_indices = next(eval_dataloader)

        x, v, x_his, v_his, clip_bound, enabled, episode_vec = init_state
        x = x.to(self.torch_device)
        v = v.to(self.torch_device)
        x_his = x_his.to(self.torch_device)
        v_his = v_his.to(self.torch_device)
    
        actions = actions.to(self.torch_device)

        gt_x, gt_v = gt_states
        gt_x = gt_x.to(self.torch_device)
        gt_v = gt_v.to(self.torch_device)
    
        batch_size = gt_x.shape[0]
        num_steps_total = gt_x.shape[1]
        num_particles = gt_x.shape[2]
        assert batch_size == 1

        sim = CacheDiffSimWithFrictionBatch(cfg, num_steps_total, batch_size, self.wp_device, requires_grad=True)

        statics = StaticsBatch()
        statics.init(shape=(batch_size, num_particles), device=self.wp_device)
        statics.update_clip_bound(clip_bound)
        statics.update_enabled(enabled)
        colliders = CollidersBatch()
        
        self.material.eval()
        self.friction.eval()

        num_grippers = cfg.sim.num_grippers

        colliders.init(shape=(batch_size, num_grippers), device=self.wp_device)
        if num_grippers > 0:
            assert len(actions.shape) > 2
            colliders.initialize_grippers(actions[:, 0])

        enabled = enabled.to(self.torch_device)
        enabled_mask = enabled.unsqueeze(-1).repeat(1, 1, 3)

        colliders_save = colliders.export()
        colliders_save = {key: torch.from_numpy(colliders_save[key])[0].to(x.device).to(x.dtype) for key in colliders_save}
        ckpt = dict(x=x[0], v=v[0], **colliders_save)

        if self.save and save:
            torch.save(ckpt, episode_state_root / f'{0:04d}.pt')

        losses = {}
        # Collect all predicted positions for metrics
        pred_positions_list = [x[0].clone()]  # Initial position
        
        with torch.no_grad():
            for step in trange(num_steps_total, desc=f"Eval {episode_label}"):
                if num_grippers > 0:
                    colliders.update_grippers(actions[:, step])
                if cfg.sim.gripper_forcing:
                    x_in = x.clone()
                else:
                    x_in = None

                pred = self.material(x, v, x_his, v_his, enabled)

                if pred.isnan().any():
                    print('pred isnan', pred.min().item(), pred.max().item())
                    break
                if pred.isinf().any():
                    print('pred isinf', pred.min().item(), pred.max().item())
                    break

                x, v = sim(statics, colliders, step, x, v, self.friction.mu[None].repeat(batch_size, 1), pred)

                if cfg.sim.gripper_forcing:
                    gripper_xyz = actions[:, step, :, :3]
                    gripper_v = actions[:, step, :, 3:6]
                    x_from_gripper = x_in[:, None] - gripper_xyz[:, :, None]
                    x_gripper_distance = torch.norm(x_from_gripper, dim=-1)
                    x_gripper_distance_mask = x_gripper_distance < cfg.model.gripper_radius
                    x_gripper_distance_mask = x_gripper_distance_mask.unsqueeze(-1).repeat(1, 1, 1, 3)
                    gripper_v_expand = gripper_v[:, :, None].repeat(1, 1, num_particles, 1)

                    gripper_closed = actions[:, step, :, -1] < 0.5
                    x_gripper_distance_mask = torch.logical_and(x_gripper_distance_mask, gripper_closed[:, :, None, None].repeat(1, 1, num_particles, 3))

                    gripper_quat_vel = actions[:, step, :, 10:13]
                    gripper_angular_vel = torch.linalg.norm(gripper_quat_vel, dim=-1, keepdims=True)
                    gripper_quat_axis = gripper_quat_vel / (gripper_angular_vel + 1e-10)

                    grid_from_gripper_axis = x_from_gripper - \
                        (gripper_quat_axis[:, :, None] * x_from_gripper).sum(dim=-1, keepdims=True) * gripper_quat_axis[:, :, None]
                    gripper_v_expand = torch.cross(gripper_quat_vel[:, :, None], grid_from_gripper_axis, dim=-1) + gripper_v_expand

                    for i in range(gripper_xyz.shape[1]):
                        x_gripper_distance_mask_single = x_gripper_distance_mask[:, i]
                        x[x_gripper_distance_mask_single] = x_in[x_gripper_distance_mask_single] + self.dt * gripper_v_expand[:, i][x_gripper_distance_mask_single]
                        v[x_gripper_distance_mask_single] = gripper_v_expand[:, i][x_gripper_distance_mask_single]
                
                if cfg.sim.n_history > 0:
                    x_his = torch.cat([x_his.reshape(batch_size, num_particles, -1, 3)[:, :, 1:], x[:, :, None].detach()], dim=2)
                    v_his = torch.cat([v_his.reshape(batch_size, num_particles, -1, 3)[:, :, 1:], v[:, :, None].detach()], dim=2)
                    x_his = x_his.reshape(batch_size, num_particles, -1)
                    v_his = v_his.reshape(batch_size, num_particles, -1)

                colliders_save = colliders.export()
                colliders_save = {key: torch.from_numpy(colliders_save[key])[0].to(x.device).to(x.dtype) for key in colliders_save}
                
                loss_x = nn.functional.mse_loss(x[enabled_mask > 0], gt_x[:, step][enabled_mask > 0])
                loss_v = nn.functional.mse_loss(v[enabled_mask > 0], gt_v[:, step][enabled_mask > 0])
                losses[step] = dict(loss_x=loss_x.item(), loss_v=loss_v.item())

                ckpt = dict(x=x[0], v=v[0], **colliders_save)
                
                # Collect predictions
                pred_positions_list.append(x[0].clone())

                if self.save and save and step % cfg.sim.skip_frame == 0:
                    torch.save(ckpt, episode_state_root / f'{int(step / cfg.sim.skip_frame):04d}.pt')

        # Stack predictions: (T, N, 3)
        pred_positions_normalized = torch.stack(pred_positions_list, dim=0)
        
        # Transform predictions from normalized space back to world coordinates
        # Load original data to get transformation parameters
        metrics = {}
        try:
            final_data_path = episode_path / 'final_data.pkl'
            if final_data_path.exists():
                with open(final_data_path, 'rb') as f:
                    original_data = pickle.load(f)
                
                gt_object_points = original_data['object_points']  # (T_gt, N_gt, 3) in world coords
                
                # Load split info to get frame alignment
                split_path = episode_path / 'split.json'
                if split_path.exists():
                    with open(split_path, 'r') as f:
                        split_data = json.load(f)
                    train_range = split_data.get('train', [0, gt_object_points.shape[0]])
                else:
                    train_range = [0, gt_object_points.shape[0]]
                
                # Transform predictions to world coordinates
                # The preprocessing applied: xyz = xyz * scale + global_translation
                # We need to invert this. We can estimate the transform by matching
                # the first frame predictions to the first frame GT.
                scale = cfg.sim.preprocess_scale
                
                # Get GT at frame 0 (aligned with train range start)
                gt_frame0 = gt_object_points[0]  # Already offset in final_data.pkl to start at train[0]
                
                # Prediction frame 0 is in normalized coordinates
                pred_frame0 = pred_positions_normalized[0].cpu().numpy()
                
                # Estimate global_translation by matching centroids
                # pred = (world * scale) + global_translation
                # So: world = (pred - global_translation) / scale
                # At frame 0, pred and gt should roughly match after inverse transform
                # Estimate: global_translation = pred_centroid - gt_centroid * scale
                pred_centroid = pred_frame0.mean(axis=0)
                gt_centroid = gt_frame0.mean(axis=0)
                global_translation = pred_centroid - gt_centroid * scale
                
                # Transform all predictions to world coordinates
                pred_positions_world = (pred_positions_normalized.cpu().numpy() - global_translation) / scale
                pred_positions_world = torch.from_numpy(pred_positions_world).float().to(self.torch_device)
                
                # Compute metrics
                print(f"\nComputing metrics for {episode_label}...")
                
                # Chamfer metric
                chamfer_results = self.chamfer_metric.evaluate(
                    str(episode_path), 
                    pred_positions_world, 
                    iteration,
                    save_results=(self.save and save)
                )
                metrics.update(chamfer_results)
                train_chamfer = chamfer_results.get('train/chamfer_error', 'N/A')
                test_chamfer = chamfer_results.get('test/chamfer_error', 'N/A')
                print(f"  Chamfer: train={train_chamfer if train_chamfer == 'N/A' else f'{train_chamfer:.6f}'}, "
                      f"test={test_chamfer if test_chamfer == 'N/A' else f'{test_chamfer:.6f}'}")
                
                # Track metric
                track_results = self.track_metric.evaluate(
                    str(episode_path), 
                    pred_positions_world, 
                    iteration,
                    save_results=(self.save and save)
                )
                metrics.update(track_results)
                train_track = track_results.get('train/track_error', 'N/A')
                test_track = track_results.get('test/track_error', 'N/A')
                print(f"  Track: train={train_track if train_track == 'N/A' else f'{train_track:.6f}'}, "
                      f"test={test_track if test_track == 'N/A' else f'{test_track:.6f}'}")
                
                # Render metric (need to include controller points for full rendering)
                # For now, just use object points
                try:
                    render_results = self.render_metric.evaluate(
                        str(episode_path), 
                        pred_positions_world, 
                        iteration,
                        cam_name=self.cam_name,
                        save_results=(self.save and save)
                    )
                    metrics.update(render_results)
                    test_psnr = render_results.get('test/psnr', 'N/A')
                    test_ssim = render_results.get('test/ssim', 'N/A')
                    test_lpips = render_results.get('test/lpips', 'N/A')
                    print(f"  Render: test PSNR={test_psnr if test_psnr == 'N/A' else f'{test_psnr:.2f}'}, "
                          f"SSIM={test_ssim if test_ssim == 'N/A' else f'{test_ssim:.4f}'}, "
                          f"LPIPS={test_lpips if test_lpips == 'N/A' else f'{test_lpips:.4f}'}")
                except Exception as e:
                    print(f"  Render metric failed: {e}")
                    import traceback
                    traceback.print_exc()
                    
            else:
                print(f"Warning: final_data.pkl not found at {final_data_path}, skipping metrics")
                
        except Exception as e:
            print(f"Error computing metrics for {episode_label}: {e}")
            import traceback
            traceback.print_exc()

        if self.save and save:
            for loss_k in losses[0].keys():
                plt.figure(figsize=(10, 5))
                loss_list = [losses[step][loss_k] for step in losses]
                plt.plot(loss_list)
                plt.title(loss_k)
                plt.grid()
                plt.savefig(state_root / f'{episode_state_tag}_{loss_k}.png', dpi=300)
                plt.close()

        return metrics

    def eval(self, eval_iteration: int, save: bool = True):
        """Evaluate over all episodes and log aggregated metrics to wandb."""
        cfg = self.cfg

        metrics_list = []
        if self.mode == "multi-object":
            eval_targets = self.eval_episode_specs
            if not save:
                eval_targets = eval_targets[:1]
        else:
            start_episode = cfg.train.eval_start_episode
            end_episode = cfg.train.eval_end_episode if save else min(cfg.train.eval_start_episode + 1, cfg.train.eval_end_episode)
            eval_targets = list(range(start_episode, end_episode))

        for episode in eval_targets:
            try:
                metrics = self.eval_episode(eval_iteration, episode, save=save)
                if metrics:
                    metrics_list.append(metrics)
            except (RuntimeError, ValueError, FileNotFoundError) as e:
                if "non-empty TensorList" in str(e) or "episode" in str(e).lower() or "reshape" in str(e).lower():
                    print(f"Skipping episode {episode}: {e}")
                    continue
                raise

        # Aggregate metrics across episodes
        if metrics_list and self.logger is not None:
            aggregated_metrics = {}
            
            # Define all metric keys we want to aggregate
            metric_keys = [
                'train/chamfer_error', 'test/chamfer_error',
                'train/chamfer_frame_num', 'test/chamfer_frame_num',
                'train/track_error', 'test/track_error',
                'train/psnr', 'test/psnr',
                'train/ssim', 'test/ssim',
                'train/lpips', 'test/lpips'
            ]
            
            for key in metric_keys:
                values = [m.get(key) for m in metrics_list if m.get(key) is not None]
                if values:
                    # Handle potential nan values
                    valid_values = [v for v in values if not (isinstance(v, float) and np.isnan(v))]
                    if valid_values:
                        # Use the key directly without 'eval/' prefix
                        aggregated_metrics[key] = np.mean(valid_values)
            
            # Log aggregated metrics to wandb
            print(f"\n{'='*60}")
            print(f"Aggregated Evaluation Metrics (iter {eval_iteration}):")
            print(f"{'='*60}")
            
            for key, value in aggregated_metrics.items():
                print(f"  {key}: {value:.6f}")
                self.logger.add_scalar(key, value, step=self.total_step_count)
            
            # Log comparison video if available (from the first episode that has it)
            for m in metrics_list:
                video_path = m.get('test/comparison_video')
                if video_path and os.path.exists(video_path):
                    print(f"  Logging comparison video: {video_path}")
                    self.logger.add_video('test/comparison_video', video_path, step=self.total_step_count)
                    break
            
            print(f"{'='*60}\n")

    def test_cuda_mem(self):
        """Test CUDA memory usage."""
        self.init_train()
        self.train(0, 10, save=False)
        self.eval(10, save=False)


@hydra.main(version_base='1.2', config_path='cfg', config_name='default')
def main(cfg: DictConfig):
    """Main training entry point."""
    trainer = Trainer(cfg)
    trainer.load_train_dataset()
    trainer.test_cuda_mem()
    trainer.init_train()
    for iteration in range(cfg.train.resume_iteration, cfg.train.num_iterations, cfg.train.iteration_eval_interval):
        start_iteration = iteration
        end_iteration = min(iteration + cfg.train.iteration_eval_interval, cfg.train.num_iterations)
        trainer.train(start_iteration, end_iteration, save=True)
        trainer.eval(end_iteration, save=True)

    # Save final checkpoint to source dataset directory (matching ParticleFormer convention)
    source_dataset_name = str(cfg.train.source_dataset_name)
    source_path = Path(source_dataset_name)
    if getattr(cfg.train, "mode", "episode") == "multi-object":
        final_ckpt_path = source_path / "pgnd_multi_object.ckpt"
    else:
        start_ep = cfg.train.training_start_episode
        end_ep = cfg.train.training_end_episode
        # Discover which episodes actually exist in the training range
        train_ep_indices = []
        for ep_i in range(start_ep, end_ep):
            if (source_path / f'episode_{ep_i}').exists():
                train_ep_indices.append(str(ep_i))
        if not train_ep_indices:
            train_ep_indices = [str(start_ep)]
        ep_suffix = "_".join(train_ep_indices)
        final_ckpt_path = source_path / f'pgnd_ep_{ep_suffix}.ckpt'
    final_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'material': trainer.material.state_dict()}, final_ckpt_path)
    print(f'Saved final checkpoint to {final_ckpt_path}')


if __name__ == '__main__':
    main()
