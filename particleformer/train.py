#!/usr/bin/env python3
"""Training script for ParticleFormer.

Usage:
    # Basic training with default config
    python -m particleformer.train
    
    # Training with custom config
    python -m particleformer.train --config path/to/config.json
    
    # Training with command line overrides
    python -m particleformer.train --batch_size 8 --learning_rate 1e-4
    
    # Multi-GPU training with accelerate
    accelerate launch -m particleformer.train --config config.json
"""

import os
os.environ["PYOPENGL_PLATFORM"] = "egl"

import argparse
import json
import pickle
import shutil
from pathlib import Path
import torch

from .config import ParticleFormerConfig
from .trainer import ParticleFormerTrainer
from .data import create_dataloader

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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train ParticleFormer model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config JSON file",
    )
    
    # Data arguments
    parser.add_argument(
        "--data_root",
        type=str,
        default="/oscar/data/gdk/hli230/projects/vitac-particle/processed",
        help="Root directory containing object data folders",
    )
    parser.add_argument(
        "--object",
        type=str,
        default="001-rope",
        help="Object name to train on (if no global split.json)",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=0,
        help="Episode ID to train on (if no global split.json)",
    )
    parser.add_argument(
        "--split_json",
        type=str,
        default="split.json",
        help="Path to split.json file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="episode",
        choices=["episode", "multi-episode", "multi-object"],
        help="Training mode: 'episode', 'multi-episode', or 'multi-object' (cross-object generalization).",
    )
    parser.add_argument(
        "--cam_name",
        type=str,
        default=None,
        help="Camera name for rendering evaluation (e.g., 'brics-odroid-022_cam1'). Falls back to first camera if not specified.",
    )
    
    # Model arguments
    parser.add_argument(
        "--d_model",
        type=int,
        default=256,
        help="Transformer model dimension",
    )
    parser.add_argument(
        "--nhead",
        type=int,
        default=8,
        help="Number of attention heads",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=3,
        help="Number of transformer layers",
    )
    parser.add_argument(
        "--dim_feedforward",
        type=int,
        default=512,
        help="Feedforward dimension in transformer",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=32,
        help="Object embedding dimension",
    )
    
    # Training arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for training",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="Weight decay",
    )
    parser.add_argument(
        "--rollout_steps",
        type=int,
        default=5,
        help="Number of rollout steps for training",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    
    # Loss arguments
    parser.add_argument(
        "--loss_alpha",
        type=float,
        default=0.5,
        help="Weight for Chamfer Distance in hybrid loss (0-1)",
    )
    
    # Logging arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for checkpoints and logs (defaults to outputs/particleformer/{object}_ep{episode})",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="Log every N steps",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=10,
        help="Save checkpoint every N epochs",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable wandb logging",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="deformable_dynamics",
        help="Wandb project name",
    )
    
    # Other arguments
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision training",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    
    return parser.parse_args()


def main():
    """Main training entry point."""
    args = parse_args()
    
    # Update output_dir if not provided
    if args.output_dir is None:
        if args.mode == "multi-object":
            args.output_dir = os.path.join("outputs/particleformer", "multi_object")
        else:
            args.output_dir = os.path.join("outputs/particleformer", f"{args.object}_ep{args.episode}")
    
    # Load or create config
    if args.config is not None:
        print(f"Loading config from {args.config}")
        config = ParticleFormerConfig.load(args.config)
        # Override with command line args if provided
        for key, value in vars(args).items():
            if key != "config" and value is not None:
                if hasattr(config, key):
                    # Only override if explicitly provided (not default)
                    setattr(config, key, value)
    else:
        # Create config from command line args
        config = ParticleFormerConfig(
            data_root=args.data_root,
            split_json=args.split_json,
            object_name=args.object,
            ep_idx=args.episode,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            embedding_dim=args.embedding_dim,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            rollout_steps=args.rollout_steps,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            loss_alpha=args.loss_alpha,
            output_dir=args.output_dir,
            log_interval=args.log_interval,
            save_interval=args.save_interval,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            mixed_precision=args.mixed_precision,
            seed=args.seed,
            resume_from=args.resume_from,
            mode=args.mode,
            cam_name=args.cam_name,
        )
    
    # Set train and test episodes based on mode
    object_id_map = None
    split_config_override = None

    if config.mode == "multi-episode":
        all_candidates = [0, 1, 2, 3, 4]
        existing_episodes = []
        for ep_id in all_candidates:
            ep_path = Path(config.data_root) / config.object_name / f"episode_{ep_id}" / "final_data.pkl"
            if ep_path.exists():
                try:
                    with open(ep_path, "rb") as f:
                        data = pickle.load(f)
                    if data["object_points"].shape[0] > config.rollout_steps:
                        existing_episodes.append(ep_id)
                    else:
                        print(f"Warning: Episode {ep_id} is too short ({data['object_points'].shape[0]} frames) for rollout_steps={config.rollout_steps}. Skipping.")
                except Exception as e:
                    print(f"Warning: Could not read {ep_path}: {e}. Skipping.")
        
        if not existing_episodes:
            raise ValueError(f"No valid episodes found in range 0-4 for {config.object_name} at {config.data_root} (all existed episodes were too short or missing)")
            
        # Last existing episode (<=4) for testing
        test_ep = existing_episodes[-1]
        config.test_episodes = [test_ep]
        
        # All remaining existing episodes (< testing) for training
        config.train_episodes = [ep for ep in existing_episodes if ep < test_ep]
        
        # Fallback if only one episode exists
        if not config.train_episodes:
            print(f"Warning: Only one episode ({test_ep}) found in range 0-4. Using it for both training and testing.")
            config.train_episodes = [test_ep]
    elif config.mode == "multi-object":
        existing_train_objects = []
        existing_test_objects = []
        split_config_override = {}

        for obj_name in ALL_OBJECTS:
            obj_dir = Path(config.data_root) / obj_name
            if not obj_dir.is_dir():
                continue

            episode_ids = []
            for ep_id in [0, 1, 2, 3, 4]:
                ep_path = obj_dir / f"episode_{ep_id}" / "final_data.pkl"
                if not ep_path.exists():
                    continue
                try:
                    with open(ep_path, "rb") as f:
                        data = pickle.load(f)
                    if data["object_points"].shape[0] > config.rollout_steps:
                        episode_ids.append(ep_id)
                except Exception as e:
                    print(f"Warning: Could not read {ep_path}: {e}. Skipping.")

            if not episode_ids:
                continue

            split_config_override[obj_name] = {
                "train_episodes": episode_ids,
                "test_episodes": episode_ids,
            }

            if obj_name in TEST_OBJECTS:
                existing_test_objects.append(obj_name)
            else:
                existing_train_objects.append(obj_name)

        if not existing_train_objects:
            raise ValueError(f"No valid training objects found under {config.data_root}")
        if not existing_test_objects:
            raise ValueError(f"No valid testing objects found under {config.data_root}")

        # Keep these fields for wandb consistency and checkpoint naming.
        config.object_name = "multi-object"
        config.train_objects = existing_train_objects
        config.test_objects = existing_test_objects
        config.train_episodes = []
        config.test_episodes = []
        config.num_objects = len(existing_train_objects) + len(existing_test_objects)

        # Use a global object-id map so test objects keep unique unseen IDs.
        object_id_map = {name: idx for idx, name in enumerate(existing_train_objects + existing_test_objects)}
    else:
        config.train_episodes = [args.episode]
        config.test_episodes = [args.episode]

    print("=" * 60)
    print("ParticleFormer Training")
    print("=" * 60)
    print(f"Config: {json.dumps(config.__dict__, indent=2, default=str)}")
    print("=" * 60)
    
    # Create dataloaders
    print("Creating dataloaders...")
    
    # In multi-episode and multi-object mode, use the full episode for validation rollouts.
    val_split = "all" if config.mode in {"multi-episode", "multi-object"} else "test"
    
    # Make split_json path relative to workspace if not absolute
    split_path = Path(args.split_json)
    if not split_path.is_absolute():
        # Try relative to current directory first
        if not split_path.exists():
            # Try relative to PhysTwin directory
            phystwin_dir = Path(__file__).parent.parent
            split_path = phystwin_dir / args.split_json
    
    if config.mode == "multi-object":
        train_split_config = {k: v for k, v in split_config_override.items() if k in set(config.train_objects)}
        test_split_config = {k: v for k, v in split_config_override.items() if k in set(config.test_objects)}

        train_dataloader = create_dataloader(
            data_root=config.data_root,
            split_path=str(split_path),
            batch_size=config.batch_size,
            rollout_steps=config.rollout_steps,
            num_workers=args.num_workers,
            shuffle=True,
            split="train",
            split_config_override=train_split_config,
            object_id_map=object_id_map,
        )

        val_dataloader = create_dataloader(
            data_root=config.data_root,
            split_path=str(split_path),
            batch_size=config.batch_size,
            rollout_steps=config.rollout_steps,
            num_workers=args.num_workers,
            shuffle=False,
            split=val_split,
            split_config_override=test_split_config,
            object_id_map=object_id_map,
        )
    else:
        train_dataloader = create_dataloader(
            data_root=config.data_root,
            split_path=str(split_path),
            batch_size=config.batch_size,
            rollout_steps=config.rollout_steps,
            num_workers=args.num_workers,
            shuffle=True,
            split="train",
            object_name=config.object_name,
            episode_ids=config.train_episodes,
        )
        
        val_dataloader = create_dataloader(
            data_root=config.data_root,
            split_path=str(split_path),
            batch_size=config.batch_size,
            rollout_steps=config.rollout_steps,
            num_workers=args.num_workers,
            shuffle=False,
            split=val_split,
            object_name=config.object_name,
            episode_ids=config.test_episodes,
        )
    
    print(f"Training samples: {len(train_dataloader.dataset)}")
    print(f"Validation samples: {len(val_dataloader.dataset)}")
    print(f"Batches per epoch: {len(train_dataloader)}")
    
    # Create trainer
    print("Creating trainer...")
    trainer = ParticleFormerTrainer(config)
    
    # Start training
    print("Starting training...")
    trainer.train(train_dataloader, val_dataloader)
    
    # Cleanup output directory after training
    if trainer.accelerator.is_main_process:
        # Save final checkpoint with custom name to data_root/object
        if config.mode == "multi-object":
            final_ckpt_path = Path(config.data_root) / "particleformer_multi_object.ckpt"
        else:
            ep_suffix = "_".join(map(str, config.train_episodes))
            final_ckpt_name = f"train_ep_{ep_suffix}.ckpt"
            final_ckpt_path = Path(config.data_root) / config.object_name / final_ckpt_name
        
        print(f"Saving final checkpoint to {final_ckpt_path}...")
        final_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model state dict
        unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)
        torch.save(unwrapped_model.state_dict(), final_ckpt_path)
        
        if os.path.exists(config.output_dir):
            print(f"Cleaning up output directory: {config.output_dir}")
            shutil.rmtree(config.output_dir)


if __name__ == "__main__":
    main()
