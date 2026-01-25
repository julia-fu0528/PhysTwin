"""Dataset for ParticleFormer training.

Loads particle trajectory data from final_data.pkl files following the
pattern used in qqtt/data/real_data.py.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class ParticleDataset(Dataset):
    """Dataset for particle dynamics training.
    
    Loads data from final_data.pkl files containing:
    - object_points: (T, N, 3) - object particle trajectories
    - controller_points: (T, M, 3) - controller/gripper particle trajectories
    - object_visibilities: (T, N) - visibility mask
    - object_motions_valid: (T, N) - motion validity mask
    
    Args:
        data_root: Root directory containing object folders.
        object_name: Name of the object folder (e.g., "001-rope").
        episode_ids: List of episode IDs to load.
        rollout_steps: Number of future steps to predict (k in paper).
        object_id: Integer ID for this object type (for embedding).
        transform: Optional transform to apply to data.
    """
    
    def __init__(
        self,
        data_root: str,
        object_name: str,
        episode_ids: List[int],
        rollout_steps: int = 5,
        object_id: int = 0,
        transform: Optional[Any] = None,
        split: str = "train",
    ):
        self.data_root = Path(data_root)
        self.object_name = object_name
        self.episode_ids = episode_ids
        self.rollout_steps = rollout_steps
        self.object_id = object_id
        self.transform = transform
        self.split = split
        
        # Load all episodes
        self.episodes = []
        self.episode_lengths = []
        self.cumulative_lengths = [0]
        
        for ep_id in episode_ids:
            episode_dir = self.data_root / object_name / f"episode_{ep_id}"
            episode_path = episode_dir / "final_data.pkl"
            split_path = episode_dir / "split.json"
            
            if episode_path.exists():
                episode_data = self._load_episode(episode_path, split_path)
                if episode_data is not None:
                    self.episodes.append(episode_data)
                    # Valid start frames: need rollout_steps future frames
                    valid_length = episode_data["num_frames"] - rollout_steps
                    self.episode_lengths.append(max(0, valid_length))
                    self.cumulative_lengths.append(
                        self.cumulative_lengths[-1] + self.episode_lengths[-1]
                    )
            else:
                print(f"Warning: Episode file not found: {episode_path}")
        
        self.total_length = self.cumulative_lengths[-1]
        
        if self.total_length == 0:
            raise ValueError(f"No valid data found for {object_name} with episodes {episode_ids} for split {split}")
        
        print(f"Loaded {len(self.episodes)} episodes for {object_name} ({split})")
        print(f"Total {split} samples: {self.total_length}")
    
    def _load_episode(self, path: Path, split_path: Path) -> Optional[Dict]:
        """Load a single episode from pickle file and apply split."""
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            
            # Extract relevant fields
            object_points = data["object_points"]  # (T, N, 3)
            controller_points = data["controller_points"]  # (T, M, 3)
            
            # Handle visibilities (optional)
            object_visibilities = data.get("object_visibilities", None)
            object_motions_valid = data.get("object_motions_valid", None)
            
            # Apply split if split.json exists
            if split_path.exists():
                with open(split_path, "r") as f:
                    split_data = json.load(f)
                
                train_range = split_data.get("train", [0, object_points.shape[0]])
                test_range = split_data.get("test", [0, 0])
                offset = train_range[0]
                
                if self.split == "train":
                    start_idx = 0
                    end_idx = train_range[1] - offset
                else:
                    start_idx = test_range[0] - offset
                    end_idx = test_range[1] - offset
                
                object_points = object_points[start_idx:end_idx]
                controller_points = controller_points[start_idx:end_idx]
                if object_visibilities is not None:
                    object_visibilities = object_visibilities[start_idx:end_idx]
                if object_motions_valid is not None:
                    object_motions_valid = object_motions_valid[start_idx:end_idx]
            
            num_frames = object_points.shape[0]
            if num_frames == 0:
                return None
                
            num_object_particles = object_points.shape[1]
            num_controller_particles = controller_points.shape[1]
            
            # Concatenate object and controller points
            # Shape: (T, N+M, 3)
            all_points = np.concatenate([object_points, controller_points], axis=1)
            
            # Create is_controller mask
            # Shape: (N+M,)
            is_controller = np.zeros(num_object_particles + num_controller_particles, dtype=bool)
            is_controller[num_object_particles:] = True
            
            # Compute actions (velocities): u = x[t] - x[t-1]
            # For object particles: u = 0 (passive, physics-driven)
            # For controller particles: u = actual velocity
            actions = np.zeros_like(all_points)
            # Controller velocities: current - previous
            actions[1:, num_object_particles:] = (
                controller_points[1:] - controller_points[:-1]
            )
            # First frame has zero action
            
            return {
                "all_points": all_points.astype(np.float32),
                "actions": actions.astype(np.float32),
                "is_controller": is_controller,
                "num_frames": num_frames,
                "num_object_particles": num_object_particles,
                "num_controller_particles": num_controller_particles,
                "object_visibilities": object_visibilities,
                "object_motions_valid": object_motions_valid,
                "episode_path": str(path.parent),  # Path to episode folder
                "object_name": self.object_name,
            }
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None
    
    def _get_episode_and_frame(self, idx: int) -> Tuple[int, int]:
        """Convert global index to (episode_idx, frame_idx)."""
        for ep_idx, (start, end) in enumerate(
            zip(self.cumulative_lengths[:-1], self.cumulative_lengths[1:])
        ):
            if start <= idx < end:
                frame_idx = idx - start
                return ep_idx, frame_idx
        raise IndexError(f"Index {idx} out of range")
    
    def __len__(self) -> int:
        return self.total_length
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a training sample.
        
        Returns:
            Dictionary containing:
                - positions: Current positions (N+M, 3)
                - actions: Actions for rollout_steps (rollout_steps, N+M, 3)
                - targets: Target positions for rollout_steps (rollout_steps, N+M, 3)
                - is_controller: Controller mask (N+M,)
                - object_id: Object type ID (scalar)
                - visibility_mask: Optional visibility mask
        """
        ep_idx, frame_idx = self._get_episode_and_frame(idx)
        episode = self.episodes[ep_idx]
        
        # Current state at time t
        positions = episode["all_points"][frame_idx]  # (N+M, 3)
        
        # Actions and targets for next k steps
        # actions[i] is the action applied at time t+i
        # targets[i] is the position at time t+i+1
        actions = episode["actions"][frame_idx + 1 : frame_idx + 1 + self.rollout_steps]
        targets = episode["all_points"][frame_idx + 1 : frame_idx + 1 + self.rollout_steps]
        
        # Convert to tensors
        sample = {
            "positions": torch.from_numpy(positions),
            "actions": torch.from_numpy(actions),
            "targets": torch.from_numpy(targets),
            "is_controller": torch.from_numpy(episode["is_controller"]),
            "object_id": torch.tensor(self.object_id, dtype=torch.long),
            "num_object_particles": episode["num_object_particles"],
            "num_controller_particles": episode["num_controller_particles"],
        }
        
        # Add visibility mask if available
        if episode["object_visibilities"] is not None:
            vis = episode["object_visibilities"][frame_idx + 1 : frame_idx + 1 + self.rollout_steps]
            sample["visibility_mask"] = torch.from_numpy(vis)
        
        # Add metadata for evaluation (string encoding handling might be needed for DataLoader)
        # Note: PyTorch DataLoader default collate fails with mixed strings/tensors unless handled.
        # We will handle this in our custom collate_fn if needed, or just let it pass if batch_size=1 during eval.
        sample["episode_path"] = episode["episode_path"]
        sample["object_name"] = episode["object_name"]
        
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample


class MultiObjectDataset(Dataset):
    """Dataset that combines multiple objects for training.
    
    Args:
        data_root: Root directory containing object folders.
        split_config: Dictionary with object names and episode IDs.
        rollout_steps: Number of future steps to predict.
        split: "train" or "test".
    """
    
    def __init__(
        self,
        data_root: str,
        split_config: Dict[str, Dict[str, List[int]]],
        rollout_steps: int = 5,
        split: str = "train",
    ):
        self.datasets = []
        self.object_names = []
        
        for obj_idx, (obj_name, episodes) in enumerate(split_config.items()):
            # For frame-level split, we might want to load the same episodes for both train and test
            # but the ParticleDataset will handle the slicing internally.
            # In the old logic, we filtered episodes by train_episodes or test_episodes.
            # Let's keep that logic but also allow passing the split down.
            episode_key = f"{split}_episodes"
            episode_ids = episodes.get(episode_key, [])
            
            # If no specific episodes for this split, try train_episodes as fallback
            if not episode_ids:
                episode_ids = episodes.get("train_episodes", [])
                
            if episode_ids:
                try:
                    dataset = ParticleDataset(
                        data_root=data_root,
                        object_name=obj_name,
                        episode_ids=episode_ids,
                        rollout_steps=rollout_steps,
                        object_id=obj_idx,
                        split=split,
                    )
                    self.datasets.append(dataset)
                    self.object_names.append(obj_name)
                except ValueError as e:
                    print(f"Skipping {obj_name}: {e}")
        
        # Build cumulative lengths for indexing
        self.cumulative_lengths = [0]
        for ds in self.datasets:
            self.cumulative_lengths.append(self.cumulative_lengths[-1] + len(ds))
        
        self.total_length = self.cumulative_lengths[-1]
        
        if self.total_length == 0:
            raise ValueError("No valid data found in any object")
        
        print(f"MultiObjectDataset: {len(self.datasets)} objects, {self.total_length} samples")
    
    def __len__(self) -> int:
        return self.total_length
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Find which dataset this index belongs to
        for ds_idx, (start, end) in enumerate(
            zip(self.cumulative_lengths[:-1], self.cumulative_lengths[1:])
        ):
            if start <= idx < end:
                local_idx = idx - start
                return self.datasets[ds_idx][local_idx]
        raise IndexError(f"Index {idx} out of range")


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Custom collate function to handle variable particle counts.
    
    Pads all samples to the maximum number of particles in the batch.
    """
    # Find max particles
    max_particles = max(b["positions"].shape[0] for b in batch)
    rollout_steps = batch[0]["actions"].shape[0]
    
    batch_size = len(batch)
    
    # Initialize padded tensors
    positions = torch.zeros(batch_size, max_particles, 3)
    actions = torch.zeros(batch_size, rollout_steps, max_particles, 3)
    targets = torch.zeros(batch_size, rollout_steps, max_particles, 3)
    is_controller = torch.zeros(batch_size, max_particles, dtype=torch.bool)
    object_ids = torch.zeros(batch_size, dtype=torch.long)
    padding_mask = torch.zeros(batch_size, max_particles, dtype=torch.bool)
    
    for i, b in enumerate(batch):
        n = b["positions"].shape[0]
        positions[i, :n] = b["positions"]
        actions[i, :, :n] = b["actions"]
        targets[i, :, :n] = b["targets"]
        is_controller[i, :n] = b["is_controller"]
        object_ids[i] = b["object_id"]
        padding_mask[i, :n] = True  # True = valid, False = padding
    
    return {
        "positions": positions,
        "actions": actions,
        "targets": targets,
        "is_controller": is_controller,
        "object_ids": object_ids,
        "object_ids": object_ids,
        "padding_mask": padding_mask,
        # Collect metadata as lists
        "episode_paths": [b["episode_path"] for b in batch],
        "object_names": [b["object_name"] for b in batch],
    }


def create_dataloader(
    data_root: str,
    split_path: str = "split.json",
    batch_size: int = 4,
    rollout_steps: int = 5,
    num_workers: int = 4,
    shuffle: bool = True,
    split: str = "train",
    object_name: Optional[str] = None,
    episode_ids: Optional[List[int]] = None,
) -> DataLoader:
    """Create a DataLoader from split configuration.
    
    Args:
        data_root: Root directory containing object folders.
        split_path: Path to split.json file.
        batch_size: Batch size.
        rollout_steps: Number of rollout steps.
        num_workers: Number of data loading workers.
        shuffle: Whether to shuffle data.
        split: "train" or "test".
        object_name: Optional object name to override split_path.
        episode_ids: Optional episode IDs to override split_path.
    
    Returns:
        DataLoader instance.
    """
    # 1. Try to load split configuration from file if it exists
    split_config = {}
    if Path(split_path).exists():
        with open(split_path, "r") as f:
            split_config = json.load(f)
    
    # 2. Determine which episodes to load
    # Priority: Explicit arguments > Episode-specific split.json > Global split.json
    
    # Check if the provided split_path is an episode-specific frame-level split
    if "train" in split_config and isinstance(split_config["train"], list) and len(split_config["train"]) == 2:
        try:
            episode_dir = Path(split_path).parent
            object_dir = episode_dir.parent
            if object_name is None:
                object_name = object_dir.name
            if episode_ids is None:
                episode_id = int(episode_dir.name.replace("episode_", ""))
                episode_ids = [episode_id]
            if not Path(data_root).exists():
                data_root = str(object_dir.parent)
            
            filtered_config = {
                object_name: {"train_episodes": episode_ids}
            }
        except (ValueError, IndexError):
            filtered_config = {"001-rope": {"train_episodes": [0]}}
    elif object_name is not None and episode_ids is not None:
        # Use explicit arguments
        filtered_config = {
            object_name: {"train_episodes": episode_ids}
        }
        # We still need to find the local split.json for frame-level slicing
        # This will be handled inside ParticleDataset
    elif split_config:
        # Use global split configuration
        if "data_root" in split_config:
            data_root = split_config["data_root"]
        
        objects_config = split_config.get("objects", {})
        episode_key = f"{split}_episodes"
        filtered_config = {}
        for obj_n, obj_config in objects_config.items():
            if episode_key in obj_config and obj_config[episode_key]:
                filtered_config[obj_n] = {episode_key: obj_config[episode_key]}
        
        if not filtered_config:
            default_obj = split_config.get("default_object")
            if default_obj and default_obj in objects_config:
                obj_config = objects_config[default_obj]
                filtered_config[default_obj] = {"train_episodes": obj_config.get("train_episodes", [])}
    else:
        # Fallback for the initial test case if no split.json is found anywhere
        if object_name is None:
            object_name = "001-rope"
        if episode_ids is None:
            episode_ids = [0]
        filtered_config = {
            object_name: {"train_episodes": episode_ids}
        }

    # Create dataset
    dataset = MultiObjectDataset(
        data_root=data_root,
        split_config=filtered_config,
        rollout_steps=rollout_steps,
        split=split,
    )

    breakpoint()
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    
    return dataloader
