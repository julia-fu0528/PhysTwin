"""Dataset for Contact Prediction training.

Loads RGB frames, contact state labels, and gripper actions across
all objects and episodes. Contact state is derived from split.json:
frames within train/test ranges are in-contact (1), others are not (0).

Reuses episode-indexing patterns from ParticleFormer's dataset.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ImageNet normalization for DINOv2
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ContactEpisode:
    """Holds precomputed data for a single episode of a single object."""

    def __init__(
        self,
        object_name: str,
        episode_id: int,
        video_path: str,
        contact_labels: np.ndarray,  # (T,) binary
        gripper_actions: np.ndarray,  # (T, 3) mean gripper velocity
        num_frames: int,
    ):
        self.object_name = object_name
        self.episode_id = episode_id
        self.video_path = video_path
        self.contact_labels = contact_labels
        self.gripper_actions = gripper_actions
        self.num_frames = num_frames


class ContactDataset(Dataset):
    """Dataset for contact prediction across multiple objects and episodes.

    For each frame t, returns:
        - rgb: (3, H, W) normalized image at frame t
        - contact_state: scalar (0 or 1) at frame t
        - gripper_action: (3,) mean gripper velocity at frame t
        - target: scalar (0 or 1) contact state at frame t+1

    Args:
        data_root: Root directory containing object folders.
        object_names: List of object names to include.
        episode_ids: List of episode IDs to load.
        cam_name: Camera name for RGB frames.
        img_size: Image size for DINOv2 input.
        split: "train" or "test" — not used for filtering frames here,
               since we load all frames and label them by split.json.
    """

    def __init__(
        self,
        data_root: str,
        object_names: List[str],
        episode_ids: List[int],
        cam_name: str = "brics-odroid-022_cam1",
        img_size: int = 224,
    ):
        self.data_root = Path(data_root)
        self.cam_name = cam_name
        self.img_size = img_size

        # Image transform: resize, center crop, normalize
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        # Load all episodes
        self.episodes: List[ContactEpisode] = []
        self.episode_lengths: List[int] = []  # valid frames per episode (T-1)
        self.cumulative_lengths: List[int] = [0]

        skipped = 0
        for obj_name in object_names:
            for ep_id in episode_ids:
                episode = self._load_episode(obj_name, ep_id)
                if episode is not None:
                    self.episodes.append(episode)
                    # We need t, so valid frames are 0..T-1
                    valid_len = episode.num_frames
                    self.episode_lengths.append(max(0, valid_len))
                    self.cumulative_lengths.append(
                        self.cumulative_lengths[-1] + self.episode_lengths[-1]
                    )
                else:
                    skipped += 1

        self.total_length = self.cumulative_lengths[-1]

        if self.total_length == 0:
            raise ValueError(
                f"No valid data found for objects={object_names[:3]}... "
                f"episodes={episode_ids}"
            )

        print(f"ContactDataset: {len(self.episodes)} episodes loaded, "
              f"{skipped} skipped, {self.total_length} samples total")

        # Compute class balance for loss weighting
        self._compute_class_balance()

    def _load_episode(
        self, object_name: str, episode_id: int
    ) -> Optional[ContactEpisode]:
        """Load a single episode: split.json + final_data.pkl + video path."""
        episode_dir = self.data_root / object_name / f"episode_{episode_id}"
        split_path = episode_dir / "split.json"
        pkl_path = episode_dir / "final_data.pkl"
        video_path = episode_dir / self.cam_name / "undistorted.mp4"

        # All three files must exist
        if not split_path.exists():
            return None
        if not pkl_path.exists():
            return None
        if not video_path.exists():
            return None

        try:
            # 1. Load split.json to build contact labels
            with open(split_path, "r") as f:
                split_data = json.load(f)

            frame_len = split_data["frame_len"]
            train_range = split_data.get("train", [0, 0])
            test_range = split_data.get("test", [0, 0])

            # Build per-frame contact labels: 1 = in contact, 0 = not
            contact_labels = np.zeros(frame_len, dtype=np.float32)
            # Frames in train range are in contact
            if train_range[1] > train_range[0]:
                contact_labels[train_range[0]:train_range[1]] = 1.0
            # Frames in test range are in contact
            if test_range[1] > test_range[0]:
                contact_labels[test_range[0]:test_range[1]] = 1.0

            # 2. Load gripper actions from final_data.pkl
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)

            controller_points = data["controller_points"]  # (T_pkl, M, 3)

            # Compute mean gripper velocity: v[t] = mean(ctrl[t] - ctrl[t-1])
            # For t=0, velocity is zero
            gripper_vel = np.zeros((controller_points.shape[0], 3), dtype=np.float32)
            if controller_points.shape[0] > 1:
                diffs = controller_points[1:] - controller_points[:-1]  # (T-1, M, 3)
                gripper_vel[1:] = diffs.mean(axis=1)  # (T-1, 3)

            # 3. Verify video frame count
            cap = cv2.VideoCapture(str(video_path))
            video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            # Use minimum length across all sources
            num_frames = min(frame_len, controller_points.shape[0], video_frames)
            if num_frames < 2:
                return None

            contact_labels = contact_labels[:num_frames]
            gripper_vel = gripper_vel[:num_frames]

            return ContactEpisode(
                object_name=object_name,
                episode_id=episode_id,
                video_path=str(video_path),
                contact_labels=contact_labels,
                gripper_actions=gripper_vel,
                num_frames=num_frames,
            )

        except Exception as e:
            print(f"Warning: Error loading {object_name}/episode_{episode_id}: {e}")
            return None

    def _compute_class_balance(self):
        """Compute positive class ratio for loss weighting."""
        total_pos = 0
        total_samples = 0
        for ep in self.episodes:
            # target labels are contact_labels
            total_pos += ep.contact_labels.sum()
            total_samples += ep.num_frames
        self.pos_ratio = total_pos / max(total_samples, 1)
        self.neg_ratio = 1.0 - self.pos_ratio
        # pos_weight for BCEWithLogitsLoss: weight for positive class
        # Higher when positives are rare
        self.pos_weight = self.neg_ratio / max(self.pos_ratio, 1e-6)
        print(f"Class balance: {self.pos_ratio:.3f} positive, "
              f"{self.neg_ratio:.3f} negative, pos_weight={self.pos_weight:.3f}")

    def _get_episode_and_frame(self, idx: int) -> Tuple[int, int]:
        """Convert global index to (episode_idx, frame_idx)."""
        for ep_idx, (start, end) in enumerate(
            zip(self.cumulative_lengths[:-1], self.cumulative_lengths[1:])
        ):
            if start <= idx < end:
                frame_idx = idx - start
                return ep_idx, frame_idx
        raise IndexError(f"Index {idx} out of range [0, {self.total_length})")

    def _read_video_frame(self, video_path: str, frame_idx: int) -> np.ndarray:
        """Read a single frame from video file.

        Returns:
            RGB numpy array (H, W, 3) in uint8.
        """
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError(
                f"Failed to read frame {frame_idx} from {video_path}"
            )
        # BGR -> RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def __len__(self) -> int:
        return self.total_length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a training sample.

        Returns:
            Dictionary containing:
                - rgb: (3, img_size, img_size) normalized image at frame t
                - gripper_action: (3,) mean gripper velocity at frame t
                - target: (1,) binary contact at frame t
                - object_name: str
                - episode_id: int
                - frame_idx: int
        """
        ep_idx, frame_idx = self._get_episode_and_frame(idx)
        episode = self.episodes[ep_idx]

        # Read RGB frame at time t
        rgb = self._read_video_frame(episode.video_path, frame_idx)
        rgb = self.transform(rgb)  # (3, H, W)

        # Gripper action at time t
        gripper_action = torch.from_numpy(
            episode.gripper_actions[frame_idx].copy()
        )

        # Target: contact state at time t
        target = torch.tensor(
            [episode.contact_labels[frame_idx]], dtype=torch.float32
        )

        return {
            "rgb": rgb,
            "gripper_action": gripper_action,
            "target": target,
            "object_name": episode.object_name,
            "episode_id": episode.episode_id,
            "frame_idx": frame_idx,
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function for ContactDataset."""
    return {
        "rgb": torch.stack([b["rgb"] for b in batch]),
        "gripper_action": torch.stack([b["gripper_action"] for b in batch]),
        "target": torch.stack([b["target"] for b in batch]),
        "object_names": [b["object_name"] for b in batch],
        "episode_ids": [b["episode_id"] for b in batch],
        "frame_idxs": [b["frame_idx"] for b in batch],
    }


def create_dataloader(
    data_root: str,
    object_names: List[str],
    episode_ids: List[int],
    cam_name: str = "brics-odroid-022_cam1",
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
) -> DataLoader:
    """Create a DataLoader for contact prediction.

    Args:
        data_root: Root directory containing object folders.
        object_names: List of object names to include.
        episode_ids: List of episode IDs to load.
        cam_name: Camera name for RGB frames.
        img_size: Image size for DINOv2 input.
        batch_size: Batch size.
        num_workers: Number of data loading workers.
        shuffle: Whether to shuffle data.

    Returns:
        DataLoader instance.
    """
    dataset = ContactDataset(
        data_root=data_root,
        object_names=object_names,
        episode_ids=episode_ids,
        cam_name=cam_name,
        img_size=img_size,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    return dataloader
