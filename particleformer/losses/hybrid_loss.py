"""Hybrid loss functions for ParticleFormer.

Implements Chamfer Distance and differentiable Hausdorff Distance as described
in the paper. The hybrid loss balances local accuracy (CD) with global structural
consistency (HD).
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


def pairwise_distances(
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Compute pairwise squared Euclidean distances between two point sets.
    
    Args:
        x: Points of shape (batch, N, D)
        y: Points of shape (batch, M, D)
    
    Returns:
        Distance matrix of shape (batch, N, M)
    """
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x @ y.T
    xx = (x ** 2).sum(dim=-1, keepdim=True)  # (batch, N, 1)
    yy = (y ** 2).sum(dim=-1, keepdim=True)  # (batch, M, 1)
    xy = torch.bmm(x, y.transpose(-2, -1))   # (batch, N, M)
    
    distances = xx + yy.transpose(-2, -1) - 2 * xy  # (batch, N, M)
    
    # Clamp to avoid numerical issues
    distances = torch.clamp(distances, min=0.0)
    
    return distances


def chamfer_distance(
    pred: torch.Tensor,
    target: torch.Tensor,
    pred_mask: Optional[torch.Tensor] = None,
    target_mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute Chamfer Distance between predicted and target point clouds.
    
    CD = (1/|P|) * sum_{p in P} min_{q in Q} ||p - q||^2
       + (1/|Q|) * sum_{q in Q} min_{p in P} ||q - p||^2
    
    Args:
        pred: Predicted points (batch, N, 3)
        target: Target points (batch, M, 3)
        pred_mask: Optional mask for valid predicted points (batch, N)
        target_mask: Optional mask for valid target points (batch, M)
        reduction: "mean", "sum", or "none"
    
    Returns:
        Chamfer distance (scalar if reduction="mean" or "sum", else (batch,))
    """
    batch_size = pred.shape[0]
    device = pred.device
    
    # Compute pairwise distances
    dist_matrix = pairwise_distances(pred, target)  # (batch, N, M)
    
    # Create working copy for pred->target direction
    dist_for_pred = dist_matrix.clone()
    
    # Apply target mask for pred->target direction
    if target_mask is not None:
        # Mask invalid target points with large distance
        mask = ~target_mask.unsqueeze(1)  # (batch, 1, M)
        dist_for_pred = dist_for_pred.masked_fill(mask, 1e10)
    
    # Nearest neighbor in target for each pred point
    min_dist_pred_to_target, _ = dist_for_pred.min(dim=-1)  # (batch, N)
    
    # Handle inf values that might occur if all targets are masked
    min_dist_pred_to_target = torch.where(
        min_dist_pred_to_target > 1e9,
        torch.zeros_like(min_dist_pred_to_target),
        min_dist_pred_to_target
    )
    
    if pred_mask is not None:
        # Zero out invalid predictions
        min_dist_pred_to_target = min_dist_pred_to_target * pred_mask.float()
        num_valid_pred = pred_mask.sum(dim=-1).float().clamp(min=1)
    else:
        num_valid_pred = torch.tensor(pred.shape[1], dtype=torch.float32, device=device)
    
    # Create working copy for target->pred direction
    dist_for_target = dist_matrix.clone()
    
    # Nearest neighbor in pred for each target point
    if pred_mask is not None:
        # Mask invalid predicted points
        mask = ~pred_mask.unsqueeze(-1)  # (batch, N, 1)
        dist_for_target = dist_for_target.masked_fill(mask, 1e10)
    
    min_dist_target_to_pred, _ = dist_for_target.min(dim=-2)  # (batch, M)
    
    # Handle inf values
    min_dist_target_to_pred = torch.where(
        min_dist_target_to_pred > 1e9,
        torch.zeros_like(min_dist_target_to_pred),
        min_dist_target_to_pred
    )
    
    if target_mask is not None:
        min_dist_target_to_pred = min_dist_target_to_pred * target_mask.float()
        num_valid_target = target_mask.sum(dim=-1).float().clamp(min=1)
    else:
        num_valid_target = torch.tensor(target.shape[1], dtype=torch.float32, device=device)
    
    # Average distances
    cd_pred = min_dist_pred_to_target.sum(dim=-1) / num_valid_pred
    cd_target = min_dist_target_to_pred.sum(dim=-1) / num_valid_target
    
    cd = cd_pred + cd_target  # (batch,)
    
    if reduction == "mean":
        return cd.mean()
    elif reduction == "sum":
        return cd.sum()
    else:
        return cd


def hausdorff_distance(
    pred: torch.Tensor,
    target: torch.Tensor,
    pred_mask: Optional[torch.Tensor] = None,
    target_mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",
    temperature: float = 0.1,
) -> torch.Tensor:
    """Compute differentiable Hausdorff Distance using soft-max approximation.
    
    The Hausdorff distance is the maximum of the minimum distances:
    HD = max(max_{p in P} min_{q in Q} ||p - q||, max_{q in Q} min_{p in P} ||q - p||)
    
    We use a soft-max approximation for differentiability:
    soft_max(x) = sum(x * softmax(x / temperature))
    
    Args:
        pred: Predicted points (batch, N, 3)
        target: Target points (batch, M, 3)
        pred_mask: Optional mask for valid predicted points (batch, N)
        target_mask: Optional mask for valid target points (batch, M)
        reduction: "mean", "sum", or "none"
        temperature: Temperature for soft-max (lower = closer to hard max)
    
    Returns:
        Hausdorff distance
    """
    batch_size = pred.shape[0]
    device = pred.device
    
    # Compute pairwise distances
    dist_matrix = pairwise_distances(pred, target)  # (batch, N, M)
    
    # Create working copy for pred direction
    dist_for_pred = dist_matrix.clone()
    
    # Apply target mask
    if target_mask is not None:
        mask = ~target_mask.unsqueeze(1)  # (batch, 1, M)
        dist_for_pred = dist_for_pred.masked_fill(mask, 1e10)
    
    # Min distance from each pred point to target
    min_dist_pred, _ = dist_for_pred.min(dim=-1)  # (batch, N)
    
    # Clamp large values
    min_dist_pred = torch.clamp(min_dist_pred, max=1e9)
    
    # Create working copy for target direction
    dist_for_target = dist_matrix.clone()
    
    # Apply pred mask
    if pred_mask is not None:
        mask = ~pred_mask.unsqueeze(-1)  # (batch, N, 1)
        dist_for_target = dist_for_target.masked_fill(mask, 1e10)
        # Mask out invalid pred points for soft-max
        min_dist_pred = min_dist_pred.masked_fill(~pred_mask, -1e10)
    
    # Min distance from each target point to pred
    min_dist_target, _ = dist_for_target.min(dim=-2)  # (batch, M)
    
    # Clamp large values
    min_dist_target = torch.clamp(min_dist_target, max=1e9)
    
    if target_mask is not None:
        min_dist_target = min_dist_target.masked_fill(~target_mask, -1e10)
    
    # Soft-max approximation of max
    def soft_max(x: torch.Tensor, temp: float) -> torch.Tensor:
        """Differentiable approximation of max using softmax."""
        # Handle very negative values (masked out)
        valid_mask = x > -1e9
        
        # Check if any valid values exist per batch
        has_valid = valid_mask.any(dim=-1)
        
        # Replace invalid with very negative value for softmax stability
        x_safe = torch.where(valid_mask, x, torch.full_like(x, -1e10))
        
        # Compute softmax weights
        weights = torch.softmax(x_safe / temp, dim=-1)
        
        # Weighted sum
        result = (x_safe * weights).sum(dim=-1)
        
        # Set to 0 for batches with no valid values
        result = torch.where(has_valid, result, torch.zeros_like(result))
        
        return result
    
    hd_pred = soft_max(min_dist_pred, temperature)     # (batch,)
    hd_target = soft_max(min_dist_target, temperature)  # (batch,)
    
    # Hausdorff is max of the two directions
    hd = torch.maximum(hd_pred, hd_target)
    
    if reduction == "mean":
        return hd.mean()
    elif reduction == "sum":
        return hd.sum()
    else:
        return hd


def hybrid_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.5,
    pred_mask: Optional[torch.Tensor] = None,
    target_mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",
    hd_temperature: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute hybrid loss combining Chamfer Distance and Hausdorff Distance.
    
    L_hybrid = alpha * L_CD + (1 - alpha) * L_HD
    
    Args:
        pred: Predicted points (batch, N, 3)
        target: Target points (batch, M, 3)
        alpha: Weight for Chamfer Distance (0 to 1)
        pred_mask: Optional mask for valid predicted points (batch, N)
        target_mask: Optional mask for valid target points (batch, M)
        reduction: "mean", "sum", or "none"
        hd_temperature: Temperature for Hausdorff soft-max
    
    Returns:
        Tuple of (total_loss, cd_loss, hd_loss)
    """
    cd = chamfer_distance(
        pred, target,
        pred_mask=pred_mask,
        target_mask=target_mask,
        reduction=reduction,
    )
    
    hd = hausdorff_distance(
        pred, target,
        pred_mask=pred_mask,
        target_mask=target_mask,
        reduction=reduction,
        temperature=hd_temperature,
    )
    
    total = alpha * cd + (1 - alpha) * hd
    
    return total, cd, hd


class HybridLoss(nn.Module):
    """Hybrid loss module for ParticleFormer training.
    
    Computes loss over multiple rollout steps, supporting:
    - Multi-step autoregressive prediction
    - Object-only loss (excluding controller particles)
    - Visibility masking
    
    Args:
        alpha: Weight for Chamfer Distance (0 to 1)
        hd_temperature: Temperature for Hausdorff soft-max
        object_only: If True, compute loss only on object particles
    """
    
    def __init__(
        self,
        alpha: float = 0.5,
        hd_temperature: float = 0.1,
        object_only: bool = True,
    ):
        super().__init__()
        self.alpha = alpha
        self.hd_temperature = hd_temperature
        self.object_only = object_only
    
    def forward(
        self,
        pred_positions: torch.Tensor,
        target_positions: torch.Tensor,
        is_controller: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        visibility_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute hybrid loss over rollout sequence.
        
        Args:
            pred_positions: Predicted positions (batch, num_steps, num_particles, 3)
            target_positions: Target positions (batch, num_steps, num_particles, 3)
            is_controller: Controller mask (batch, num_particles)
            padding_mask: Padding mask (batch, num_particles), True = valid
            visibility_mask: Optional visibility mask (batch, num_steps, num_particles)
        
        Returns:
            Tuple of (loss, metrics_dict)
        """
        batch_size, num_steps, num_particles, _ = pred_positions.shape
        
        # Build particle mask
        if self.object_only:
            # Exclude controller particles
            particle_mask = ~is_controller  # (batch, num_particles)
        else:
            particle_mask = torch.ones(batch_size, num_particles, dtype=torch.bool,
                                       device=pred_positions.device)
        
        # Combine with padding mask
        if padding_mask is not None:
            particle_mask = particle_mask & padding_mask
        
        total_loss = 0.0
        total_cd = 0.0
        total_hd = 0.0
        
        for t in range(num_steps):
            pred_t = pred_positions[:, t]    # (batch, num_particles, 3)
            target_t = target_positions[:, t]  # (batch, num_particles, 3)
            
            # Get mask for this timestep
            mask_t = particle_mask
            if visibility_mask is not None:
                mask_t = mask_t & visibility_mask[:, t]
            
            loss_t, cd_t, hd_t = hybrid_loss(
                pred_t, target_t,
                alpha=self.alpha,
                pred_mask=mask_t,
                target_mask=mask_t,
                reduction="mean",
                hd_temperature=self.hd_temperature,
            )
            
            total_loss += loss_t
            total_cd += cd_t
            total_hd += hd_t
        
        # Average over steps
        total_loss = total_loss / num_steps
        total_cd = total_cd / num_steps
        total_hd = total_hd / num_steps
        
        metrics = {
            "loss": total_loss.item(),
            "chamfer_distance": total_cd.item(),
            "hausdorff_distance": total_hd.item(),
        }
        
        return total_loss, metrics
