"""ParticleFormer: Transformer-based 3D Point Cloud World Model.

This implements the core architecture from the paper:
1. State Projector: Encodes [position, material_embedding, action] -> latent z
2. Transformer Encoder: Multi-head self-attention for particle interactions
3. State Predictor: Predicts displacement delta_x, final position = x + delta_x
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any

from .embeddings import ObjectEmbedding, CombinedEmbedding


class StateProjector(nn.Module):
    """Projects particle state features to latent space.
    
    Input features are: [position (3), material_embedding (d_embed), action (3)]
    
    Args:
        input_dim: Total input dimension.
        d_model: Output latent dimension.
        dropout: Dropout probability.
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project input features to latent space.
        
        Args:
            x: Input tensor of shape (batch, num_particles, input_dim)
        
        Returns:
            Latent tensor of shape (batch, num_particles, d_model)
        """
        return self.projector(x)


class StatePredictor(nn.Module):
    """Predicts particle displacement from latent representation.
    
    Args:
        d_model: Input latent dimension.
        output_dim: Output dimension (typically 3 for xyz displacement).
        dropout: Dropout probability.
    """
    
    def __init__(
        self,
        d_model: int = 256,
        output_dim: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim),
        )
        
        # Initialize last layer to zero for identity dynamics at start
        nn.init.zeros_(self.predictor[-1].weight)
        nn.init.zeros_(self.predictor[-1].bias)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Predict displacement from latent representation.
        
        Args:
            z: Latent tensor of shape (batch, num_particles, d_model)
        
        Returns:
            Displacement tensor of shape (batch, num_particles, output_dim)
        """
        return self.predictor(z)


class ParticleFormer(nn.Module):
    """ParticleFormer: Transformer-based 3D Point Cloud World Model.
    
    This model predicts the next state of particles given current state and actions.
    It uses a Transformer encoder to model particle interactions through self-attention.
    
    Args:
        d_model: Latent dimension for transformer.
        nhead: Number of attention heads.
        num_layers: Number of transformer encoder layers.
        dim_feedforward: Dimension of feedforward network in transformer.
        dropout: Dropout probability.
        embedding_dim: Dimension of object embeddings.
        num_objects: Maximum number of object types.
        position_dim: Dimension of position (default 3 for xyz).
        action_dim: Dimension of action (default 3 for velocity xyz).
    """
    
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        embedding_dim: int = 32,
        num_objects: int = 10,
        position_dim: int = 3,
        action_dim: int = 3,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.position_dim = position_dim
        self.action_dim = action_dim
        
        # Combined embedding for object type and particle type (object vs controller)
        self.embedding = CombinedEmbedding(
            num_objects=num_objects,
            embedding_dim=embedding_dim,
            combine_method="add",
        )
        
        # Input dimension: position + embedding + action
        input_dim = position_dim + self.embedding.output_dim + action_dim
        
        # State Projector: maps input features to latent space
        self.state_projector = StateProjector(
            input_dim=input_dim,
            d_model=d_model,
            dropout=dropout,
        )
        
        # Transformer Encoder: models particle interactions
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,  # Pre-norm for better training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )
        
        # State Predictor: predicts displacement
        self.state_predictor = StatePredictor(
            d_model=d_model,
            output_dim=position_dim,
            dropout=dropout,
        )
    
    def forward(
        self,
        positions: torch.Tensor,
        actions: torch.Tensor,
        object_ids: torch.Tensor,
        is_controller: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass to predict next positions.
        
        Args:
            positions: Current particle positions (batch, num_particles, 3)
            actions: Particle actions/velocities (batch, num_particles, 3)
            object_ids: Object type IDs (batch, num_particles) or (batch,)
            is_controller: Boolean mask for controller particles (batch, num_particles)
            attention_mask: Optional attention mask (batch, num_particles)
        
        Returns:
            Tuple of:
                - next_positions: Predicted next positions (batch, num_particles, 3)
                - displacement: Predicted displacement (batch, num_particles, 3)
        """
        batch_size, num_particles, _ = positions.shape
        
        # Handle object_ids that are scalar per batch
        if object_ids.dim() == 1:
            # Expand to (batch, num_particles)
            object_ids = object_ids.unsqueeze(1).expand(-1, num_particles)
            
        # 1. Coordinate Normalization (Zero-Mean, Unit-Std locally)
        # Helps the Transformer focus on relative structure
        pos_mean = positions.mean(dim=1, keepdim=True)
        pos_std = positions.std(dim=1, keepdim=True).clamp(min=1e-6)
        positions_norm = (positions - pos_mean) / pos_std
        
        # 2. Scale actions to match normalized position magnitude
        # Since positions are std=1, we scale actions accordingly
        actions_norm = actions * 10.0 
        
        # Get embeddings
        embeddings = self.embedding(object_ids, is_controller)  # (batch, num_particles, embed_dim)
        
        # Concatenate input features: [position_norm, embedding, action_norm]
        input_features = torch.cat([positions_norm, embeddings, actions_norm], dim=-1)
        
        # Project to latent space
        z = self.state_projector(input_features)  # (batch, num_particles, d_model)
        
        # Apply transformer encoder
        if attention_mask is not None:
            src_key_padding_mask = ~attention_mask
        else:
            src_key_padding_mask = None
        
        z = self.transformer(z, src_key_padding_mask=src_key_padding_mask)
        
        # Predict displacement
        displacement = self.state_predictor(z)  # (batch, num_particles, 3)
        
        # Compute next positions: x_{t+1} = x_t + displacement
        next_positions = positions + displacement

        
        return next_positions, displacement
    
    def rollout(
        self,
        initial_positions: torch.Tensor,
        actions_sequence: torch.Tensor,
        object_ids: torch.Tensor,
        is_controller: torch.Tensor,
        controller_positions_sequence: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Autoregressive rollout for multiple steps.
        
        Args:
            initial_positions: Initial particle positions (batch, num_particles, 3)
            actions_sequence: Actions for each step (batch, num_steps, num_particles, 3)
            object_ids: Object type IDs (batch, num_particles) or (batch,)
            is_controller: Boolean mask for controller particles (batch, num_particles)
            controller_positions_sequence: Ground truth controller positions for each step
                                          (batch, num_steps, num_controller, 3). If provided,
                                          controller particles use these positions.
            num_steps: Number of rollout steps (inferred from actions_sequence if not given)
        
        Returns:
            Tuple of:
                - positions_sequence: Predicted positions (batch, num_steps, num_particles, 3)
                - displacements_sequence: Predicted displacements (batch, num_steps, num_particles, 3)
        """
        if num_steps is None:
            num_steps = actions_sequence.shape[1]
        
        batch_size, num_particles, _ = initial_positions.shape
        device = initial_positions.device
        
        positions_list = []
        displacements_list = []
        
        current_positions = initial_positions
        
        for t in range(num_steps):
            actions = actions_sequence[:, t]  # (batch, num_particles, 3)
            
            # Forward pass
            next_positions, displacement = self.forward(
                positions=current_positions,
                actions=actions,
                object_ids=object_ids,
                is_controller=is_controller,
            )
            
            # If ground truth controller positions are provided, use them
            if controller_positions_sequence is not None:
                # Replace controller particle positions with ground truth
                controller_mask = is_controller.unsqueeze(-1).expand_as(next_positions)
                gt_controller = controller_positions_sequence[:, t]  # (batch, num_controller, 3)
                
                # Need to scatter the controller positions to the right indices
                # This assumes is_controller is aligned with controller_positions
                next_positions = torch.where(controller_mask, gt_controller, next_positions)
            
            positions_list.append(next_positions)
            displacements_list.append(displacement)
            
            # Update for next step
            current_positions = next_positions
        
        positions_sequence = torch.stack(positions_list, dim=1)
        displacements_sequence = torch.stack(displacements_list, dim=1)
        
        return positions_sequence, displacements_sequence
    
    def get_attention_weights(
        self,
        positions: torch.Tensor,
        actions: torch.Tensor,
        object_ids: torch.Tensor,
        is_controller: torch.Tensor,
    ) -> torch.Tensor:
        """Get attention weights for visualization.
        
        This is useful for understanding learned particle interactions.
        
        Returns:
            Attention weights from the last transformer layer.
        """
        batch_size, num_particles, _ = positions.shape
        
        if object_ids.dim() == 1:
            object_ids = object_ids.unsqueeze(1).expand(-1, num_particles)
        
        embeddings = self.embedding(object_ids, is_controller)
        input_features = torch.cat([positions, embeddings, actions], dim=-1)
        z = self.state_projector(input_features)
        
        # Get attention weights from last layer
        # Note: This requires accessing internal state, may need hook for production
        attn_weights = None
        
        def hook_fn(module, input, output):
            nonlocal attn_weights
            # MultiheadAttention returns (output, attn_weights) when need_weights=True
            # But TransformerEncoderLayer doesn't expose this directly
            pass
        
        # For now, just return None - implement with hooks if needed
        return attn_weights
    
    @classmethod
    def from_config(cls, config) -> "ParticleFormer":
        """Create model from config object.
        
        Args:
            config: ParticleFormerConfig instance.
        
        Returns:
            ParticleFormer model instance.
        """
        return cls(
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            embedding_dim=config.embedding_dim,
            num_objects=config.num_objects,
            position_dim=config.position_dim,
            action_dim=config.action_dim,
        )
