"""Learnable embeddings for ParticleFormer.

Unlike the paper which uses one-hot material encoding, we use learnable
embeddings per object type for better representation learning.
"""

import torch
import torch.nn as nn
from typing import Optional


class ObjectEmbedding(nn.Module):
    """Learnable embedding for different object types.
    
    Each object (e.g., rope, cloth, sloth) gets a learnable embedding vector
    that is shared across all particles of that object. This replaces the
    one-hot material encoding from the original paper.
    
    Args:
        num_objects: Maximum number of object types to support.
        embedding_dim: Dimension of the embedding vector.
    """
    
    def __init__(
        self,
        num_objects: int = 10,
        embedding_dim: int = 32,
    ):
        super().__init__()
        self.num_objects = num_objects
        self.embedding_dim = embedding_dim
        
        # Learnable embedding table
        self.embeddings = nn.Embedding(num_objects, embedding_dim)
        
        # Initialize with small random values
        nn.init.normal_(self.embeddings.weight, mean=0.0, std=0.02)
    
    def forward(self, object_ids: torch.Tensor) -> torch.Tensor:
        """Get embeddings for given object IDs.
        
        Args:
            object_ids: Tensor of shape (batch_size,) or (batch_size, num_particles)
                       containing object IDs in range [0, num_objects).
        
        Returns:
            Embeddings of shape (..., embedding_dim)
        """
        return self.embeddings(object_ids)
    
    def get_embedding(self, object_id: int) -> torch.Tensor:
        """Get embedding for a single object ID.
        
        Args:
            object_id: Integer object ID.
        
        Returns:
            Embedding vector of shape (embedding_dim,)
        """
        return self.embeddings.weight[object_id]


class ParticleTypeEmbedding(nn.Module):
    """Embedding to distinguish between object particles and controller particles.
    
    This helps the model understand that object particles have different dynamics
    (passive, physics-driven) compared to controller particles (active, externally controlled).
    
    Args:
        embedding_dim: Dimension of the embedding vector.
    """
    
    def __init__(self, embedding_dim: int = 32):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Two types: 0 = object particle, 1 = controller particle
        self.embeddings = nn.Embedding(2, embedding_dim)
        nn.init.normal_(self.embeddings.weight, mean=0.0, std=0.02)
    
    def forward(self, is_controller: torch.Tensor) -> torch.Tensor:
        """Get embeddings based on particle type.
        
        Args:
            is_controller: Boolean tensor indicating if particle is controller.
        
        Returns:
            Embeddings of shape (..., embedding_dim)
        """
        # Convert bool to int indices
        indices = is_controller.long()
        return self.embeddings(indices)


class CombinedEmbedding(nn.Module):
    """Combined embedding that includes object type and particle type.
    
    This creates a rich embedding that encodes:
    1. Which object the particle belongs to (object embedding)
    2. Whether it's an object particle or controller particle (type embedding)
    
    Args:
        num_objects: Maximum number of object types.
        embedding_dim: Dimension for each embedding component.
        combine_method: How to combine embeddings ("add" or "concat").
    """
    
    def __init__(
        self,
        num_objects: int = 10,
        embedding_dim: int = 32,
        combine_method: str = "add",
    ):
        super().__init__()
        self.num_objects = num_objects
        self.embedding_dim = embedding_dim
        self.combine_method = combine_method
        
        self.object_embedding = ObjectEmbedding(num_objects, embedding_dim)
        self.type_embedding = ParticleTypeEmbedding(embedding_dim)
        
        if combine_method == "concat":
            self.output_dim = embedding_dim * 2
        else:
            self.output_dim = embedding_dim
    
    def forward(
        self,
        object_ids: torch.Tensor,
        is_controller: torch.Tensor,
    ) -> torch.Tensor:
        """Get combined embeddings.
        
        Args:
            object_ids: Tensor of object IDs.
            is_controller: Boolean tensor for particle type.
        
        Returns:
            Combined embeddings of shape (..., output_dim)
        """
        obj_emb = self.object_embedding(object_ids)
        type_emb = self.type_embedding(is_controller)
        
        if self.combine_method == "concat":
            return torch.cat([obj_emb, type_emb], dim=-1)
        else:
            return obj_emb + type_emb
