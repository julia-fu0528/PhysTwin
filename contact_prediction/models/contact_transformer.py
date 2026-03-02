"""Contact Transformer: DINOv2 + Transformer for contact prediction.

Architecture:
1. DINOv2 ViT-B/14 (frozen) extracts visual tokens from RGB image
2. Visual tokens projected to d_model dimensions
3. Contact state embedded via nn.Embedding
4. Gripper action projected via Linear
5. Learnable [PRED] token prepended to sequence
6. TransformerEncoder processes all tokens
7. [PRED] token output → MLP → binary logit for next-frame contact
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class ContactTransformer(nn.Module):
    """Transformer-based contact prediction model.

    Args:
        config: ContactPredictionConfig instance.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_model = config.d_model

        # 1. Visual encoder: DINOv2
        self.visual_encoder = torch.hub.load(
            "facebookresearch/dinov2", config.visual_encoder
        )
        # Get DINOv2 output dimension
        self.visual_dim = self.visual_encoder.embed_dim  # 768 for vitb14

        if config.freeze_visual_encoder:
            for param in self.visual_encoder.parameters():
                param.requires_grad = False
            self.visual_encoder.eval()

        # 2. Visual token projector
        self.visual_proj = nn.Sequential(
            nn.Linear(self.visual_dim, config.d_model),
            nn.LayerNorm(config.d_model),
        )

        # Removed contact state embedding

        # 4. Gripper action projector
        self.action_proj = nn.Sequential(
            nn.Linear(config.action_dim, config.d_model),
            nn.LayerNorm(config.d_model),
        )

        # 5. Learnable [PRED] token
        self.pred_token = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)

        # 6. Token type embeddings (to distinguish visual / action / pred)
        self.token_type_embedding = nn.Embedding(3, config.d_model)
        # 0=PRED, 1=visual, 2=action

        # 7. Positional encoding for visual tokens
        # DINOv2 ViT-B/14 with 224x224 input: 16x16 = 256 patches + 1 CLS
        max_visual_tokens = (config.img_size // 14) ** 2 + 1  # patches + CLS
        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_visual_tokens + 2, config.d_model) * 0.02
        )  # +2 for PRED token, action token

        # 8. Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
            norm=nn.LayerNorm(config.d_model),
        )

        # 9. Classification head: [PRED] token → binary logit
        self.classifier = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize model weights (excluding frozen visual encoder)."""
        for module in [self.visual_proj, self.action_proj, self.classifier]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        nn.init.normal_(self.token_type_embedding.weight, std=0.02)

    @torch.no_grad()
    def _extract_visual_features(self, rgb: torch.Tensor) -> torch.Tensor:
        """Extract visual features from DINOv2.

        Args:
            rgb: (B, 3, H, W) normalized images.

        Returns:
            (B, 1+N_patches, visual_dim) CLS + patch tokens.
        """
        if self.config.freeze_visual_encoder:
            self.visual_encoder.eval()

        # DINOv2 forward: get both CLS and patch tokens
        features = self.visual_encoder.forward_features(rgb)
        patch_tokens = features["x_norm_patchtokens"]  # (B, N_patches, D)
        cls_token = features["x_norm_clstoken"]  # (B, D)

        # Concatenate CLS + patch tokens
        cls_token = cls_token.unsqueeze(1)  # (B, 1, D)
        visual_tokens = torch.cat([cls_token, patch_tokens], dim=1)  # (B, 1+N, D)

        return visual_tokens

    def forward(
        self,
        rgb: torch.Tensor,
        gripper_action: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            rgb: (B, 3, H, W) normalized RGB images at time t.
            gripper_action: (B, 3) gripper action (velocity) at time t.

        Returns:
            (B, 1) logits for current-frame contact prediction.
        """
        B = rgb.shape[0]
        device = rgb.device

        # 1. Extract visual features
        if self.config.freeze_visual_encoder:
            with torch.no_grad():
                visual_tokens = self._extract_visual_features(rgb)
        else:
            visual_tokens = self._extract_visual_features(rgb)
        # visual_tokens: (B, 1+N_patches, visual_dim)

        # 2. Project visual tokens to d_model
        visual_tokens = self.visual_proj(visual_tokens)  # (B, 1+N, d_model)

        # 4. Project gripper action
        action_token = self.action_proj(gripper_action)  # (B, d_model)
        action_token = action_token.unsqueeze(1)  # (B, 1, d_model)

        # 5. Prepend [PRED] token
        pred_token = self.pred_token.expand(B, -1, -1)  # (B, 1, d_model)

        # 6. Concatenate all tokens: [PRED, visual..., action]
        tokens = torch.cat(
            [pred_token, visual_tokens, action_token], dim=1
        )
        # tokens: (B, 1 + 1+N_patches + 1, d_model)

        # 7. Add token type embeddings
        n_vis = visual_tokens.shape[1]
        type_ids = torch.cat([
            torch.zeros(1, dtype=torch.long, device=device),        # PRED
            torch.ones(n_vis, dtype=torch.long, device=device),     # visual
            torch.full((1,), 2, dtype=torch.long, device=device),   # action
        ])  # (seq_len,)
        type_emb = self.token_type_embedding(type_ids)  # (seq_len, d_model)
        tokens = tokens + type_emb.unsqueeze(0)

        # 8. Add positional encoding
        seq_len = tokens.shape[1]
        tokens = tokens + self.pos_encoding[:, :seq_len, :]

        # 9. Transformer encoder
        tokens = self.transformer(tokens)  # (B, seq_len, d_model)

        # 10. Take [PRED] token output → classify
        pred_output = tokens[:, 0, :]  # (B, d_model)
        logits = self.classifier(pred_output)  # (B, 1)

        return logits
