"""
Foundation Model backbones for person re-identification.
Supports Vision Transformer architectures (CLIP ViT-L, EVA-02 Large).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import torch
import torch.nn as nn


class BaseBackbone(ABC, nn.Module):
    """Abstract base class for all backbone architectures."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the backbone.

        Args:
            x: Input images [B, 3, H, W]

        Returns:
            Features in format specified by output_format()
            - 'spatial': [B, C, H', W'] for CNN backbones
            - 'sequence': [B, D] CLS token for ViT backbones
        """
        pass

    @abstractmethod
    def output_dim(self) -> int:
        """Return the output feature dimension."""
        pass

    @abstractmethod
    def output_format(self) -> Literal['spatial', 'sequence']:
        """
        Return the output format type.

        Returns:
            'spatial': CNN-style feature maps [B, C, H, W]
            'sequence': ViT-style CLS token [B, D]
        """
        pass


class CLIPViTBackbone(BaseBackbone):
    """
    CLIP Vision Transformer Large (ViT-L/14) backbone.

    Pretrained on 400M image-text pairs with contrastive learning.
    Expected performance on PRCC: 70-73% mAP.
    """

    def __init__(self, pretrained: bool = True, freeze_patch_embed: bool = True):
        super().__init__()
        try:
            from transformers import CLIPVisionModel
        except ImportError:
            raise ImportError(
                "transformers is required for CLIP backbone. "
                "Install with: pip install transformers"
            )

        model_name = "openai/clip-vit-large-patch14"
        self.model = CLIPVisionModel.from_pretrained(model_name) if pretrained else CLIPVisionModel.from_config(model_name)

        # Freeze patch embedding and position embedding (common practice)
        if freeze_patch_embed:
            for param in self.model.embeddings.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CLIP ViT.

        Args:
            x: Input tensor [B, 3, H, W] - any size will be resized to 224x224

        Returns:
            CLS token features [B, 1024]
        """
        # CLIP expects 224x224 input, resize if needed
        if x.shape[2] != 224 or x.shape[3] != 224:
            x = torch.nn.functional.interpolate(
                x, size=(224, 224), mode='bilinear', align_corners=False
            )

        outputs = self.model(x, output_hidden_states=False)
        # Return the pooled output (CLS token after final layer norm)
        return outputs.pooler_output

    def output_dim(self) -> int:
        return 1024  # CLIP-L hidden size

    def output_format(self) -> Literal['spatial', 'sequence']:
        return 'sequence'


class EVA02LBackbone(BaseBackbone):
    """
    EVA-02 Large backbone from timm.

    Pretrained on large-scale supervised learning.
    Expected performance on PRCC: 72-75% mAP.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        try:
            import timm
        except ImportError:
            raise ImportError(
                "timm is required for EVA02 backbone. "
                "Install with: pip install timm"
            )

        # Load EVA-02 Large from timm
        self.model = timm.create_model(
            'eva02_large_patch14_448.mim_in22k_ft_in1k',
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool=''   # We'll handle pooling ourselves
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through EVA-02.

        Args:
            x: Input tensor [B, 3, H, W] - any size will be resized to 448x448

        Returns:
            CLS token features [B, 1024]
        """
        # EVA-02 expects 448x448 input, resize if needed
        if x.shape[2] != 448 or x.shape[3] != 448:
            x = torch.nn.functional.interpolate(
                x, size=(448, 448), mode='bilinear', align_corners=False
            )

        x = self.model.forward_features(x)
        # Take CLS token (first token)
        return x[:, 0]

    def output_dim(self) -> int:
        return 1024  # EVA-02-L hidden size

    def output_format(self) -> Literal['spatial', 'sequence']:
        return 'sequence'


def create_backbone(
    backbone_type: Literal['clip_vit_l', 'eva02_l'],
    pretrained: bool = True
) -> BaseBackbone:
    """
    Factory function to create Foundation Model backbones.

    Args:
        backbone_type: Type of backbone to create
        pretrained: Whether to load pretrained weights

    Returns:
        Initialized backbone instance
    """
    if backbone_type == 'clip_vit_l':
        return CLIPViTBackbone(pretrained=pretrained)
    elif backbone_type == 'eva02_l':
        return EVA02LBackbone(pretrained=pretrained)
    else:
        raise ValueError(
            f"Unknown backbone type: {backbone_type}. "
            f"Supported: clip_vit_l, eva02_l"
        )
