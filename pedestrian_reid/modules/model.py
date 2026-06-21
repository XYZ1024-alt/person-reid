from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


INPUT_CHANNELS = 3
EMBEDDING_DIM = 256
PART_EMBEDDING_DIM = 256
DEFAULT_NUM_PARTS = 6
DEFAULT_COMBINED_GLOBAL_WEIGHT = 0.7
DEFAULT_COMBINED_PART_WEIGHT = 0.3
GRAD_REVERSE_SCALE = 1.0
MIN_PARTS = 1


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = scale
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx, gradients: torch.Tensor):
        return gradients.neg().mul(ctx.scale), None


class PartFeatureBranch(nn.Module):
    def __init__(self, num_parts: int, embedding_dim: int, in_channels: int):
        super().__init__()
        if num_parts < MIN_PARTS:
            raise ValueError(f"num_parts must be >= {MIN_PARTS}, got {num_parts}")
        self.num_parts = num_parts
        self.embedding_dim = embedding_dim
        self.in_channels = in_channels
        self.pool = nn.AdaptiveAvgPool2d((num_parts, 1))
        self.embeddings = nn.ModuleList(
            nn.Linear(in_channels, embedding_dim, bias=False) for _ in range(num_parts)
        )
        self.bnnecks = nn.ModuleList(nn.BatchNorm1d(embedding_dim) for _ in range(num_parts))

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        stripes = self.pool(feature_map).squeeze(-1).permute(0, 2, 1)
        part_features = [self._part_feature(stripes[:, index, :], index) for index in range(self.num_parts)]
        return torch.stack(part_features, dim=1)

    def _part_feature(self, stripe: torch.Tensor, index: int) -> torch.Tensor:
        embedding = self.embeddings[index](stripe)
        return F.normalize(self.bnnecks[index](embedding), dim=1)


class DomainDiscriminator(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class PedestrianReIDNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = EMBEDDING_DIM,
        num_clothes_classes: int = 0,
        *,
        use_part_branch: bool = False,
        num_parts: int = DEFAULT_NUM_PARTS,
        part_embedding_dim: int = PART_EMBEDDING_DIM,
        combined_global_weight: float = DEFAULT_COMBINED_GLOBAL_WEIGHT,
        combined_part_weight: float = DEFAULT_COMBINED_PART_WEIGHT,
        use_dual_classifier: bool = False,
        num_market_classes: int = 0,
        num_prcc_classes: int = 0,
        use_domain_adversarial: bool = False,
        backbone_type: str = 'clip_vit_l',
        backbone_pretrained: bool = True,
    ):
        super().__init__()
        _validate_combined_weights(combined_global_weight, combined_part_weight)
        self.embedding_dim = embedding_dim
        self.use_part_branch = use_part_branch
        self.num_parts = num_parts
        self.part_embedding_dim = part_embedding_dim
        self.combined_global_weight = combined_global_weight
        self.combined_part_weight = combined_part_weight
        self.use_dual_classifier = use_dual_classifier
        self.num_market_classes = num_market_classes
        self.num_prcc_classes = num_prcc_classes
        self.use_domain_adversarial = use_domain_adversarial
        self.backbone_type = backbone_type

        # Dynamic backbone creation
        from pedestrian_reid.modules.backbones import create_backbone
        self.backbone = create_backbone(backbone_type, pretrained=backbone_pretrained)
        backbone_dim = self.backbone.output_dim()

        # Pooling layer depends on backbone output format
        if self.backbone.output_format() == 'spatial':
            self.pool = nn.AdaptiveAvgPool2d(1)
        else:
            # ViT already outputs CLS token, no pooling needed
            self.pool = nn.Identity()

        self.embedding = nn.Linear(backbone_dim, embedding_dim, bias=False)
        self.bnneck = nn.BatchNorm1d(embedding_dim)
        self.classifier = _identity_classifier(embedding_dim, num_market_classes if use_dual_classifier else num_classes)
        self.prcc_classifier = _identity_classifier(embedding_dim, num_prcc_classes) if use_dual_classifier else None
        self.clothes_classifier = _clothes_classifier(embedding_dim, num_clothes_classes)
        self.part_branch = _part_branch(use_part_branch, num_parts, part_embedding_dim, backbone_dim)
        self.domain_discriminator = DomainDiscriminator(embedding_dim) if use_domain_adversarial else None

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        feature_map = self.backbone(images)

        # Handle different backbone output formats
        if self.backbone.output_format() == 'spatial':
            # CNN backbone: [B, C, H, W]
            pooled = self.pool(feature_map).flatten(1)
        else:
            # ViT backbone: [B, D] (already CLS token)
            pooled = feature_map

        embedding = self.embedding(pooled)
        bn_features = self.bnneck(embedding)
        outputs = {
            "logits": self._market_logits(bn_features),
            "features": F.normalize(embedding, dim=1),
            "bn_features": F.normalize(bn_features, dim=1),
        }
        if self.prcc_classifier is not None:
            outputs["prcc_logits"] = self.prcc_classifier(bn_features)
        if self.part_branch is not None:
            # Part branch only for CNN backbones
            part_features = self.part_branch(feature_map)
            outputs["part_features"] = part_features
            outputs["combined_features"] = _combined_features(
                outputs["bn_features"],
                part_features,
                global_weight=self.combined_global_weight,
                part_weight=self.combined_part_weight,
            )
        else:
            # No part branch for ViT, use global features only
            outputs["combined_features"] = outputs["bn_features"]
        if self.clothes_classifier is not None or self.domain_discriminator is not None:
            reversed_features = GradientReverse.apply(bn_features, GRAD_REVERSE_SCALE)
            if self.clothes_classifier is not None:
                outputs["clothes_logits"] = self.clothes_classifier(reversed_features)
            if self.domain_discriminator is not None:
                outputs["domain_logits"] = self.domain_discriminator(reversed_features)
        return outputs

    def _market_logits(self, bn_features: torch.Tensor) -> torch.Tensor:
        if self.classifier is None:
            raise RuntimeError("Model has no identity classifier")
        return self.classifier(bn_features)


def _clothes_classifier(embedding_dim: int, num_clothes_classes: int) -> nn.Linear | None:
    if num_clothes_classes <= 0:
        return None
    return nn.Linear(embedding_dim, num_clothes_classes, bias=True)


def _identity_classifier(embedding_dim: int, num_classes: int) -> nn.Linear | None:
    if num_classes <= 0:
        return None
    return nn.Linear(embedding_dim, num_classes, bias=False)


def _part_branch(use_part_branch: bool, num_parts: int, part_embedding_dim: int, backbone_dim: int) -> PartFeatureBranch | None:
    if not use_part_branch:
        return None
    return PartFeatureBranch(num_parts, part_embedding_dim, in_channels=backbone_dim)


def _combined_features(
    global_features: torch.Tensor,
    part_features: torch.Tensor,
    *,
    global_weight: float,
    part_weight: float,
) -> torch.Tensor:
    pooled_parts = F.normalize(part_features.mean(dim=1), dim=1)
    weighted_global = global_features * global_weight
    weighted_parts = pooled_parts * part_weight
    return F.normalize(torch.cat((weighted_global, weighted_parts), dim=1), dim=1)


def _validate_combined_weights(global_weight: float, part_weight: float) -> None:
    if global_weight < 0.0 or part_weight < 0.0:
        raise ValueError("combined feature weights must be >= 0")
    if global_weight == 0.0 and part_weight == 0.0:
        raise ValueError("at least one combined feature weight must be > 0")


