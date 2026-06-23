from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


MIN_VALID_ANCHORS = 1
DISTANCE_EPSILON = 1e-12
MIN_POSITIVE_COUNT = 1
MIN_CONTRASTIVE_TEMPERATURE = 0.0
MIN_HARD_NEGATIVE_WEIGHT = 0.0


def pairwise_distances(features: torch.Tensor) -> torch.Tensor:
    squared_norm = torch.sum(features * features, dim=1, keepdim=True)
    distances = squared_norm + squared_norm.t() - 2.0 * features @ features.t()
    return distances.clamp(min=DISTANCE_EPSILON).sqrt()


def batch_hard_triplet_loss(features: torch.Tensor, labels: torch.Tensor, margin: float) -> torch.Tensor:
    distances = pairwise_distances(features)
    same_identity = labels.unsqueeze(0).eq(labels.unsqueeze(1))
    different_identity = ~same_identity
    same_identity.fill_diagonal_(False)
    valid_anchor = same_identity.any(dim=1) & different_identity.any(dim=1)
    if valid_anchor.sum().item() < MIN_VALID_ANCHORS:
        raise ValueError("Batch-hard triplet loss needs positive and negative pairs")
    positive = distances.masked_fill(~same_identity, -1.0).max(dim=1).values
    negative = distances.masked_fill(~different_identity, float("inf")).min(dim=1).values
    return F.relu(positive - negative + margin)[valid_anchor].mean()


class ClothInvariantContrastiveLoss(nn.Module):
    def __init__(self, *, temperature: float, hard_negative_weight: float, feature_key: str) -> None:
        super().__init__()
        _validate_contrastive_config(temperature, hard_negative_weight)
        self.temperature = float(temperature)
        self.hard_negative_weight = float(hard_negative_weight)
        self.feature_key = feature_key

    def forward(self, outputs: dict[str, torch.Tensor], labels: torch.Tensor, clothes_labels: torch.Tensor) -> torch.Tensor:
        features = _contrastive_features(outputs, self.feature_key)
        masks = _contrastive_masks(labels, clothes_labels)
        positive_rows = masks["positive"].any(dim=1)
        if not positive_rows.any().item():
            raise ValueError("cross-clothes contrastive loss found no positive cross-clothes pairs")
        similarities = _pairwise_cosine(features.float()) / self.temperature
        weighted = _weight_hard_negatives(similarities, masks["hard_negative"], self.hard_negative_weight)
        log_prob = weighted - weighted.masked_fill(~masks["denominator"], float("-inf")).logsumexp(dim=1, keepdim=True)
        positive_log_prob = _positive_log_probability(log_prob, masks["positive"])
        return -positive_log_prob[positive_rows].mean()


def _validate_contrastive_config(temperature: float, hard_negative_weight: float) -> None:
    if temperature <= MIN_CONTRASTIVE_TEMPERATURE:
        raise ValueError("contrastive_temperature must be > 0")
    if hard_negative_weight <= MIN_HARD_NEGATIVE_WEIGHT:
        raise ValueError("cross_clothes_hard_negative_weight must be > 0")


def _contrastive_features(outputs: dict[str, torch.Tensor], feature_key: str) -> torch.Tensor:
    if feature_key not in outputs:
        raise ValueError(f"Model did not produce feature_key={feature_key}; enable the matching model branch")
    features = outputs[feature_key]
    if features.ndim != 2:
        raise ValueError(f"feature_key={feature_key} must produce a 2D tensor")
    return F.normalize(features.float(), dim=1)


def _contrastive_masks(labels: torch.Tensor, clothes_labels: torch.Tensor) -> dict[str, torch.Tensor]:
    same_identity = labels.unsqueeze(0).eq(labels.unsqueeze(1))
    same_clothes = clothes_labels.unsqueeze(0).eq(clothes_labels.unsqueeze(1))
    different_identity = ~same_identity
    positive_mask = same_identity & ~same_clothes
    hard_negative_mask = different_identity & same_clothes
    denominator_mask = positive_mask | different_identity
    return {
        "positive": positive_mask,
        "hard_negative": hard_negative_mask,
        "denominator": denominator_mask,
    }


def _pairwise_cosine(features: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(features, dim=1)
    return normalized @ normalized.t()


def _weight_hard_negatives(
    similarities: torch.Tensor,
    hard_negative_mask: torch.Tensor,
    hard_negative_weight: float,
) -> torch.Tensor:
    weighted = similarities.clone()
    weighted[hard_negative_mask] = weighted[hard_negative_mask] + math.log(hard_negative_weight)
    return weighted


def _positive_log_probability(log_prob: torch.Tensor, positive_mask: torch.Tensor) -> torch.Tensor:
    positive_count = positive_mask.sum(dim=1).clamp(min=MIN_POSITIVE_COUNT)
    return (log_prob * positive_mask.float()).sum(dim=1) / positive_count
