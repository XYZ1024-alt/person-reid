from __future__ import annotations

import torch
import torch.nn.functional as F


MIN_VALID_ANCHORS = 1
DISTANCE_EPSILON = 1e-12


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

