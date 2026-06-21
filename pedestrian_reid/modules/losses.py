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


def cross_clothes_contrastive_loss_v2(
    features: torch.Tensor,
    pids: torch.Tensor,
    clothes_ids: torch.Tensor,
    temperature: float = 0.07,
    hard_negative_weight: float = 2.0,
) -> torch.Tensor:
    """
    Enhanced cross-clothes contrastive loss with hard negative emphasis.

    Addresses the critical challenge in CC-ReID:
    - Positive pairs: same ID, different clothes (pull together cross-clothes)
    - Hard negatives: different ID, same clothes (push apart similar-dressed people)

    The hard negative weighting aggressively penalizes false positives where
    different people wearing similar outfits are incorrectly matched.

    Args:
        features: L2-normalized features [B, D]
        pids: Person identity labels [B]
        clothes_ids: Clothes labels [B]
        temperature: Temperature scaling for softmax (default: 0.07)
        hard_negative_weight: Multiplier for hard negative similarities (default: 2.0)

    Returns:
        Scalar loss value (InfoNCE with hard negative emphasis)

    Raises:
        ValueError: If batch contains no valid positive pairs
    """
    # Cosine similarity matrix (features already L2-normalized)
    sim_matrix = features @ features.T / temperature  # [B, B]

    # Mask construction
    same_pid = pids.unsqueeze(0) == pids.unsqueeze(1)
    diff_pid = ~same_pid
    same_clothes = clothes_ids.unsqueeze(0) == clothes_ids.unsqueeze(1)
    diff_clothes = ~same_clothes

    # Positive pairs: same ID, different clothes (cross-clothes matching)
    pos_mask = same_pid & diff_clothes

    # Hard negative pairs: different ID, same clothes (confusing cases)
    hard_neg_mask = diff_pid & same_clothes

    # Check for valid anchors (rows with at least one positive pair)
    valid_anchors = pos_mask.any(dim=1)

    # Fallback: if no cross-clothes pairs, use standard identity contrastive
    if not valid_anchors.any():
        pos_mask = same_pid
        pos_mask.fill_diagonal_(False)  # Exclude self-similarity
        valid_anchors = pos_mask.any(dim=1)

        if not valid_anchors.any():
            raise ValueError(
                "Batch contains no valid positive pairs for contrastive learning. "
                "Need either cross-clothes pairs (same ID, different clothes) or "
                "at least 2 samples per identity."
            )

    # Apply hard negative weighting: scale up similarities for confusing pairs
    weighted_sim = sim_matrix.clone()
    weighted_sim[hard_neg_mask] = weighted_sim[hard_neg_mask] * hard_negative_weight

    # InfoNCE loss formulation
    # Denominator: positive pairs + all negative pairs (different IDs)
    denominator_mask = pos_mask | diff_pid

    # Log-sum-exp trick for numerical stability
    log_prob = weighted_sim - weighted_sim.masked_fill(
        ~denominator_mask, float('-inf')
    ).logsumexp(dim=1, keepdim=True)

    # Average log probability over positive pairs per anchor
    pos_count = pos_mask.sum(dim=1).clamp(min=1)
    pos_log_prob = (log_prob * pos_mask.float()).sum(dim=1) / pos_count

    # Return negative log-likelihood (only for valid anchors)
    return -pos_log_prob[valid_anchors].mean()
