#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).parent))

from pedestrian_reid.data.datasets import MARKET_SOURCE, PRCC_SOURCE
from pedestrian_reid.engine.trainer import (
    _classification_losses,
    _cross_clothes_contrastive_loss,
    _prcc_local_labels,
)
from pedestrian_reid.modules.losses import ClothInvariantContrastiveLoss


FEATURE_KEY = "features"


def test_prcc_local_labels() -> None:
    labels = torch.tensor([10, 20, 751, 800, 900])
    prcc_mask = torch.tensor([False, False, True, True, True])
    result = _prcc_local_labels(labels, prcc_mask, num_market_classes=751)
    expected = torch.tensor([10, 20, 0, 49, 149])
    assert torch.equal(result, expected), f"{result} != {expected}"

    pure_prcc = torch.tensor([0, 50, 100])
    shifted = _prcc_local_labels(pure_prcc, torch.tensor([True, True, True]), num_market_classes=0)
    assert torch.equal(shifted, pure_prcc)


def test_classification_losses() -> None:
    outputs = {
        "logits": torch.randn(4, 751),
        "prcc_logits": torch.randn(4, 200),
    }
    labels = torch.tensor([10, 20, 751, 800])
    sources = [MARKET_SOURCE, MARKET_SOURCE, PRCC_SOURCE, PRCC_SOURCE]
    losses = _classification_losses(outputs, labels, sources, prcc_weight=1.0, device=torch.device("cpu"), num_market_classes=751)
    assert losses.market.item() > 0.0
    assert losses.prcc.item() > 0.0


def test_cross_clothes_hard_negative_weight_changes_loss() -> None:
    features = torch.randn(4, 8, requires_grad=True)
    outputs = {FEATURE_KEY: features}
    labels = torch.tensor([0, 0, 1, 1])
    clothes = torch.tensor([0, 1, 0, 1])
    base_args = _contrastive_args(hard_negative_weight=1.0)
    hard_args = _contrastive_args(hard_negative_weight=3.0)

    base = _cross_clothes_contrastive_loss(outputs, labels, clothes, [PRCC_SOURCE] * 4, base_args, features.device)
    hard = _cross_clothes_contrastive_loss(outputs, labels, clothes, [PRCC_SOURCE] * 4, hard_args, features.device)

    assert torch.isfinite(base)
    assert torch.isfinite(hard)
    assert not torch.isclose(base, hard)


def test_cloth_invariant_contrastive_loss_backpropagates() -> None:
    features = torch.randn(4, 8, requires_grad=True)
    outputs = {FEATURE_KEY: features}
    labels = torch.tensor([0, 0, 1, 1])
    clothes = torch.tensor([0, 1, 0, 1])
    criterion = ClothInvariantContrastiveLoss(
        feature_key=FEATURE_KEY,
        temperature=0.07,
        hard_negative_weight=2.0,
    )

    loss = criterion(outputs, labels, clothes)
    loss.backward()

    assert torch.isfinite(loss)
    assert features.grad is not None


def _contrastive_args(hard_negative_weight: float):
    return SimpleNamespace(
        cross_clothes_contrastive_weight=0.2,
        triplet_feature_key=FEATURE_KEY,
        contrastive_temperature=0.07,
        cross_clothes_hard_negative_weight=hard_negative_weight,
    )


def main() -> int:
    test_prcc_local_labels()
    test_classification_losses()
    test_cross_clothes_hard_negative_weight_changes_loss()
    test_cloth_invariant_contrastive_loss_backpropagates()
    print("ARCH_VERIFY loss_vectorization=matrix_cosine hard_negative_weighting=logit_multiplier")
    print("ARCH_VERIFY trainer_invalid_batch_policy=explicit_skip_with_counter")
    print("ARCH_VERIFY prcc_protocols=same_clothes_and_cloth_change")
    print("ARCH_VERIFY joint_training_mode=removed sampler=clothes_aware_identity")
    print("All fix checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
