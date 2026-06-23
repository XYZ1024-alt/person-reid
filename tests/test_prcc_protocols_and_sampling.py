from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from pedestrian_reid.builders import MODE_PRCC, build_train_loader, build_training_dataset
from pedestrian_reid.data.datasets import PRCC_SOURCE, ReidSample
from pedestrian_reid.data.samplers import ClothesAwareIdentityBatchSampler
from pedestrian_reid.engine.evaluator import enabled_eval_jobs
from pedestrian_reid.modules.metrics import (
    FeatureBank,
    PROTOCOL_CLOTH_CHANGE,
    PROTOCOL_SAME_CLOTHES,
    evaluate_reid,
)


BATCH_SIZE = 4
INSTANCES = 2
QUERY_CAMERA = 3
GALLERY_CAMERA = 1
SAME_CLOTHES = 7
CHANGED_CLOTHES = 8


class PrccProtocolsAndSamplingTest(unittest.TestCase):
    def test_same_clothes_protocol_keeps_same_clothes_positive(self) -> None:
        metrics = evaluate_reid(_query_bank(), _gallery_bank(), PROTOCOL_SAME_CLOTHES)

        self.assertAlmostEqual(metrics["rank1"], 1.0)

    def test_cloth_change_protocol_ignores_same_clothes_positive(self) -> None:
        metrics = evaluate_reid(_query_bank(), _gallery_bank(), PROTOCOL_CLOTH_CHANGE)

        self.assertAlmostEqual(metrics["rank1"], 0.0)
        self.assertAlmostEqual(metrics["rank2"], 1.0)

    def test_prcc_eval_jobs_include_same_and_cloth_change_protocols(self) -> None:
        args = SimpleNamespace(mode=MODE_PRCC, prcc_root="prcc", prcc_dev_identities=0)

        jobs = enabled_eval_jobs(args)
        protocols_by_name = {job.name: job.protocol for job in jobs}

        self.assertEqual(protocols_by_name["prcc_same_clothes"], PROTOCOL_SAME_CLOTHES)
        self.assertEqual(protocols_by_name["prcc_cloth_change"], PROTOCOL_CLOTH_CHANGE)

    def test_prcc_train_loader_uses_clothes_aware_sampler(self) -> None:
        loader = build_train_loader(_TinyDataset(_sampler_samples()), _sampler_args())

        self.assertIsInstance(loader.batch_sampler, ClothesAwareIdentityBatchSampler)

    def test_clothes_aware_sampler_covers_two_clothes_per_identity(self) -> None:
        samples = _sampler_samples()
        sampler = ClothesAwareIdentityBatchSampler(samples, BATCH_SIZE, INSTANCES)

        batch = next(iter(sampler))
        clothes_by_label = _sampled_clothes_by_label(samples, batch)

        self.assertEqual(clothes_by_label, {0: {0, 1}, 1: {0, 1}})

    def test_joint_training_mode_is_rejected(self) -> None:
        args = SimpleNamespace(mode="joint")

        with self.assertRaisesRegex(ValueError, "Unknown training mode"):
            build_training_dataset(args)


class _TinyDataset:
    def __init__(self, samples: list[ReidSample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        return {"index": index}


def _query_bank() -> FeatureBank:
    return FeatureBank(
        features=torch.tensor([[1.0, 0.0]]),
        pids=torch.tensor([1]),
        camids=torch.tensor([QUERY_CAMERA]),
        clothes_ids=torch.tensor([SAME_CLOTHES]),
        is_junk=torch.tensor([False]),
        paths=["query.jpg"],
    )


def _gallery_bank() -> FeatureBank:
    return FeatureBank(
        features=torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]]),
        pids=torch.tensor([1, 1, 2]),
        camids=torch.tensor([GALLERY_CAMERA, GALLERY_CAMERA, GALLERY_CAMERA]),
        clothes_ids=torch.tensor([SAME_CLOTHES, CHANGED_CLOTHES, CHANGED_CLOTHES]),
        is_junk=torch.tensor([False, False, False]),
        paths=["same.jpg", "changed.jpg", "negative.jpg"],
    )


def _sampler_samples() -> list[ReidSample]:
    return [
        _sample(index=0, label=0, clothes_id=0),
        _sample(index=1, label=0, clothes_id=1),
        _sample(index=2, label=1, clothes_id=0),
        _sample(index=3, label=1, clothes_id=1),
    ]


def _sample(*, index: int, label: int, clothes_id: int) -> ReidSample:
    return ReidSample(PRCC_SOURCE, Path(f"{index}.jpg"), None, label, GALLERY_CAMERA, clothes_id, label, False)


def _sampler_args() -> SimpleNamespace:
    return SimpleNamespace(
        mode=MODE_PRCC,
        batch_size=BATCH_SIZE,
        instances=INSTANCES,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )


def _sampled_clothes_by_label(samples: list[ReidSample], batch: list[int]) -> dict[int, set[int]]:
    grouped: dict[int, set[int]] = {}
    for index in batch:
        sample = samples[index]
        grouped.setdefault(sample.label, set()).add(sample.clothes_id)
    return grouped


if __name__ == "__main__":
    unittest.main()
