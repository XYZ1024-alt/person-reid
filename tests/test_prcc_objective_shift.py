from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch

from pedestrian_reid.builders import MODE_PRCC, MODE_PRCC_DEV, build_eval_loader, build_training_dataset
from pedestrian_reid.builders import selected_prcc_dev_pids
from pedestrian_reid.data.datasets import PRCC_CAMERAS, PRCC_GALLERY_CAMERA, PRCC_QUERY_CAMERA, PRCC_SOURCE
from pedestrian_reid.data.transforms import VARIANT_STANDARD
from pedestrian_reid.engine import trainer


class PrccObjectiveShiftTest(unittest.TestCase):
    def test_prcc_dev_split_excludes_training_identities(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            _write_prcc_train(Path(root), range(1, 7))
            args = _args(root, prcc_dev_identities=2, prcc_dev_seed=3)

            dev_pids = set(selected_prcc_dev_pids(args))
            training_dataset = build_training_dataset(args)
            training_pids = {sample.pid for sample in training_dataset.samples}
            query_loader = build_eval_loader(root, MODE_PRCC_DEV, "query", VARIANT_STANDARD, args)
            gallery_loader = build_eval_loader(root, MODE_PRCC_DEV, "gallery", VARIANT_STANDARD, args)

            self.assertTrue(dev_pids)
            self.assertTrue(dev_pids.isdisjoint(training_pids))
            self.assertEqual({sample.pid for sample in query_loader.dataset.samples}, dev_pids)
            self.assertEqual({sample.pid for sample in gallery_loader.dataset.samples}, dev_pids)
            self.assertEqual({sample.camid for sample in query_loader.dataset.samples}, {PRCC_CAMERAS[PRCC_QUERY_CAMERA]})
            self.assertEqual({sample.camid for sample in gallery_loader.dataset.samples}, {PRCC_CAMERAS[PRCC_GALLERY_CAMERA]})

    def test_prcc_ce_weight_ramps_to_final_weight(self) -> None:
        args = SimpleNamespace(prcc_ce_weight=0.2, prcc_ce_final_weight=0.0, prcc_ce_ramp_epochs=5)

        self.assertAlmostEqual(trainer._effective_prcc_ce_weight(args, 0), 0.2)
        self.assertAlmostEqual(trainer._effective_prcc_ce_weight(args, 4), 0.0)
        self.assertAlmostEqual(trainer._effective_prcc_ce_weight(args, 5), 0.0)

    def test_training_feature_key_fails_when_missing(self) -> None:
        with self.assertRaisesRegex(ValueError, "triplet_feature_key=combined_features"):
            trainer._training_feature_output({"features": torch.zeros(2, 4)}, "combined_features")

    def test_cross_clothes_contrastive_backpropagates(self) -> None:
        features = torch.randn(4, 8, requires_grad=True)
        outputs = {"combined_features": features}
        labels = torch.tensor([0, 0, 1, 1])
        clothes = torch.tensor([0, 1, 0, 1])
        args = _contrastive_args()

        loss = trainer._cross_clothes_contrastive_loss(outputs, labels, clothes, [PRCC_SOURCE] * 4, args, features.device)
        loss.backward()

        self.assertTrue(torch.isfinite(loss).item())
        self.assertIsNotNone(features.grad)

    def test_cross_clothes_contrastive_requires_positive_pairs(self) -> None:
        features = torch.randn(2, 8)
        outputs = {"combined_features": features}
        labels = torch.tensor([0, 1])
        clothes = torch.tensor([0, 1])
        args = _contrastive_args()

        with self.assertRaisesRegex(ValueError, "no positive cross-clothes pairs"):
            trainer._cross_clothes_contrastive_loss(outputs, labels, clothes, [PRCC_SOURCE] * 2, args, features.device)


def _write_prcc_train(root: Path, pids) -> None:
    for pid in pids:
        pid_dir = root / "rgb" / "train" / f"{pid:03d}"
        pid_dir.mkdir(parents=True)
        for camera in ("A", "B", "C"):
            (pid_dir / f"{camera}_sample.jpg").write_bytes(b"placeholder")


def _args(root: str, *, prcc_dev_identities: int, prcc_dev_seed: int) -> SimpleNamespace:
    return SimpleNamespace(
        mode=MODE_PRCC,
        prcc_root=root,
        market_root="",
        use_prcc_sketch=False,
        prcc_dev_identities=prcc_dev_identities,
        prcc_dev_seed=prcc_dev_seed,
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        flip_probability=0.0,
        color_jitter_probability=0.0,
        random_grayscale_probability=0.0,
        dark_augment_probability=0.0,
        occlusion_augment_probability=0.0,
    )


def _contrastive_args() -> SimpleNamespace:
    return SimpleNamespace(
        cross_clothes_contrastive_weight=0.2,
        triplet_feature_key="combined_features",
        contrastive_temperature=0.07,
    )


if __name__ == "__main__":
    unittest.main()
