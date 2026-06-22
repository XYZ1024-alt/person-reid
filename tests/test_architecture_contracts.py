from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from pedestrian_reid.engine import evaluator, trainer
from pedestrian_reid.modules.model import PedestrianReIDNet


class ArchitectureContractsTest(unittest.TestCase):
    def test_load_model_requires_architecture_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            checkpoint_path = Path(root) / "missing_config.pth"
            torch.save({"model": {}, "num_classes": 2, "num_clothes_classes": 0, "model_config": {}}, checkpoint_path)

            with self.assertRaisesRegex(ValueError, "required model_config"):
                evaluator.load_model(str(checkpoint_path), torch.device("cpu"))

    def test_load_model_uses_checkpoint_architecture(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            with patch("pedestrian_reid.modules.backbones.create_backbone", _fake_create_backbone):
                model = _sketch_fusion_model(backbone_type="eva02_l")
                checkpoint_path = _write_checkpoint(Path(root), model)
                loaded = evaluator.load_model(str(checkpoint_path), torch.device("cpu"))

            self.assertEqual(loaded.backbone_type, "eva02_l")
            self.assertTrue(loaded.use_sketch_fusion)

    def test_backbone_freeze_restores_initial_trainability(self) -> None:
        model = _TinyModel()
        first_name, first_parameter = next(model.backbone.named_parameters())
        first_parameter.requires_grad = False
        trainer.record_initial_backbone_trainability(model)
        args = SimpleNamespace(freeze_backbone_all_epochs=False, freeze_backbone_epochs=1)

        trainer.configure_backbone_freeze(model, args, 0, distributed=trainer.DistributedContext())
        self.assertTrue(all(not parameter.requires_grad for parameter in model.backbone.parameters()))

        trainer.configure_backbone_freeze(model, args, 1, distributed=trainer.DistributedContext())
        restored = dict(model.backbone.named_parameters())
        self.assertFalse(restored[first_name].requires_grad)
        self.assertTrue(any(parameter.requires_grad for name, parameter in restored.items() if name != first_name))

    def test_sketch_fusion_returns_sketch_outputs(self) -> None:
        with patch("pedestrian_reid.modules.backbones.create_backbone", _fake_create_backbone):
            model = _sketch_fusion_model(backbone_type="clip_vit_l")
            outputs, sketch_outputs = model(
                torch.randn(2, 3, 4, 4),
                sketch_images=torch.randn(2, 3, 4, 4),
                return_sketch_outputs=True,
            )

        self.assertIn("features", outputs)
        self.assertIn("features", sketch_outputs)
        self.assertIn("combined_features", outputs)

    def test_part_branch_rejected_for_sequence_backbone(self) -> None:
        with patch("pedestrian_reid.modules.backbones.create_backbone", _fake_create_backbone):
            with self.assertRaisesRegex(ValueError, "spatial backbone"):
                PedestrianReIDNet(2, use_part_branch=True)


class _FakeSequenceBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, 8)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.projection(images.mean(dim=(2, 3)))

    def output_dim(self) -> int:
        return 8

    def output_format(self) -> str:
        return "sequence"


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))


def _fake_create_backbone(backbone_type: str, pretrained: bool = True):
    return _FakeSequenceBackbone()


def _sketch_fusion_model(backbone_type: str) -> PedestrianReIDNet:
    return PedestrianReIDNet(
        2,
        embedding_dim=4,
        num_clothes_classes=2,
        num_prcc_classes=2,
        backbone_type=backbone_type,
        use_sketch_fusion=True,
    )


def _write_checkpoint(root: Path, model: PedestrianReIDNet) -> Path:
    path = root / "model.pth"
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": trainer._model_config(model),
            "num_classes": 2,
            "num_clothes_classes": 2,
            "num_market_classes": 0,
            "num_prcc_classes": 2,
        },
        path,
    )
    return path
