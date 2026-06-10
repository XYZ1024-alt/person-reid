from __future__ import annotations

from pathlib import Path

from PIL import Image
import torch

from pedestrian_reid.data.transforms import ReIDTransform, TransformConfig
from pedestrian_reid.engine.evaluator import load_model


class PedestrianReIDPredictor:
    def __init__(self, checkpoint_path: str | Path, device: str | None = None):
        self.device = torch.device(device or _default_device())
        self.model = load_model(str(checkpoint_path), self.device)
        self.transform = ReIDTransform(TransformConfig(train=False))

    def __call__(self, image) -> torch.Tensor:
        tensor = self.transform(_to_pil_image(image)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(tensor)
        return outputs.get("combined_features", outputs["bn_features"]).squeeze(0)


def init_yolo(model_path: str):
    from ultralytics import YOLO

    return YOLO(model_path)


def init_reid(checkpoint_path: str | Path, device: str | None = None) -> PedestrianReIDPredictor:
    return PedestrianReIDPredictor(checkpoint_path, device)


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _to_pil_image(image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image
    return Image.fromarray(image[:, :, ::-1].copy())
