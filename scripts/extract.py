from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
import torch

from pedestrian_reid.data.transforms import ReIDTransform, TransformConfig
from pedestrian_reid.engine.evaluator import load_model
from pedestrian_reid.modules.metrics import FEATURE_KEYS, REID_FEATURE_KEY


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract one embedding with the PedestrianReID model")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--feature-key", choices=sorted(FEATURE_KEYS), default=REID_FEATURE_KEY)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    transform = ReIDTransform(TransformConfig(train=False))
    image = transform(Image.open(Path(args.image))).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(image)[args.feature_key].cpu().flatten()
    print(" ".join(f"{value:.6f}" for value in embedding.tolist()))


if __name__ == "__main__":
    main()
