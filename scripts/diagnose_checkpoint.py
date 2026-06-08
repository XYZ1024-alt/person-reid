from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from pedestrian_reid.data.datasets import ReIDDataset, load_market_samples, relabel_samples
from pedestrian_reid.data.transforms import ReIDTransform, TransformConfig
from pedestrian_reid.engine.evaluator import load_model
from pedestrian_reid.modules.metrics import FEATURE_KEYS
from pedestrian_reid.runtime import configure_torch_runtime


DEFAULT_MARKET_ROOT = "Market-1501"
DEFAULT_BATCH_SIZE = 128
DEFAULT_WORKERS = 4
CHUNK_SIZE = 512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose checkpoint feature quality on Market train split")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--market-root", default=DEFAULT_MARKET_ROOT)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    configure_torch_runtime()
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    print_checkpoint_metadata(checkpoint)
    model = load_model(args.checkpoint, device)
    loader = build_market_train_loader(args)
    print(f"train_classifier_acc={classifier_accuracy(model, loader, device):.4f}")
    for feature_key in sorted(FEATURE_KEYS):
        features, pids = collect_features(model, loader, device, feature_key)
        print(f"train_nn_rank1/{feature_key}={leave_one_out_rank1(features, pids):.4f}")


def print_checkpoint_metadata(checkpoint: dict) -> None:
    keys = ["epoch", "best_metric", "best_variant", "best_metric_value", "num_classes", "num_clothes_classes"]
    for key in keys:
        print(f"checkpoint_{key}={checkpoint.get(key, '')}")


def build_market_train_loader(args: argparse.Namespace) -> DataLoader:
    samples = relabel_samples(load_market_samples(args.market_root, "train"))
    dataset = ReIDDataset(samples, ReIDTransform(TransformConfig(train=False)))
    return DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)


def classifier_accuracy(model, loader: DataLoader, device: torch.device) -> float:
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            predictions = model(images)["logits"].argmax(dim=1)
            correct += int(predictions.eq(labels).sum().item())
            total += int(labels.numel())
    return correct / total


def collect_features(model, loader: DataLoader, device: torch.device, feature_key: str) -> tuple[torch.Tensor, torch.Tensor]:
    features = []
    pids = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            features.append(model(images)[feature_key].cpu())
            pids.append(batch["pid"])
    return torch.cat(features), torch.cat(pids)


def leave_one_out_rank1(features: torch.Tensor, pids: torch.Tensor) -> float:
    features = torch.nn.functional.normalize(features, dim=1)
    hits = 0
    for start in range(0, len(features), CHUNK_SIZE):
        stop = min(start + CHUNK_SIZE, len(features))
        similarities = features[start:stop] @ features.t()
        row_indices = torch.arange(start, stop)
        similarities[torch.arange(stop - start), row_indices] = -float("inf")
        nearest = similarities.argmax(dim=1)
        hits += int(pids[nearest].eq(pids[start:stop]).sum().item())
    return hits / len(features)


if __name__ == "__main__":
    main()
