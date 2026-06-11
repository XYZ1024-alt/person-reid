from __future__ import annotations

import argparse

import torch

from pedestrian_reid.builders import MODE_MARKET, MODE_PRCC, MODE_PRCC_DEV
from pedestrian_reid.engine.evaluator import evaluate_checkpoint
from pedestrian_reid.modules.metrics import FEATURE_KEYS, REID_FEATURE_KEY
from pedestrian_reid.runtime import configure_torch_runtime


DEFAULT_BATCH_SIZE = 64
DEFAULT_WORKERS = 4
DEFAULT_MARKET_ROOT = "Market-1501"
DEFAULT_PRCC_ROOT = "prcc"
DEFAULT_PRCC_DEV_IDENTITIES = 0
DEFAULT_PRCC_DEV_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a PedestrianReID checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", choices=[MODE_MARKET, MODE_PRCC, MODE_PRCC_DEV], default=MODE_MARKET)
    parser.add_argument("--root", default="")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--prcc-dev-identities", type=int, default=DEFAULT_PRCC_DEV_IDENTITIES)
    parser.add_argument("--prcc-dev-seed", type=int, default=DEFAULT_PRCC_DEV_SEED)
    parser.add_argument("--feature-key", choices=sorted(FEATURE_KEYS), default=REID_FEATURE_KEY)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.root = args.root or default_root(args.dataset)
    return args


def default_root(dataset: str) -> str:
    if dataset in {MODE_PRCC, MODE_PRCC_DEV}:
        return DEFAULT_PRCC_ROOT
    return DEFAULT_MARKET_ROOT


def main() -> None:
    configure_torch_runtime()
    evaluate_checkpoint(parse_args())


if __name__ == "__main__":
    main()
