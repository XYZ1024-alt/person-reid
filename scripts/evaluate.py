from __future__ import annotations

import argparse

import torch

from robust_person_reid.builders import MODE_MARKET, MODE_PRCC
from robust_person_reid.engine.evaluator import evaluate_checkpoint


DEFAULT_BATCH_SIZE = 64
DEFAULT_WORKERS = 4
DEFAULT_MARKET_ROOT = "Market-1501"
DEFAULT_PRCC_ROOT = "prcc"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a RobustPersonReID checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", choices=[MODE_MARKET, MODE_PRCC], default=MODE_MARKET)
    parser.add_argument("--root", default="")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.root = args.root or default_root(args.dataset)
    return args


def default_root(dataset: str) -> str:
    if dataset == MODE_PRCC:
        return DEFAULT_PRCC_ROOT
    return DEFAULT_MARKET_ROOT


def main() -> None:
    evaluate_checkpoint(parse_args())


if __name__ == "__main__":
    main()
