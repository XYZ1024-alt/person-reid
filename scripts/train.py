from __future__ import annotations

import argparse

import torch

from robust_person_reid.builders import MODE_JOINT, MODE_MARKET, MODE_PRCC
from robust_person_reid.engine.trainer import train_from_args


DEFAULT_EPOCHS = 60
DEFAULT_BATCH_SIZE = 64
DEFAULT_MARKET_ROOT = "Market-1501"
DEFAULT_PRCC_ROOT = "prcc"
DEFAULT_INSTANCES = 4
DEFAULT_LR = 3e-4
DEFAULT_WEIGHT_DECAY = 5e-4
DEFAULT_MARGIN = 0.3
DEFAULT_TRIPLET_WEIGHT = 1.0
DEFAULT_CAL_WEIGHT = 0.5
DEFAULT_WORKERS = 4
DEFAULT_EVAL_PERIOD = 5
DEFAULT_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a pure PyTorch ReID model")
    parser.add_argument("--mode", choices=[MODE_MARKET, MODE_PRCC, MODE_JOINT], default=MODE_JOINT)
    parser.add_argument("--market-root", default=DEFAULT_MARKET_ROOT)
    parser.add_argument("--prcc-root", default=DEFAULT_PRCC_ROOT)
    parser.add_argument("--output-dir", default="outputs/robust_person_reid")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--instances", type=int, default=DEFAULT_INSTANCES)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--triplet-margin", type=float, default=DEFAULT_MARGIN)
    parser.add_argument("--triplet-weight", type=float, default=DEFAULT_TRIPLET_WEIGHT)
    parser.add_argument("--cal-weight", type=float, default=DEFAULT_CAL_WEIGHT)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--eval-period", type=int, default=DEFAULT_EVAL_PERIOD)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--resume", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    train_from_args(parse_args())


if __name__ == "__main__":
    main()
