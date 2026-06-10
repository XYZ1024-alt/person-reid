from __future__ import annotations

import argparse

import torch

from pedestrian_reid.builders import MODE_JOINT, MODE_MARKET, MODE_PRCC
from pedestrian_reid.data.transforms import VARIANT_DARK, VARIANT_OCCLUDED, VARIANT_STANDARD
from pedestrian_reid.engine.trainer import train_from_args
from pedestrian_reid.modules.metrics import FEATURE_KEYS, REID_FEATURE_KEY
from pedestrian_reid.modules.model import (
    DEFAULT_COMBINED_GLOBAL_WEIGHT,
    DEFAULT_COMBINED_PART_WEIGHT,
    DEFAULT_NUM_PARTS,
    PART_EMBEDDING_DIM,
)
from pedestrian_reid.runtime import configure_torch_runtime


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
DEFAULT_CAL_WARMUP_EPOCHS = 10
DEFAULT_CAL_RAMP_EPOCHS = 10
DEFAULT_PRCC_IDENTITIES_RATIO = 0.5
DEFAULT_USE_PRCC_SKETCH = True
DEFAULT_SKETCH_LOSS_WEIGHT = 0.5
DEFAULT_RGB_SKETCH_CONSISTENCY_WEIGHT = 0.2
DEFAULT_SKETCH_WARMUP_EPOCHS = 0
DEFAULT_SKETCH_RAMP_EPOCHS = 0
DEFAULT_WORKERS = 4
DEFAULT_EVAL_PERIOD = 10
DEFAULT_SEED = 42
DEFAULT_PIN_MEMORY = True
DEFAULT_MULTI_GPU = False
DEFAULT_DISTRIBUTED = False
DEFAULT_DDP_FIND_UNUSED_PARAMETERS = "auto"
DEFAULT_BEST_METRIC = "rank1"
DEFAULT_BEST_VARIANT = VARIANT_STANDARD
DEFAULT_FEATURE_KEY = REID_FEATURE_KEY
DEFAULT_FREEZE_BACKBONE_EPOCHS = 0
DEFAULT_FREEZE_BACKBONE_LAYERS = "stem,layer1,layer2"
DEFAULT_USE_PART_BRANCH = False
DEFAULT_PARTS = DEFAULT_NUM_PARTS
DEFAULT_PART_EMBEDDING_DIM = PART_EMBEDDING_DIM
DEFAULT_PART_TRIPLET_WEIGHT = 0.0
DEFAULT_CLOTH_INVARIANT_WEIGHT = 0.0
DEFAULT_COMBINED_GLOBAL = DEFAULT_COMBINED_GLOBAL_WEIGHT
DEFAULT_COMBINED_PART = DEFAULT_COMBINED_PART_WEIGHT
DEFAULT_DISTILL_WEIGHT = 0.0
DEFAULT_DISTILL_FINAL_WEIGHT = 0.0
DEFAULT_DISTILL_HOLD_EPOCHS = 0
DEFAULT_DISTILL_RAMP_EPOCHS = 0
DEFAULT_FREEZE_BACKBONE_ALL_EPOCHS = False
DEFAULT_LR_MILESTONES = "40,70,100"
DEFAULT_LR_GAMMA = 0.1
DEFAULT_FLIP_PROBABILITY = 0.5
DEFAULT_COLOR_JITTER_PROBABILITY = 0.5
DEFAULT_RANDOM_GRAYSCALE_PROBABILITY = 0.0
DEFAULT_DARK_AUGMENT_PROBABILITY = 0.10
DEFAULT_OCCLUSION_AUGMENT_PROBABILITY = 0.10
PRECISION_FP16 = "fp16"
PRECISION_FP32 = "fp32"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a pure PyTorch ReID model")
    parser.add_argument("--mode", choices=[MODE_MARKET, MODE_PRCC, MODE_JOINT], default=MODE_JOINT)
    parser.add_argument("--market-root", default=DEFAULT_MARKET_ROOT)
    parser.add_argument("--prcc-root", default=DEFAULT_PRCC_ROOT)
    parser.add_argument("--output-dir", default="outputs/pedestrian_reid")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--instances", type=int, default=DEFAULT_INSTANCES)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--triplet-margin", type=float, default=DEFAULT_MARGIN)
    parser.add_argument("--triplet-weight", type=float, default=DEFAULT_TRIPLET_WEIGHT)
    parser.add_argument("--cal-weight", type=float, default=DEFAULT_CAL_WEIGHT)
    parser.add_argument("--cal-warmup-epochs", type=int, default=DEFAULT_CAL_WARMUP_EPOCHS)
    parser.add_argument("--cal-ramp-epochs", type=int, default=DEFAULT_CAL_RAMP_EPOCHS)
    parser.add_argument("--prcc-identities-ratio", type=float, default=DEFAULT_PRCC_IDENTITIES_RATIO)
    parser.add_argument("--disable-source-balanced-sampling", action="store_true")
    parser.add_argument("--use-prcc-sketch", action=argparse.BooleanOptionalAction, default=DEFAULT_USE_PRCC_SKETCH)
    parser.add_argument("--sketch-loss-weight", type=float, default=DEFAULT_SKETCH_LOSS_WEIGHT)
    parser.add_argument("--rgb-sketch-consistency-weight", type=float, default=DEFAULT_RGB_SKETCH_CONSISTENCY_WEIGHT)
    parser.add_argument("--sketch-warmup-epochs", type=int, default=DEFAULT_SKETCH_WARMUP_EPOCHS)
    parser.add_argument("--sketch-ramp-epochs", type=int, default=DEFAULT_SKETCH_RAMP_EPOCHS)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=DEFAULT_PIN_MEMORY)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--precision", choices=[PRECISION_FP16, PRECISION_FP32], default=default_precision())
    parser.add_argument("--multi-gpu", action=argparse.BooleanOptionalAction, default=DEFAULT_MULTI_GPU)
    parser.add_argument("--distributed", action=argparse.BooleanOptionalAction, default=DEFAULT_DISTRIBUTED)
    parser.add_argument("--ddp-find-unused-parameters", choices=["auto", "true", "false"], default=DEFAULT_DDP_FIND_UNUSED_PARAMETERS)
    add_augmentation_args(parser)
    parser.add_argument("--best-metric", choices=["rank1", "mAP"], default=DEFAULT_BEST_METRIC)
    parser.add_argument("--best-variant", choices=[VARIANT_STANDARD, VARIANT_DARK, VARIANT_OCCLUDED], default=DEFAULT_BEST_VARIANT)
    parser.add_argument("--feature-key", choices=sorted(FEATURE_KEYS), default=DEFAULT_FEATURE_KEY)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=DEFAULT_FREEZE_BACKBONE_EPOCHS)
    parser.add_argument("--freeze-backbone-layers", default=DEFAULT_FREEZE_BACKBONE_LAYERS)
    parser.add_argument("--use-part-branch", action=argparse.BooleanOptionalAction, default=DEFAULT_USE_PART_BRANCH)
    parser.add_argument("--num-parts", type=int, default=DEFAULT_PARTS)
    parser.add_argument("--part-embedding-dim", type=int, default=DEFAULT_PART_EMBEDDING_DIM)
    parser.add_argument("--part-triplet-weight", type=float, default=DEFAULT_PART_TRIPLET_WEIGHT)
    parser.add_argument("--cloth-invariant-weight", type=float, default=DEFAULT_CLOTH_INVARIANT_WEIGHT)
    parser.add_argument("--combined-global-weight", type=float, default=DEFAULT_COMBINED_GLOBAL)
    parser.add_argument("--combined-part-weight", type=float, default=DEFAULT_COMBINED_PART)
    parser.add_argument("--teacher-checkpoint", default="")
    parser.add_argument("--distill-weight", type=float, default=DEFAULT_DISTILL_WEIGHT)
    parser.add_argument("--distill-final-weight", type=float, default=DEFAULT_DISTILL_FINAL_WEIGHT)
    parser.add_argument("--distill-hold-epochs", type=int, default=DEFAULT_DISTILL_HOLD_EPOCHS)
    parser.add_argument("--distill-ramp-epochs", type=int, default=DEFAULT_DISTILL_RAMP_EPOCHS)
    parser.add_argument("--freeze-backbone-all-epochs", action="store_true", default=DEFAULT_FREEZE_BACKBONE_ALL_EPOCHS)
    parser.add_argument("--lr-milestones", default=DEFAULT_LR_MILESTONES)
    parser.add_argument("--lr-gamma", type=float, default=DEFAULT_LR_GAMMA)
    parser.add_argument("--eval-period", type=int, default=DEFAULT_EVAL_PERIOD)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--resume", default="")
    parser.add_argument("--pretrained-checkpoint", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def add_augmentation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--flip-probability", type=float, default=DEFAULT_FLIP_PROBABILITY)
    parser.add_argument("--color-jitter-probability", type=float, default=DEFAULT_COLOR_JITTER_PROBABILITY)
    parser.add_argument("--random-grayscale-probability", type=float, default=DEFAULT_RANDOM_GRAYSCALE_PROBABILITY)
    parser.add_argument("--dark-augment-probability", type=float, default=DEFAULT_DARK_AUGMENT_PROBABILITY)
    parser.add_argument("--occlusion-augment-probability", type=float, default=DEFAULT_OCCLUSION_AUGMENT_PROBABILITY)


def default_precision() -> str:
    if torch.cuda.is_available():
        return PRECISION_FP16
    return PRECISION_FP32


def main() -> None:
    configure_torch_runtime()
    train_from_args(parse_args())


if __name__ == "__main__":
    main()
