from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import random

from torch.utils.data import DataLoader

from pedestrian_reid.data.datasets import (
    PRCC_CAMERAS,
    PRCC_GALLERY_CAMERA,
    PRCC_QUERY_CAMERA,
    ReIDDataset,
    ReidSample,
    load_market_samples,
    load_prcc_samples,
    relabel_samples,
)
from pedestrian_reid.data.samplers import (
    ClothesAwareIdentityBatchSampler,
    IdentityBatchSampler,
    SourceBalancedIdentityBatchSampler,
    SourceBalancedSamplerConfig,
)
from pedestrian_reid.data.transforms import ReIDTransform, TransformConfig


MODE_MARKET = "market"
MODE_PRCC = "prcc"
MODE_PRCC_DEV = "prcc_dev"
MODE_JOINT = "joint"
NO_PRCC_DEV_IDENTITIES = 0


def build_training_dataset(args: Namespace) -> ReIDDataset:
    samples = _training_samples(args)
    transform = ReIDTransform(_training_transform_config(args))
    return ReIDDataset(relabel_samples(samples), transform)


def build_eval_loader(root: str | Path, dataset_name: str, split: str, variant: str, args: Namespace) -> DataLoader:
    samples = _eval_samples(root, dataset_name, split, args)
    transform = ReIDTransform(TransformConfig(train=False, variant=variant))
    dataset = ReIDDataset(samples, transform)
    return DataLoader(dataset, batch_size=args.batch_size, **_eval_loader_kwargs(args))


def build_train_loader(dataset: ReIDDataset, args: Namespace, distributed=None) -> DataLoader:
    batch_size = _train_batch_size(args, distributed)
    if _use_source_balanced_sampling(args):
        config = SourceBalancedSamplerConfig(
            samples=dataset.samples,
            batch_size=batch_size,
            instances=args.instances,
            source_ratio=args.prcc_identities_ratio,
            epoch_batch_size=args.batch_size,
        )
        sampler = SourceBalancedIdentityBatchSampler(config)
        return DataLoader(dataset, batch_sampler=sampler, **_loader_kwargs(args))
    if args.mode == MODE_PRCC:
        sampler = ClothesAwareIdentityBatchSampler(
            dataset.samples,
            batch_size,
            args.instances,
            epoch_batch_size=args.batch_size,
        )
        return DataLoader(dataset, batch_sampler=sampler, **_loader_kwargs(args))
    sampler = IdentityBatchSampler(dataset.samples, batch_size, args.instances, epoch_batch_size=args.batch_size)
    return DataLoader(dataset, batch_sampler=sampler, **_loader_kwargs(args))


def _use_source_balanced_sampling(args: Namespace) -> bool:
    return args.mode == MODE_JOINT and not args.disable_source_balanced_sampling


def _train_batch_size(args: Namespace, distributed) -> int:
    if not getattr(distributed, "enabled", False):
        return args.batch_size
    return args.batch_size // distributed.world_size


def _training_transform_config(args: Namespace) -> TransformConfig:
    return TransformConfig(
        train=True,
        flip_probability=args.flip_probability,
        color_jitter_probability=args.color_jitter_probability,
        random_grayscale_probability=args.random_grayscale_probability,
        dark_probability=args.dark_augment_probability,
        occlusion_probability=args.occlusion_augment_probability,
    )


def _loader_kwargs(args: Namespace) -> dict:
    num_workers = args.num_workers
    return {
        "num_workers": num_workers,
        "pin_memory": bool(getattr(args, "pin_memory", False)),
        "persistent_workers": _persistent_workers(args, num_workers),
    }


def _eval_loader_kwargs(args: Namespace) -> dict:
    return {
        "num_workers": args.num_workers,
        "pin_memory": bool(getattr(args, "pin_memory", False)),
        "persistent_workers": False,
    }


def _persistent_workers(args: Namespace, num_workers: int) -> bool:
    requested = getattr(args, "persistent_workers", None)
    if requested is None:
        return num_workers > 0
    return bool(requested) and num_workers > 0


def _training_samples(args: Namespace) -> list[ReidSample]:
    if args.mode == MODE_MARKET:
        return load_market_samples(args.market_root, "train")
    if args.mode == MODE_PRCC:
        return _exclude_prcc_dev(load_prcc_samples(args.prcc_root, "train", args.use_prcc_sketch), args)
    _require_prcc_root(args.prcc_root)
    market_samples = load_market_samples(args.market_root, "train")
    prcc_samples = _exclude_prcc_dev(load_prcc_samples(args.prcc_root, "train", args.use_prcc_sketch), args)
    return market_samples + prcc_samples


def _eval_samples(root: str | Path, dataset_name: str, split: str, args: Namespace) -> list[ReidSample]:
    if dataset_name == MODE_MARKET:
        return load_market_samples(root, split)
    if dataset_name == MODE_PRCC:
        return load_prcc_samples(root, split)
    if dataset_name == MODE_PRCC_DEV:
        return _prcc_dev_eval_samples(root, split, args)
    raise ValueError(f"Unknown eval dataset: {dataset_name}")


def selected_prcc_dev_pids(args: Namespace) -> list[int]:
    count = int(getattr(args, "prcc_dev_identities", NO_PRCC_DEV_IDENTITIES))
    if count == NO_PRCC_DEV_IDENTITIES:
        return []
    pids = sorted({sample.pid for sample in load_prcc_samples(_prcc_root(args), "train")})
    _validate_prcc_dev_count(count, pids)
    return sorted(random.Random(args.prcc_dev_seed).sample(pids, count))


def _exclude_prcc_dev(samples: list[ReidSample], args: Namespace) -> list[ReidSample]:
    dev_pids = set(selected_prcc_dev_pids(args))
    if not dev_pids:
        return samples
    return [sample for sample in samples if sample.pid not in dev_pids]


def _prcc_dev_eval_samples(root: str | Path, split: str, args: Namespace) -> list[ReidSample]:
    dev_pids = set(selected_prcc_dev_pids(args))
    if not dev_pids:
        raise ValueError("prcc_dev evaluation requires --prcc-dev-identities > 0")
    samples = [sample for sample in load_prcc_samples(root, "train") if sample.pid in dev_pids]
    return _filter_prcc_dev_split(samples, split)


def _filter_prcc_dev_split(samples: list[ReidSample], split: str) -> list[ReidSample]:
    if split == "query":
        return _filter_by_prcc_camera(samples, PRCC_QUERY_CAMERA)
    if split == "gallery":
        return _filter_by_prcc_camera(samples, PRCC_GALLERY_CAMERA)
    raise ValueError(f"prcc_dev only supports query/gallery splits, got {split}")


def _filter_by_prcc_camera(samples: list[ReidSample], camera: str) -> list[ReidSample]:
    camid = PRCC_CAMERAS[camera]
    selected = [sample for sample in samples if sample.camid == camid]
    if not selected:
        raise ValueError(f"No PRCC dev samples found for camera {camera}")
    return selected


def _validate_prcc_dev_count(count: int, pids: list[int]) -> None:
    if count < NO_PRCC_DEV_IDENTITIES:
        raise ValueError("prcc_dev_identities must be >= 0")
    if count >= len(pids):
        raise ValueError(f"prcc_dev_identities must be less than PRCC train identities: {count} >= {len(pids)}")


def _prcc_root(args: Namespace) -> str | Path:
    root = getattr(args, "prcc_root", None)
    if root:
        return root
    fallback = getattr(args, "root", None)
    if fallback:
        return fallback
    raise AttributeError("args must provide prcc_root or root for PRCC data")


def _require_prcc_root(root: str | Path) -> None:
    if not Path(root).exists():
        raise FileNotFoundError(f"PRCC root is required for this mode: {root}")
