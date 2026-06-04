from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from torch.utils.data import DataLoader

from robust_person_reid.data.datasets import (
    ReIDDataset,
    ReidSample,
    load_market_samples,
    load_prcc_samples,
    relabel_samples,
)
from robust_person_reid.data.samplers import (
    ClothesAwareIdentityBatchSampler,
    IdentityBatchSampler,
    SourceBalancedIdentityBatchSampler,
    SourceBalancedSamplerConfig,
)
from robust_person_reid.data.transforms import ReIDTransform, TransformConfig


MODE_MARKET = "market"
MODE_PRCC = "prcc"
MODE_JOINT = "joint"


def build_training_dataset(args: Namespace) -> ReIDDataset:
    samples = _training_samples(args)
    transform = ReIDTransform(TransformConfig(train=True))
    return ReIDDataset(relabel_samples(samples), transform)


def build_eval_loader(root: str | Path, dataset_name: str, split: str, variant: str, args: Namespace) -> DataLoader:
    samples = _eval_samples(root, dataset_name, split)
    transform = ReIDTransform(TransformConfig(train=False, variant=variant))
    dataset = ReIDDataset(samples, transform)
    return DataLoader(dataset, batch_size=args.batch_size, **_eval_loader_kwargs(args))


def build_train_loader(dataset: ReIDDataset, args: Namespace) -> DataLoader:
    if _use_source_balanced_sampling(args):
        config = SourceBalancedSamplerConfig(
            samples=dataset.samples,
            batch_size=args.batch_size,
            instances=args.instances,
            source_ratio=args.prcc_identities_ratio,
        )
        sampler = SourceBalancedIdentityBatchSampler(config)
        return DataLoader(dataset, batch_sampler=sampler, **_loader_kwargs(args))
    if args.mode == MODE_PRCC:
        sampler = ClothesAwareIdentityBatchSampler(dataset.samples, args.batch_size, args.instances)
        return DataLoader(dataset, batch_sampler=sampler, **_loader_kwargs(args))
    sampler = IdentityBatchSampler(dataset.samples, args.batch_size, args.instances)
    return DataLoader(dataset, batch_sampler=sampler, **_loader_kwargs(args))


def _use_source_balanced_sampling(args: Namespace) -> bool:
    return args.mode == MODE_JOINT and not args.disable_source_balanced_sampling


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
        return load_prcc_samples(args.prcc_root, "train", args.use_prcc_sketch)
    _require_prcc_root(args.prcc_root)
    market_samples = load_market_samples(args.market_root, "train")
    prcc_samples = load_prcc_samples(args.prcc_root, "train", args.use_prcc_sketch)
    return market_samples + prcc_samples


def _eval_samples(root: str | Path, dataset_name: str, split: str) -> list[ReidSample]:
    if dataset_name == MODE_MARKET:
        return load_market_samples(root, split)
    if dataset_name == MODE_PRCC:
        return load_prcc_samples(root, split)
    raise ValueError(f"Unknown eval dataset: {dataset_name}")


def _require_prcc_root(root: str | Path) -> None:
    if not Path(root).exists():
        raise FileNotFoundError(f"PRCC root is required for this mode: {root}")
