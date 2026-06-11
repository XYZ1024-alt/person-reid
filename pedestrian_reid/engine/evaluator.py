from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass

import torch

from pedestrian_reid.builders import MODE_JOINT, MODE_MARKET, MODE_PRCC, MODE_PRCC_DEV, build_eval_loader
from pedestrian_reid.data.transforms import VARIANT_DARK, VARIANT_OCCLUDED, VARIANT_STANDARD
from pedestrian_reid.modules.metrics import PROTOCOL_CLOTH_CHANGE, PROTOCOL_STANDARD
from pedestrian_reid.modules.metrics import evaluate_reid, extract_feature_bank
from pedestrian_reid.modules.model import (
    DEFAULT_COMBINED_GLOBAL_WEIGHT,
    DEFAULT_COMBINED_PART_WEIGHT,
    DEFAULT_NUM_PARTS,
    EMBEDDING_DIM,
    PART_EMBEDDING_DIM,
    PedestrianReIDNet,
)


@dataclass(frozen=True)
class EvalJob:
    name: str
    root: str
    protocol: str


def validate_dataset(model, root: str, name: str, protocol: str, device: torch.device, args: Namespace):
    gallery_loader = build_eval_loader(root, name, "gallery", VARIANT_STANDARD, args)
    gallery_bank = extract_feature_bank(model, gallery_loader, device, args.feature_key)
    return {
        VARIANT_STANDARD: _validate_variant(model, root, name, VARIANT_STANDARD, gallery_bank, protocol, device, args),
        VARIANT_DARK: _validate_variant(model, root, name, VARIANT_DARK, gallery_bank, protocol, device, args),
        VARIANT_OCCLUDED: _validate_variant(model, root, name, VARIANT_OCCLUDED, gallery_bank, protocol, device, args),
    }


def evaluate_enabled_datasets(model, device: torch.device, args: Namespace):
    jobs = enabled_eval_jobs(args)
    return [_run_eval_job(model, job, device, args) for job in jobs]


def enabled_eval_jobs(args: Namespace) -> list[EvalJob]:
    jobs: list[EvalJob] = []
    if args.mode in {MODE_MARKET, MODE_JOINT}:
        jobs.append(EvalJob(MODE_MARKET, args.market_root, PROTOCOL_STANDARD))
    if args.mode in {MODE_PRCC, MODE_JOINT}:
        jobs.append(_prcc_eval_job(args))
    return jobs


def primary_eval_metric(eval_results, metric_name: str, variant: str, best_dataset: str = "auto") -> float:
    target = _primary_job_name(eval_results, best_dataset)
    for job, metrics in eval_results:
        if job.name == target:
            return _variant_metric(metrics, metric_name, variant)
    raise ValueError(f"best_dataset={target} was not evaluated")


def _prcc_eval_job(args: Namespace) -> EvalJob:
    if int(getattr(args, "prcc_dev_identities", 0)) > 0:
        return EvalJob(MODE_PRCC_DEV, args.prcc_root, PROTOCOL_CLOTH_CHANGE)
    return EvalJob(MODE_PRCC, args.prcc_root, PROTOCOL_CLOTH_CHANGE)


def _primary_job_name(eval_results, best_dataset: str) -> str:
    if best_dataset != "auto":
        return best_dataset
    names = [job.name for job, _ in eval_results]
    if MODE_PRCC_DEV in names:
        return MODE_PRCC_DEV
    if MODE_PRCC in names:
        return MODE_PRCC
    return names[0]


def _variant_metric(metrics: dict[str, dict[str, float]], metric_name: str, variant: str) -> float:
    if variant not in metrics:
        raise ValueError(f"Unknown evaluation variant: {variant}")
    if metric_name not in metrics[variant]:
        raise ValueError(f"Unknown evaluation metric: {metric_name}")
    return metrics[variant][metric_name]


def evaluate_checkpoint(args: Namespace) -> None:
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    protocol = _checkpoint_protocol(args.dataset)
    metrics = validate_dataset(model, args.root, args.dataset, protocol, device, args)
    print_metrics(metrics)


def _checkpoint_protocol(dataset: str) -> str:
    if dataset in {MODE_PRCC, MODE_PRCC_DEV}:
        return PROTOCOL_CLOTH_CHANGE
    return PROTOCOL_STANDARD


def load_model(checkpoint_path: str, device: torch.device) -> PedestrianReIDNet:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint.get("model_config", {})
    model = PedestrianReIDNet(
        int(checkpoint["num_classes"]),
        embedding_dim=int(model_config.get("embedding_dim", EMBEDDING_DIM)),
        num_clothes_classes=int(checkpoint["num_clothes_classes"]),
        use_part_branch=bool(model_config.get("use_part_branch", False)),
        num_parts=int(model_config.get("num_parts", DEFAULT_NUM_PARTS)),
        part_embedding_dim=int(model_config.get("part_embedding_dim", PART_EMBEDDING_DIM)),
        combined_global_weight=float(model_config.get("combined_global_weight", DEFAULT_COMBINED_GLOBAL_WEIGHT)),
        combined_part_weight=float(model_config.get("combined_part_weight", DEFAULT_COMBINED_PART_WEIGHT)),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def print_metrics(metrics: dict[str, dict[str, float]], prefix: str = "") -> None:
    for variant, values in metrics.items():
        label = f"{prefix}/{variant}" if prefix else variant
        print(f"{label} rank1={values['rank1']:.4f} rank5={values['rank5']:.4f} mAP={values['mAP']:.4f}")


def _run_eval_job(model, job: EvalJob, device: torch.device, args: Namespace):
    metrics = validate_dataset(model, job.root, job.name, job.protocol, device, args)
    print_metrics(metrics, prefix=job.name)
    return job, metrics


def _validate_variant(model, root: str, name: str, variant: str, gallery_bank, protocol: str, device, args):
    query_loader = build_eval_loader(root, name, "query", variant, args)
    query_bank = extract_feature_bank(model, query_loader, device, args.feature_key)
    return evaluate_reid(query_bank, gallery_bank, protocol)
