from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass

import torch

from robust_person_reid.builders import MODE_JOINT, MODE_MARKET, MODE_PRCC, build_eval_loader
from robust_person_reid.data.transforms import VARIANT_DARK, VARIANT_OCCLUDED, VARIANT_STANDARD
from robust_person_reid.modules.metrics import PROTOCOL_CLOTH_CHANGE, PROTOCOL_STANDARD
from robust_person_reid.modules.metrics import evaluate_reid, extract_feature_bank
from robust_person_reid.modules.model import RobustPersonReIDNet


@dataclass(frozen=True)
class EvalJob:
    name: str
    root: str
    protocol: str


def validate_dataset(model, root: str, name: str, protocol: str, device: torch.device, args: Namespace):
    gallery_loader = build_eval_loader(root, name, "gallery", VARIANT_STANDARD, args)
    gallery_bank = extract_feature_bank(model, gallery_loader, device)
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
        jobs.append(EvalJob(MODE_PRCC, args.prcc_root, PROTOCOL_CLOTH_CHANGE))
    return jobs


def primary_eval_metric(eval_results, metric_name: str, variant: str) -> float:
    for job, metrics in eval_results:
        if job.name == MODE_PRCC:
            return _variant_metric(metrics, metric_name, variant)
    return _variant_metric(eval_results[0][1], metric_name, variant)


def _variant_metric(metrics: dict[str, dict[str, float]], metric_name: str, variant: str) -> float:
    if variant not in metrics:
        raise ValueError(f"Unknown evaluation variant: {variant}")
    if metric_name not in metrics[variant]:
        raise ValueError(f"Unknown evaluation metric: {metric_name}")
    return metrics[variant][metric_name]


def evaluate_checkpoint(args: Namespace) -> None:
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    protocol = PROTOCOL_CLOTH_CHANGE if args.dataset == MODE_PRCC else PROTOCOL_STANDARD
    metrics = validate_dataset(model, args.root, args.dataset, protocol, device, args)
    print_metrics(metrics)


def load_model(checkpoint_path: str, device: torch.device) -> RobustPersonReIDNet:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = RobustPersonReIDNet(
        int(checkpoint["num_classes"]),
        num_clothes_classes=int(checkpoint["num_clothes_classes"]),
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
    query_bank = extract_feature_bank(model, query_loader, device)
    return evaluate_reid(query_bank, gallery_bank, protocol)
