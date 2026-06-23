from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass

import torch

from pedestrian_reid.builders import MODE_MARKET, MODE_PRCC, MODE_PRCC_DEV, build_eval_loader
from pedestrian_reid.data.transforms import VARIANT_DARK, VARIANT_OCCLUDED, VARIANT_STANDARD
from pedestrian_reid.modules.metrics import PROTOCOL_CLOTH_CHANGE, PROTOCOL_SAME_CLOTHES, PROTOCOL_STANDARD
from pedestrian_reid.modules.metrics import evaluate_reid, extract_feature_bank
from pedestrian_reid.modules.model import (
    DEFAULT_COMBINED_GLOBAL_WEIGHT,
    DEFAULT_COMBINED_PART_WEIGHT,
    DEFAULT_NUM_PARTS,
    EMBEDDING_DIM,
    PART_EMBEDDING_DIM,
    PedestrianReIDNet,
)


REQUIRED_MODEL_CONFIG_KEYS = ("backbone_type", "use_sketch_fusion")


@dataclass(frozen=True)
class EvalJob:
    name: str
    root: str
    protocol: str
    dataset_name: str = ""


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
    if args.mode == MODE_MARKET:
        jobs.append(EvalJob(MODE_MARKET, args.market_root, PROTOCOL_STANDARD))
    if args.mode == MODE_PRCC:
        jobs.extend(_prcc_eval_jobs(args))
    return jobs


def primary_eval_metric(eval_results, metric_name: str, variant: str, best_dataset: str = "auto") -> float:
    target = _primary_job_name(eval_results, best_dataset)
    for job, metrics in eval_results:
        if job.name == target:
            return _variant_metric(metrics, metric_name, variant)
    raise ValueError(f"best_dataset={target} was not evaluated")


def _prcc_eval_jobs(args: Namespace) -> list[EvalJob]:
    if int(getattr(args, "prcc_dev_identities", 0)) > 0:
        return _prcc_protocol_jobs(MODE_PRCC_DEV, args.prcc_root)
    return _prcc_protocol_jobs(MODE_PRCC, args.prcc_root)


def _prcc_protocol_jobs(dataset_name: str, root: str) -> list[EvalJob]:
    return [
        EvalJob(f"{dataset_name}_same_clothes", root, PROTOCOL_SAME_CLOTHES, dataset_name),
        EvalJob(f"{dataset_name}_cloth_change", root, PROTOCOL_CLOTH_CHANGE, dataset_name),
    ]


def _primary_job_name(eval_results, best_dataset: str) -> str:
    names = [job.name for job, _ in eval_results]
    if best_dataset != "auto":
        return _requested_job_name(best_dataset, names)
    for dataset in (MODE_PRCC_DEV, MODE_PRCC):
        name = f"{dataset}_cloth_change"
        if name in names:
            return name
    return names[0]


def _requested_job_name(best_dataset: str, names: list[str]) -> str:
    if best_dataset in names:
        return best_dataset
    if best_dataset in {MODE_PRCC, MODE_PRCC_DEV}:
        return f"{best_dataset}_cloth_change"
    return best_dataset


def _variant_metric(metrics: dict[str, dict[str, float]], metric_name: str, variant: str) -> float:
    if variant not in metrics:
        raise ValueError(f"Unknown evaluation variant: {variant}")
    if metric_name not in metrics[variant]:
        raise ValueError(f"Unknown evaluation metric: {metric_name}")
    return metrics[variant][metric_name]


def evaluate_checkpoint(args: Namespace) -> None:
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    jobs = _checkpoint_eval_jobs(args)
    for job in jobs:
        metrics = validate_dataset(model, job.root, _job_dataset_name(job), job.protocol, device, args)
        print_metrics(metrics, prefix=_checkpoint_metric_prefix(job, jobs))


def _checkpoint_eval_jobs(args: Namespace) -> list[EvalJob]:
    if args.dataset in {MODE_PRCC, MODE_PRCC_DEV}:
        return _prcc_protocol_jobs(args.dataset, args.root)
    return [EvalJob(args.dataset, args.root, PROTOCOL_STANDARD)]


def _checkpoint_metric_prefix(job: EvalJob, jobs: list[EvalJob]) -> str:
    if len(jobs) == 1 and job.name == MODE_MARKET:
        return ""
    return job.name


def load_model(checkpoint_path: str, device: torch.device) -> PedestrianReIDNet:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = _checkpoint_model_config(checkpoint, checkpoint_path)
    model = PedestrianReIDNet(
        int(checkpoint["num_classes"]),
        embedding_dim=int(model_config.get("embedding_dim", EMBEDDING_DIM)),
        num_clothes_classes=int(checkpoint["num_clothes_classes"]),
        use_part_branch=bool(model_config.get("use_part_branch", False)),
        num_parts=int(model_config.get("num_parts", DEFAULT_NUM_PARTS)),
        part_embedding_dim=int(model_config.get("part_embedding_dim", PART_EMBEDDING_DIM)),
        combined_global_weight=float(model_config.get("combined_global_weight", DEFAULT_COMBINED_GLOBAL_WEIGHT)),
        combined_part_weight=float(model_config.get("combined_part_weight", DEFAULT_COMBINED_PART_WEIGHT)),
        use_dual_classifier=bool(model_config.get("use_dual_classifier", False)),
        num_market_classes=int(checkpoint.get("num_market_classes", 0)),
        num_prcc_classes=int(checkpoint.get("num_prcc_classes", 0)),
        use_domain_adversarial=bool(model_config.get("use_domain_adversarial", False)),
        backbone_type=str(model_config["backbone_type"]),
        use_sketch_fusion=bool(model_config["use_sketch_fusion"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def _checkpoint_model_config(checkpoint: dict, checkpoint_path: str) -> dict:
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError(f"Checkpoint missing model_config: {checkpoint_path}")
    missing = [key for key in REQUIRED_MODEL_CONFIG_KEYS if key not in model_config]
    if missing:
        raise ValueError(f"Checkpoint missing required model_config keys {missing}: {checkpoint_path}")
    return model_config


def print_metrics(metrics: dict[str, dict[str, float]], prefix: str = "") -> None:
    for variant, values in metrics.items():
        label = f"{prefix}/{variant}" if prefix else variant
        print(f"{label} rank1={values['rank1']:.4f} rank5={values['rank5']:.4f} mAP={values['mAP']:.4f}")


def _run_eval_job(model, job: EvalJob, device: torch.device, args: Namespace):
    metrics = validate_dataset(model, job.root, _job_dataset_name(job), job.protocol, device, args)
    print_metrics(metrics, prefix=job.name)
    return job, metrics


def _job_dataset_name(job: EvalJob) -> str:
    return job.dataset_name or job.name


def _validate_variant(model, root: str, name: str, variant: str, gallery_bank, protocol: str, device, args):
    query_loader = build_eval_loader(root, name, "query", variant, args)
    query_bank = extract_feature_bank(model, query_loader, device, args.feature_key)
    return evaluate_reid(query_bank, gallery_bank, protocol)
