from __future__ import annotations

from argparse import Namespace
from contextlib import nullcontext
import csv
from pathlib import Path
import random

import torch
import torch.nn.functional as F
from tqdm import tqdm

from robust_person_reid.builders import build_train_loader, build_training_dataset
from robust_person_reid.engine.evaluator import evaluate_enabled_datasets, primary_rank1
from robust_person_reid.modules.losses import batch_hard_triplet_loss
from robust_person_reid.modules.model import RobustPersonReIDNet, load_imagenet_pretrained_backbone


CHECKPOINT_LAST = "last.pth"
CHECKPOINT_BEST = "best.pth"
TRAIN_METRICS_CSV = "training_metrics.csv"
EVAL_METRICS_CSV = "evaluation_metrics.csv"
TRAIN_METRIC_FIELDS = [
    "epoch",
    "loss",
    "ce",
    "triplet",
    "cal",
    "sketch",
    "consistency",
    "effective_cal_weight",
    "effective_sketch_consistency_weight",
]
EVAL_METRIC_FIELDS = ["rank1", "rank2", "rank3", "rank4", "rank5", "mAP"]
NO_CAL_LOSS = 0.0
NO_SKETCH_LOSS = 0.0
PRECISION_FP16 = "fp16"
PRECISION_FP32 = "fp32"
CUDA_DEVICE_TYPE = "cuda"
PARTIAL_PRETRAIN_PREFIXES = ("backbone.", "embedding.", "bnneck.")


def train_from_args(args: Namespace) -> None:
    set_seed(args.seed)
    validate_training_args(args)
    device = torch.device(args.device)
    dataset = build_training_dataset(args)
    validate_training_dataset(dataset)
    loader = build_train_loader(dataset, args)
    _require_cal_labels(dataset.num_clothes_classes, args.cal_weight)
    model = RobustPersonReIDNet(dataset.num_classes, num_clothes_classes=dataset.num_clothes_classes).to(device)
    initialize_model_weights(model, args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = _build_grad_scaler(args, device)
    start_epoch = load_checkpoint(args.resume, model, optimizer)
    model = configure_multi_gpu(model, args, device)
    run_training(model, loader, optimizer, scaler, start_epoch, dataset, device, args)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def initialize_model_weights(model: RobustPersonReIDNet, args: Namespace) -> None:
    if args.resume:
        return
    if args.pretrained_checkpoint:
        load_partial_pretrained_checkpoint(model, args.pretrained_checkpoint)
        return
    load_imagenet_pretrained_backbone(model.backbone)


def load_partial_pretrained_checkpoint(model: RobustPersonReIDNet, path: str) -> None:
    checkpoint = torch.load(path, map_location="cpu")
    source = checkpoint["model"]
    target = model.state_dict()
    selected = _partial_pretrained_state(source, target)
    if not selected:
        raise ValueError(f"No compatible partial pretrained parameters found in {path}")
    target.update(selected)
    model.load_state_dict(target)
    print(f"Loaded partial pretrained parameters: {len(selected)} from {path}")
    print(f"Partial pretrained prefixes: {PARTIAL_PRETRAIN_PREFIXES}")


def _partial_pretrained_state(source: dict[str, torch.Tensor], target: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    selected = {}
    for key, value in source.items():
        clean_key = _strip_module_prefix(key)
        if clean_key in target and _is_partial_pretrain_key(clean_key) and target[clean_key].shape == value.shape:
            selected[clean_key] = value
    return selected


def _strip_module_prefix(key: str) -> str:
    if key.startswith("module."):
        return key.removeprefix("module.")
    return key


def _is_partial_pretrain_key(key: str) -> bool:
    return key.startswith(PARTIAL_PRETRAIN_PREFIXES)


def _build_grad_scaler(args: Namespace, device: torch.device):
    enabled = _use_amp(args, device)
    return torch.amp.GradScaler(CUDA_DEVICE_TYPE, enabled=enabled)


def _autocast_context(args: Namespace, device: torch.device):
    if not _use_amp(args, device):
        return nullcontext()
    return torch.amp.autocast(CUDA_DEVICE_TYPE, dtype=torch.float16)


def _use_amp(args: Namespace, device: torch.device) -> bool:
    return args.precision == PRECISION_FP16 and device.type == CUDA_DEVICE_TYPE


def configure_multi_gpu(model: torch.nn.Module, args: Namespace, device: torch.device) -> torch.nn.Module:
    if not args.multi_gpu:
        return model
    gpu_count = torch.cuda.device_count()
    print(f"multi_gpu=True data_parallel_gpus={gpu_count} primary_device={device}")
    return torch.nn.DataParallel(model)


def train_one_epoch(model, loader, optimizer, scaler, device: torch.device, args: Namespace, epoch: int) -> dict[str, float]:
    model.train()
    totals = _empty_epoch_totals()
    effective_cal_weight = _effective_cal_weight(args, epoch)
    effective_consistency_weight = _effective_sketch_consistency_weight(args, epoch)
    progress = tqdm(loader, desc="batches", unit="batch")
    for batch in progress:
        losses = _train_batch(model, batch, optimizer, scaler, device, args, effective_cal_weight, effective_consistency_weight)
        _accumulate(totals, losses)
        progress.set_postfix(_batch_metrics(losses, effective_cal_weight, effective_consistency_weight))
    metrics = {key: value / len(loader) for key, value in totals.items()}
    metrics["effective_cal_weight"] = effective_cal_weight
    metrics["effective_sketch_consistency_weight"] = effective_consistency_weight
    return metrics


def validate_training_dataset(dataset) -> None:
    _validate_contiguous_targets(_target_values(dataset.samples, "label"), dataset.num_classes, "identity label")
    clothes = _target_values(dataset.samples, "clothes_id")
    known_clothes = [value for value in clothes if value >= 0]
    if known_clothes:
        _validate_contiguous_targets(known_clothes, dataset.num_clothes_classes, "clothes label")


def validate_training_args(args: Namespace) -> None:
    if args.cal_warmup_epochs < 0:
        raise ValueError("cal_warmup_epochs must be >= 0")
    if args.cal_ramp_epochs < 0:
        raise ValueError("cal_ramp_epochs must be >= 0")
    if args.precision == PRECISION_FP16 and not args.device.startswith(CUDA_DEVICE_TYPE):
        raise ValueError("fp16 precision requires a CUDA device")
    if args.sketch_loss_weight < 0:
        raise ValueError("sketch_loss_weight must be >= 0")
    if args.rgb_sketch_consistency_weight < 0:
        raise ValueError("rgb_sketch_consistency_weight must be >= 0")
    if args.sketch_warmup_epochs < 0:
        raise ValueError("sketch_warmup_epochs must be >= 0")
    if args.sketch_ramp_epochs < 0:
        raise ValueError("sketch_ramp_epochs must be >= 0")
    if args.resume and args.pretrained_checkpoint:
        raise ValueError("--resume and --pretrained-checkpoint are mutually exclusive")
    _validate_multi_gpu_args(args)


def save_checkpoint(path: Path, model, optimizer, epoch: int, metric: float, dataset) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": _unwrap_model(model).state_dict(), "optimizer": optimizer.state_dict()}
    payload.update(_checkpoint_metadata(epoch, metric, dataset))
    torch.save(payload, path)


def load_checkpoint(path: str, model, optimizer) -> int:
    if not path:
        return 0
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint["epoch"]) + 1


def run_training(model, loader, optimizer, scaler, start_epoch: int, dataset, device, args) -> None:
    best_rank1 = 0.0
    output_dir = Path(args.output_dir)
    _initialize_metric_files(output_dir, start_epoch)
    print(f"precision={args.precision} pin_memory={args.pin_memory} persistent_workers={_loader_has_persistent_workers(loader)}")
    for epoch in range(start_epoch, args.epochs):
        print(f"epoch={epoch + 1}/{args.epochs}")
        metrics = train_one_epoch(model, loader, optimizer, scaler, device, args, epoch)
        _print_epoch(epoch, metrics)
        _write_train_metrics(output_dir, epoch, metrics)
        if (epoch + 1) % args.eval_period == 0 or epoch + 1 == args.epochs:
            best_rank1 = _evaluate_and_save(model, optimizer, epoch, dataset, best_rank1, device, args)


def _train_batch(
    model,
    batch,
    optimizer,
    scaler,
    device: torch.device,
    args: Namespace,
    effective_cal_weight: float,
    effective_consistency_weight: float,
):
    images = batch["image"].to(device, non_blocking=args.pin_memory)
    labels = batch["label"]
    clothes_labels = batch["clothes_label"]
    has_sketch = batch["has_sketch"].bool()
    with _autocast_context(args, device):
        outputs = model(images)
    _validate_batch_targets(labels, outputs["logits"].size(1), "identity label")
    _validate_batch_clothes_targets(clothes_labels, outputs, effective_cal_weight)
    labels = labels.to(device, non_blocking=args.pin_memory)
    clothes_labels = clothes_labels.to(device, non_blocking=args.pin_memory)
    ce_loss = F.cross_entropy(outputs["logits"].float(), labels)
    triplet = batch_hard_triplet_loss(outputs["features"].float(), labels, args.triplet_margin)
    cal_loss = _cal_loss(outputs, clothes_labels, effective_cal_weight)
    sketch_loss, consistency_loss = _sketch_losses(model, batch, outputs, labels, has_sketch, device, args, effective_consistency_weight)
    loss = _total_loss(args, ce_loss, triplet, cal_loss, sketch_loss, consistency_loss, effective_cal_weight, effective_consistency_weight)
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return {
        "loss": loss,
        "ce": ce_loss,
        "triplet": triplet,
        "cal": cal_loss,
        "sketch": sketch_loss,
        "consistency": consistency_loss,
    }


def _sketch_losses(model, batch, rgb_outputs, labels: torch.Tensor, has_sketch: torch.Tensor, device, args, consistency_weight: float):
    if not _use_sketch_path(args, has_sketch, consistency_weight):
        return _zero_loss(device), _zero_loss(device)
    device_mask = has_sketch.to(device, non_blocking=args.pin_memory)
    sketch_images = batch["sketch_image"][has_sketch].to(device, non_blocking=args.pin_memory)
    sketch_loss = _sketch_identity_loss(model, sketch_images, labels[device_mask], device, args)
    consistency = _sketch_consistency_loss(model, sketch_images, rgb_outputs["features"][device_mask], device, args, consistency_weight)
    return sketch_loss, consistency


def _use_sketch_path(args: Namespace, has_sketch: torch.Tensor, consistency_weight: float) -> bool:
    if not args.use_prcc_sketch:
        return False
    if args.sketch_loss_weight <= NO_SKETCH_LOSS and consistency_weight <= NO_SKETCH_LOSS:
        return False
    return bool(has_sketch.any().item())


def _sketch_identity_loss(model, sketch_images: torch.Tensor, sketch_labels: torch.Tensor, device, args) -> torch.Tensor:
    if args.sketch_loss_weight <= NO_SKETCH_LOSS:
        return _zero_loss(device)
    with _autocast_context(args, device):
        sketch_outputs = model(sketch_images)
    sketch_ce = F.cross_entropy(sketch_outputs["logits"].float(), sketch_labels)
    sketch_triplet = _optional_triplet(sketch_outputs["features"].float(), sketch_labels, args.triplet_margin)
    return sketch_ce + args.triplet_weight * sketch_triplet


def _sketch_consistency_loss(model, sketch_images: torch.Tensor, rgb_features: torch.Tensor, device, args, weight: float) -> torch.Tensor:
    if weight <= NO_SKETCH_LOSS:
        return _zero_loss(device)
    sketch_outputs = _extract_sketch_targets(model, sketch_images, device, args)
    return _consistency_loss(rgb_features.float(), sketch_outputs["features"].float())


def _extract_sketch_targets(model, sketch_images: torch.Tensor, device, args) -> dict[str, torch.Tensor]:
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            with _autocast_context(args, device):
                return model(sketch_images)
    finally:
        if was_training:
            model.train()


def _optional_triplet(features: torch.Tensor, labels: torch.Tensor, margin: float) -> torch.Tensor:
    if len(labels.unique()) < 2:
        return _zero_loss(features.device)
    return batch_hard_triplet_loss(features, labels, margin)


def _consistency_loss(rgb_features: torch.Tensor, sketch_features: torch.Tensor) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(rgb_features, sketch_features, dim=1).mean()


def _total_loss(args, ce_loss, triplet, cal_loss, sketch_loss, consistency_loss, effective_cal_weight: float, consistency_weight: float):
    loss = ce_loss + args.triplet_weight * triplet + effective_cal_weight * cal_loss
    loss = loss + args.sketch_loss_weight * sketch_loss
    return loss + consistency_weight * consistency_loss


def _evaluate_and_save(model, optimizer, epoch: int, dataset, best: float, device, args) -> float:
    eval_metrics = evaluate_enabled_datasets(model, device, args)
    selected_rank1 = primary_rank1(eval_metrics)
    output_dir = Path(args.output_dir)
    _write_eval_metrics(output_dir, epoch, eval_metrics)
    save_checkpoint(output_dir / CHECKPOINT_LAST, model, optimizer, epoch, selected_rank1, dataset)
    if selected_rank1 <= best:
        return best
    save_checkpoint(output_dir / CHECKPOINT_BEST, model, optimizer, epoch, selected_rank1, dataset)
    return selected_rank1


def _accumulate(totals: dict[str, float], losses: dict[str, torch.Tensor]) -> None:
    for key in totals:
        totals[key] += losses[key].item()


def _empty_epoch_totals() -> dict[str, float]:
    return {"loss": 0.0, "ce": 0.0, "triplet": 0.0, "cal": 0.0, "sketch": 0.0, "consistency": 0.0}


def _zero_loss(device: torch.device) -> torch.Tensor:
    return torch.zeros((), device=device)


def _initialize_metric_files(output_dir: Path, start_epoch: int) -> None:
    if start_epoch > 0:
        _ensure_train_metric_header(output_dir / TRAIN_METRICS_CSV)
        _ensure_eval_metric_header(output_dir / EVAL_METRICS_CSV)
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_header(output_dir / TRAIN_METRICS_CSV, TRAIN_METRIC_FIELDS)
    _write_header(output_dir / EVAL_METRICS_CSV, _eval_fieldnames())


def _write_header(path: Path, fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=fieldnames).writeheader()


def _write_train_metrics(output_dir: Path, epoch: int, metrics: dict[str, float]) -> None:
    row = {"epoch": epoch + 1, **metrics}
    _append_row(output_dir / TRAIN_METRICS_CSV, TRAIN_METRIC_FIELDS, row)


def _write_eval_metrics(output_dir: Path, epoch: int, eval_results) -> None:
    for job, metrics_by_variant in eval_results:
        for variant, metrics in metrics_by_variant.items():
            row = {"epoch": epoch + 1, "dataset": job.name, "variant": variant, **metrics}
            _append_row(output_dir / EVAL_METRICS_CSV, _eval_fieldnames(), row)


def _append_row(path: Path, fieldnames: list[str], row: dict) -> None:
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writerow(row)


def _eval_fieldnames() -> list[str]:
    return ["epoch", "dataset", "variant", *EVAL_METRIC_FIELDS]


def _ensure_train_metric_header(path: Path) -> None:
    if not path.exists():
        _write_header(path, TRAIN_METRIC_FIELDS)
        return
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if rows and set(TRAIN_METRIC_FIELDS).issubset(rows[0].keys()):
        return
    _rewrite_metric_file(path, rows, TRAIN_METRIC_FIELDS)


def _ensure_eval_metric_header(path: Path) -> None:
    if not path.exists():
        _write_header(path, _eval_fieldnames())
        return
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if rows and set(_eval_fieldnames()).issubset(rows[0].keys()):
        return
    _rewrite_metric_file(path, rows, _eval_fieldnames())


def _rewrite_metric_file(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _batch_metrics(losses: dict[str, torch.Tensor], cal_weight: float, consistency_weight: float) -> dict[str, str]:
    return {
        "loss": f"{losses['loss'].item():.4f}",
        "ce": f"{losses['ce'].item():.4f}",
        "tri": f"{losses['triplet'].item():.4f}",
        "cal": f"{losses['cal'].item():.4f}",
        "sk": f"{losses['sketch'].item():.4f}",
        "con": f"{losses['consistency'].item():.4f}",
        "cal_w": f"{cal_weight:.4f}",
        "con_w": f"{consistency_weight:.4f}",
    }


def _cal_loss(outputs: dict[str, torch.Tensor], clothes_labels: torch.Tensor, cal_weight: float) -> torch.Tensor:
    if cal_weight <= NO_CAL_LOSS:
        return torch.zeros((), device=clothes_labels.device)
    known = clothes_labels.ge(0)
    if not known.any():
        return torch.zeros((), device=clothes_labels.device)
    return F.cross_entropy(outputs["clothes_logits"][known].float(), clothes_labels[known])


def _checkpoint_metadata(epoch: int, metric: float, dataset) -> dict[str, int | float]:
    return {
        "epoch": epoch,
        "rank1": metric,
        "num_classes": dataset.num_classes,
        "num_clothes_classes": dataset.num_clothes_classes,
    }


def _print_epoch(epoch: int, metrics: dict[str, float]) -> None:
    print(
        f"epoch={epoch + 1} loss={metrics['loss']:.4f} ce={metrics['ce']:.4f} "
        f"triplet={metrics['triplet']:.4f} cal={metrics['cal']:.4f} "
        f"sketch={metrics['sketch']:.4f} consistency={metrics['consistency']:.4f} "
        f"cal_w={metrics['effective_cal_weight']:.4f} "
        f"con_w={metrics['effective_sketch_consistency_weight']:.4f}"
    )


def _require_cal_labels(num_clothes_classes: int, cal_weight: float) -> None:
    if cal_weight > NO_CAL_LOSS and num_clothes_classes <= 0:
        raise ValueError("CAL requires clothes labels; use PRCC or joint mode, or set --cal-weight 0")


def _effective_cal_weight(args: Namespace, epoch: int) -> float:
    if args.cal_weight <= NO_CAL_LOSS or epoch < args.cal_warmup_epochs:
        return NO_CAL_LOSS
    ramp_index = epoch - args.cal_warmup_epochs + 1
    if args.cal_ramp_epochs == 0 or ramp_index >= args.cal_ramp_epochs:
        return args.cal_weight
    return args.cal_weight * ramp_index / args.cal_ramp_epochs


def _effective_sketch_consistency_weight(args: Namespace, epoch: int) -> float:
    if args.rgb_sketch_consistency_weight <= NO_SKETCH_LOSS or epoch < args.sketch_warmup_epochs:
        return NO_SKETCH_LOSS
    ramp_index = epoch - args.sketch_warmup_epochs + 1
    if args.sketch_ramp_epochs == 0 or ramp_index >= args.sketch_ramp_epochs:
        return args.rgb_sketch_consistency_weight
    return args.rgb_sketch_consistency_weight * ramp_index / args.sketch_ramp_epochs


def _loader_has_persistent_workers(loader) -> bool:
    return bool(getattr(loader, "persistent_workers", False))


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if isinstance(model, torch.nn.DataParallel):
        return model.module
    return model


def _validate_multi_gpu_args(args: Namespace) -> None:
    if not args.multi_gpu:
        return
    if not args.device.startswith(CUDA_DEVICE_TYPE):
        raise ValueError("--multi-gpu requires a CUDA device")
    if args.device not in {CUDA_DEVICE_TYPE, f"{CUDA_DEVICE_TYPE}:0"}:
        raise ValueError("--multi-gpu requires --device cuda or --device cuda:0")
    if torch.cuda.device_count() < 2:
        raise ValueError("--multi-gpu requires at least 2 visible CUDA GPUs")


def _target_values(samples, field_name: str) -> list[int]:
    return [int(getattr(sample, field_name)) for sample in samples if not sample.is_junk]


def _validate_contiguous_targets(values: list[int], class_count: int, name: str) -> None:
    if not values:
        raise ValueError(f"No valid {name} values found")
    expected = set(range(class_count))
    actual = set(values)
    if actual != expected:
        raise ValueError(f"{name} values must be contiguous 0..{class_count - 1}; got min={min(actual)} max={max(actual)}")


def _validate_batch_targets(targets: torch.Tensor, class_count: int, name: str) -> None:
    minimum = int(targets.min().item())
    maximum = int(targets.max().item())
    if minimum < 0 or maximum >= class_count:
        raise ValueError(f"{name} out of range for {class_count} classes: min={minimum} max={maximum}")


def _validate_batch_clothes_targets(clothes_labels: torch.Tensor, outputs: dict[str, torch.Tensor], cal_weight: float) -> None:
    if cal_weight <= NO_CAL_LOSS:
        return
    known = clothes_labels.ge(0)
    if not known.any():
        return
    _validate_batch_targets(clothes_labels[known], outputs["clothes_logits"].size(1), "clothes label")
