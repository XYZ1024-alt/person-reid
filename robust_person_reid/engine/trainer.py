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
TRAIN_METRIC_FIELDS = ["epoch", "loss", "ce", "triplet", "cal", "effective_cal_weight"]
EVAL_METRIC_FIELDS = ["rank1", "rank2", "rank3", "rank4", "rank5", "mAP"]
NO_CAL_LOSS = 0.0
PRECISION_FP16 = "fp16"
PRECISION_FP32 = "fp32"
CUDA_DEVICE_TYPE = "cuda"


def train_from_args(args: Namespace) -> None:
    set_seed(args.seed)
    validate_training_args(args)
    device = torch.device(args.device)
    dataset = build_training_dataset(args)
    validate_training_dataset(dataset)
    loader = build_train_loader(dataset, args)
    _require_cal_labels(dataset.num_clothes_classes, args.cal_weight)
    model = RobustPersonReIDNet(dataset.num_classes, num_clothes_classes=dataset.num_clothes_classes).to(device)
    initialize_backbone(model, args.resume)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = _build_grad_scaler(args, device)
    start_epoch = load_checkpoint(args.resume, model, optimizer)
    run_training(model, loader, optimizer, scaler, start_epoch, dataset, device, args)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def initialize_backbone(model: RobustPersonReIDNet, resume_path: str) -> None:
    if resume_path:
        return
    load_imagenet_pretrained_backbone(model.backbone)


def _build_grad_scaler(args: Namespace, device: torch.device):
    enabled = _use_amp(args, device)
    return torch.amp.GradScaler(CUDA_DEVICE_TYPE, enabled=enabled)


def _autocast_context(args: Namespace, device: torch.device):
    if not _use_amp(args, device):
        return nullcontext()
    return torch.amp.autocast(CUDA_DEVICE_TYPE, dtype=torch.float16)


def _use_amp(args: Namespace, device: torch.device) -> bool:
    return args.precision == PRECISION_FP16 and device.type == CUDA_DEVICE_TYPE


def train_one_epoch(model, loader, optimizer, scaler, device: torch.device, args: Namespace, epoch: int) -> dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "ce": 0.0, "triplet": 0.0, "cal": 0.0}
    effective_cal_weight = _effective_cal_weight(args, epoch)
    progress = tqdm(loader, desc="batches", unit="batch")
    for batch in progress:
        loss, ce_loss, triplet, cal_loss = _train_batch(model, batch, optimizer, scaler, device, args, effective_cal_weight)
        _accumulate(totals, loss.item(), ce_loss.item(), triplet.item(), cal_loss.item())
        progress.set_postfix(_batch_metrics(loss, ce_loss, triplet, cal_loss, effective_cal_weight))
    metrics = {key: value / len(loader) for key, value in totals.items()}
    metrics["effective_cal_weight"] = effective_cal_weight
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


def save_checkpoint(path: Path, model, optimizer, epoch: int, metric: float, dataset) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
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


def _train_batch(model, batch, optimizer, scaler, device: torch.device, args: Namespace, effective_cal_weight: float):
    images = batch["image"].to(device, non_blocking=args.pin_memory)
    labels = batch["label"]
    clothes_labels = batch["clothes_label"]
    with _autocast_context(args, device):
        outputs = model(images)
    _validate_batch_targets(labels, outputs["logits"].size(1), "identity label")
    _validate_batch_clothes_targets(clothes_labels, outputs, effective_cal_weight)
    labels = labels.to(device, non_blocking=args.pin_memory)
    clothes_labels = clothes_labels.to(device, non_blocking=args.pin_memory)
    ce_loss = F.cross_entropy(outputs["logits"].float(), labels)
    triplet = batch_hard_triplet_loss(outputs["features"].float(), labels, args.triplet_margin)
    cal_loss = _cal_loss(outputs, clothes_labels, effective_cal_weight)
    loss = ce_loss + args.triplet_weight * triplet + effective_cal_weight * cal_loss
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return loss, ce_loss, triplet, cal_loss


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


def _accumulate(totals: dict[str, float], loss: float, ce_loss: float, triplet: float, cal_loss: float) -> None:
    totals["loss"] += loss
    totals["ce"] += ce_loss
    totals["triplet"] += triplet
    totals["cal"] += cal_loss


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


def _batch_metrics(loss, ce_loss, triplet, cal_loss, cal_weight: float) -> dict[str, str]:
    return {
        "loss": f"{loss.item():.4f}",
        "ce": f"{ce_loss.item():.4f}",
        "tri": f"{triplet.item():.4f}",
        "cal": f"{cal_loss.item():.4f}",
        "cal_w": f"{cal_weight:.4f}",
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
        f"cal_w={metrics['effective_cal_weight']:.4f}"
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


def _loader_has_persistent_workers(loader) -> bool:
    return bool(getattr(loader, "persistent_workers", False))


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
