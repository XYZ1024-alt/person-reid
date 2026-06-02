from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import random

import torch
import torch.nn.functional as F
from tqdm import tqdm

from robust_person_reid.builders import build_train_loader, build_training_dataset
from robust_person_reid.engine.evaluator import evaluate_enabled_datasets, primary_rank1
from robust_person_reid.modules.losses import batch_hard_triplet_loss
from robust_person_reid.modules.model import RobustPersonReIDNet


CHECKPOINT_LAST = "last.pth"
CHECKPOINT_BEST = "best.pth"
NO_CAL_LOSS = 0.0


def train_from_args(args: Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device)
    dataset = build_training_dataset(args)
    validate_training_dataset(dataset)
    loader = build_train_loader(dataset, args)
    _require_cal_labels(dataset.num_clothes_classes, args.cal_weight)
    model = RobustPersonReIDNet(dataset.num_classes, num_clothes_classes=dataset.num_clothes_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_epoch = load_checkpoint(args.resume, model, optimizer)
    run_training(model, loader, optimizer, start_epoch, dataset, device, args)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, device: torch.device, args: Namespace) -> dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "ce": 0.0, "triplet": 0.0, "cal": 0.0}
    progress = tqdm(loader, desc="batches", unit="batch")
    for batch in progress:
        loss, ce_loss, triplet, cal_loss = _train_batch(model, batch, optimizer, device, args)
        _accumulate(totals, loss.item(), ce_loss.item(), triplet.item(), cal_loss.item())
        progress.set_postfix(_batch_metrics(loss, ce_loss, triplet, cal_loss))
    return {key: value / len(loader) for key, value in totals.items()}


def validate_training_dataset(dataset) -> None:
    _validate_contiguous_targets(_target_values(dataset.samples, "label"), dataset.num_classes, "identity label")
    clothes = _target_values(dataset.samples, "clothes_id")
    known_clothes = [value for value in clothes if value >= 0]
    if known_clothes:
        _validate_contiguous_targets(known_clothes, dataset.num_clothes_classes, "clothes label")


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


def run_training(model, loader, optimizer, start_epoch: int, dataset, device, args) -> None:
    best_rank1 = 0.0
    for epoch in range(start_epoch, args.epochs):
        print(f"epoch={epoch + 1}/{args.epochs}")
        metrics = train_one_epoch(model, loader, optimizer, device, args)
        _print_epoch(epoch, metrics)
        if (epoch + 1) % args.eval_period == 0 or epoch + 1 == args.epochs:
            best_rank1 = _evaluate_and_save(model, optimizer, epoch, dataset, best_rank1, device, args)


def _train_batch(model, batch, optimizer, device: torch.device, args: Namespace):
    images = batch["image"].to(device)
    labels = batch["label"]
    clothes_labels = batch["clothes_label"]
    outputs = model(images)
    _validate_batch_targets(labels, outputs["logits"].size(1), "identity label")
    _validate_batch_clothes_targets(clothes_labels, outputs, args.cal_weight)
    labels = labels.to(device)
    clothes_labels = clothes_labels.to(device)
    ce_loss = F.cross_entropy(outputs["logits"], labels)
    triplet = batch_hard_triplet_loss(outputs["features"], labels, args.triplet_margin)
    cal_loss = _cal_loss(outputs, clothes_labels, args.cal_weight)
    loss = ce_loss + args.triplet_weight * triplet + args.cal_weight * cal_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss, ce_loss, triplet, cal_loss


def _evaluate_and_save(model, optimizer, epoch: int, dataset, best: float, device, args) -> float:
    eval_metrics = evaluate_enabled_datasets(model, device, args)
    selected_rank1 = primary_rank1(eval_metrics)
    output_dir = Path(args.output_dir)
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


def _batch_metrics(loss, ce_loss, triplet, cal_loss) -> dict[str, str]:
    return {
        "loss": f"{loss.item():.4f}",
        "ce": f"{ce_loss.item():.4f}",
        "tri": f"{triplet.item():.4f}",
        "cal": f"{cal_loss.item():.4f}",
    }


def _cal_loss(outputs: dict[str, torch.Tensor], clothes_labels: torch.Tensor, cal_weight: float) -> torch.Tensor:
    if cal_weight <= NO_CAL_LOSS:
        return torch.zeros((), device=clothes_labels.device)
    known = clothes_labels.ge(0)
    if not known.any():
        return torch.zeros((), device=clothes_labels.device)
    return F.cross_entropy(outputs["clothes_logits"][known], clothes_labels[known])


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
        f"triplet={metrics['triplet']:.4f} cal={metrics['cal']:.4f}"
    )


def _require_cal_labels(num_clothes_classes: int, cal_weight: float) -> None:
    if cal_weight > NO_CAL_LOSS and num_clothes_classes <= 0:
        raise ValueError("CAL requires clothes labels; use PRCC or joint mode, or set --cal-weight 0")


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
