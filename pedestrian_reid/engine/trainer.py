from __future__ import annotations

from argparse import Namespace
from contextlib import nullcontext
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import random

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from pedestrian_reid.builders import build_train_loader, build_training_dataset
from pedestrian_reid.engine.evaluator import evaluate_enabled_datasets, primary_eval_metric
from pedestrian_reid.data.transforms import VARIANT_DARK, VARIANT_OCCLUDED, VARIANT_STANDARD
from pedestrian_reid.modules.losses import batch_hard_triplet_loss
from pedestrian_reid.modules.model import PedestrianReIDNet, load_imagenet_pretrained_backbone


CHECKPOINT_LAST = "last.pth"
CHECKPOINT_BEST = "best.pth"
RUN_CONFIG_JSON = "run_config.json"
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
BEST_METRIC_RANK1 = "rank1"
BEST_METRIC_MAP = "mAP"
BEST_METRIC_CHOICES = {BEST_METRIC_RANK1, BEST_METRIC_MAP}
BEST_VARIANT_CHOICES = {VARIANT_STANDARD, VARIANT_DARK, VARIANT_OCCLUDED}
NO_CAL_LOSS = 0.0
NO_SKETCH_LOSS = 0.0
PRECISION_FP16 = "fp16"
PRECISION_FP32 = "fp32"
CUDA_DEVICE_TYPE = "cuda"
DDP_BACKEND = "nccl"
DDP_FIND_UNUSED_AUTO = "auto"
DDP_FIND_UNUSED_TRUE = "true"
DDP_FIND_UNUSED_FALSE = "false"
ENV_RANK = "RANK"
ENV_WORLD_SIZE = "WORLD_SIZE"
ENV_LOCAL_RANK = "LOCAL_RANK"
FREEZABLE_BACKBONE_LAYERS = {"stem", "layer1", "layer2", "layer3", "layer4"}
PRETRAIN_SKIP_PREVIEW_LIMIT = 10
MIN_IMAGENET_PRETRAINED_TENSORS = 300
AUGMENT_PROBABILITY_ARGS = (
    "flip_probability",
    "color_jitter_probability",
    "random_grayscale_probability",
    "dark_augment_probability",
    "occlusion_augment_probability",
)


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool = False
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def train_from_args(args: Namespace) -> None:
    distributed = initialize_distributed(args)
    try:
        set_seed(args.seed + distributed.rank)
        validate_training_args(args, distributed)
        device = training_device(args, distributed)
        dataset = build_training_dataset(args)
        validate_training_dataset(dataset)
        loader = build_train_loader(dataset, args, distributed)
        _require_cal_labels(dataset.num_clothes_classes, args.cal_weight)
        model = PedestrianReIDNet(dataset.num_classes, num_clothes_classes=dataset.num_clothes_classes).to(device)
        pretrained_count = initialize_model_weights(model, args, distributed)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = build_lr_scheduler(optimizer, args)
        scaler = _build_grad_scaler(args, device)
        start_epoch = load_checkpoint(args.resume, model, optimizer, scheduler)
        model = configure_parallel_model(model, args, device, distributed)
        run_training(model, loader, optimizer, scheduler, scaler, start_epoch, dataset, device, args, distributed, pretrained_count)
    finally:
        cleanup_distributed(distributed)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def initialize_model_weights(model: PedestrianReIDNet, args: Namespace, distributed: DistributedContext) -> int | None:
    if args.resume:
        return None
    if args.pretrained_checkpoint:
        return load_compatible_pretrained_checkpoint(model, args.pretrained_checkpoint, distributed)
    loaded = load_imagenet_pretrained_backbone(model.backbone, verbose=distributed.is_main)
    _require_imagenet_pretrained_parameters(loaded)
    return loaded


def load_compatible_pretrained_checkpoint(model: PedestrianReIDNet, path: str, distributed: DistributedContext) -> int:
    checkpoint = torch.load(path, map_location="cpu")
    source = checkpoint["model"]
    target = model.state_dict()
    selected, skipped = _compatible_pretrained_state(source, target)
    if not selected:
        raise ValueError(f"No compatible pretrained parameters found in {path}")
    target.update(selected)
    model.load_state_dict(target)
    loaded_heads = _count_head_parameters(selected)
    rank_zero_print(distributed, f"Loaded compatible pretrained parameters: {len(selected)} from {path}")
    rank_zero_print(distributed, f"Loaded compatible classifier parameters: {loaded_heads}")
    _print_skipped_pretrained_parameters(distributed, skipped)
    return len(selected)


def _require_imagenet_pretrained_parameters(loaded: int) -> None:
    if loaded < MIN_IMAGENET_PRETRAINED_TENSORS:
        raise RuntimeError(f"ImageNet pretrained backbone load is incomplete: loaded_tensors={loaded}")


def _compatible_pretrained_state(
    source: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], list[str]]:
    selected = {}
    skipped = []
    for key, value in source.items():
        clean_key = _strip_module_prefix(key)
        if clean_key in target and target[clean_key].shape == value.shape:
            selected[clean_key] = value
            continue
        skipped.append(clean_key)
    return selected, skipped


def _strip_module_prefix(key: str) -> str:
    if key.startswith("module."):
        return key.removeprefix("module.")
    return key


def _count_head_parameters(source: dict[str, torch.Tensor]) -> int:
    return sum(1 for key in source if _is_head_key(_strip_module_prefix(key)))


def _is_head_key(key: str) -> bool:
    return key.startswith(("classifier.", "clothes_classifier."))


def _print_skipped_pretrained_parameters(distributed: DistributedContext, skipped: list[str]) -> None:
    if not skipped:
        return
    preview = ", ".join(skipped[:PRETRAIN_SKIP_PREVIEW_LIMIT])
    rank_zero_print(distributed, f"Skipped incompatible pretrained parameters: {len(skipped)}")
    rank_zero_print(distributed, f"Skipped incompatible preview: {preview}")


def _build_grad_scaler(args: Namespace, device: torch.device):
    enabled = _use_amp(args, device)
    return torch.amp.GradScaler(CUDA_DEVICE_TYPE, enabled=enabled)


def build_lr_scheduler(optimizer: torch.optim.Optimizer, args: Namespace):
    milestones = _lr_milestones(args.lr_milestones)
    return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=args.lr_gamma)


def _lr_milestones(raw: str) -> list[int]:
    if not raw.strip():
        return []
    milestones = [int(part.strip()) for part in raw.split(",") if part.strip()]
    _validate_lr_milestone_values(milestones)
    return milestones


def _validate_lr_milestone_values(milestones: list[int]) -> None:
    if any(milestone <= 0 for milestone in milestones):
        raise ValueError(f"lr_milestones must be positive epoch numbers: {milestones}")
    if len(set(milestones)) != len(milestones):
        raise ValueError(f"lr_milestones must not contain duplicates: {milestones}")
    if milestones != sorted(milestones):
        raise ValueError(f"lr_milestones must be sorted ascending: {milestones}")


def _autocast_context(args: Namespace, device: torch.device):
    if not _use_amp(args, device):
        return nullcontext()
    return torch.amp.autocast(CUDA_DEVICE_TYPE, dtype=torch.float16)


def _use_amp(args: Namespace, device: torch.device) -> bool:
    return args.precision == PRECISION_FP16 and device.type == CUDA_DEVICE_TYPE


def initialize_distributed(args: Namespace) -> DistributedContext:
    if not args.distributed:
        return DistributedContext()
    _require_distributed_env()
    if not torch.cuda.is_available():
        raise ValueError("--distributed requires CUDA")
    rank = int(os.environ[ENV_RANK])
    world_size = int(os.environ[ENV_WORLD_SIZE])
    local_rank = int(os.environ[ENV_LOCAL_RANK])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=DDP_BACKEND, device_id=torch.device(CUDA_DEVICE_TYPE, local_rank))
    return DistributedContext(True, rank, world_size, local_rank)


def cleanup_distributed(distributed: DistributedContext) -> None:
    if distributed.enabled and dist.is_initialized():
        dist.destroy_process_group()


def training_device(args: Namespace, distributed: DistributedContext) -> torch.device:
    if distributed.enabled:
        return torch.device(CUDA_DEVICE_TYPE, distributed.local_rank)
    return torch.device(args.device)


def configure_parallel_model(
    model: torch.nn.Module,
    args: Namespace,
    device: torch.device,
    distributed: DistributedContext,
) -> torch.nn.Module:
    if distributed.enabled:
        find_unused_parameters = ddp_find_unused_parameters(args)
        rank_zero_print(
            distributed,
            f"distributed=True world_size={distributed.world_size} "
            f"local_batch_size={args.batch_size // distributed.world_size} "
            f"find_unused_parameters={find_unused_parameters}",
        )
        return DistributedDataParallel(
            model,
            device_ids=[distributed.local_rank],
            output_device=distributed.local_rank,
            find_unused_parameters=find_unused_parameters,
        )
    if not args.multi_gpu:
        return model
    gpu_count = torch.cuda.device_count()
    print(f"multi_gpu=True data_parallel_gpus={gpu_count} primary_device={device}")
    return torch.nn.DataParallel(model)


def _require_distributed_env() -> None:
    missing = [name for name in (ENV_RANK, ENV_WORLD_SIZE, ENV_LOCAL_RANK) if name not in os.environ]
    if missing:
        raise RuntimeError(f"--distributed must be launched with torchrun; missing env vars: {missing}")


def ddp_find_unused_parameters(args: Namespace) -> bool:
    if args.ddp_find_unused_parameters == DDP_FIND_UNUSED_TRUE:
        return True
    if args.ddp_find_unused_parameters == DDP_FIND_UNUSED_FALSE:
        return False
    return _needs_ddp_unused_parameter_detection(args)


def _needs_ddp_unused_parameter_detection(args: Namespace) -> bool:
    return _cal_has_inactive_epochs(args) or _sketch_path_is_conditional(args)


def _cal_has_inactive_epochs(args: Namespace) -> bool:
    return args.cal_weight > NO_CAL_LOSS and args.cal_warmup_epochs > 0


def _sketch_path_is_conditional(args: Namespace) -> bool:
    if not args.use_prcc_sketch:
        return False
    return args.sketch_loss_weight > NO_SKETCH_LOSS or args.rgb_sketch_consistency_weight > NO_SKETCH_LOSS


def configure_backbone_freeze(model, args: Namespace, epoch: int, distributed: DistributedContext) -> None:
    layers = _freeze_backbone_layers(args)
    if not layers:
        return
    freeze = epoch < args.freeze_backbone_epochs
    base_model = _unwrap_model(model)
    changed = _set_backbone_layers_trainable(base_model, layers, not freeze)
    _set_frozen_backbone_layers_eval(base_model, layers, freeze)
    if changed:
        action = "frozen" if freeze else "unfrozen"
        rank_zero_print(distributed, f"backbone_layers_{action}={','.join(layers)} epoch={epoch + 1}")


def _set_backbone_layers_trainable(model, layer_names: list[str], trainable: bool) -> int:
    changed = 0
    for layer_name in layer_names:
        for parameter in getattr(model.backbone, layer_name).parameters():
            changed += int(parameter.requires_grad != trainable)
            parameter.requires_grad = trainable
    return changed


def _set_frozen_backbone_layers_eval(model, layer_names: list[str], freeze: bool) -> None:
    if not freeze:
        return
    for layer_name in layer_names:
        getattr(model.backbone, layer_name).eval()


def _freeze_backbone_layers(args: Namespace) -> list[str]:
    return [name.strip() for name in args.freeze_backbone_layers.split(",") if name.strip()]


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device: torch.device,
    args: Namespace,
    epoch: int,
    distributed: DistributedContext,
) -> dict[str, float]:
    model.train()
    configure_backbone_freeze(model, args, epoch, distributed)
    totals = _empty_epoch_totals()
    effective_cal_weight = _effective_cal_weight(args, epoch)
    effective_consistency_weight = _effective_sketch_consistency_weight(args, epoch)
    progress = tqdm(loader, desc="batches", unit="batch", disable=not distributed.is_main)
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


def validate_training_args(args: Namespace, distributed: DistributedContext) -> None:
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
    if args.eval_period <= 0:
        raise ValueError("eval_period must be > 0")
    if args.lr_gamma <= 0.0:
        raise ValueError("lr_gamma must be > 0")
    _lr_milestones(args.lr_milestones)
    if args.ddp_find_unused_parameters not in {DDP_FIND_UNUSED_AUTO, DDP_FIND_UNUSED_TRUE, DDP_FIND_UNUSED_FALSE}:
        raise ValueError("ddp_find_unused_parameters must be one of: auto, true, false")
    if args.resume and args.pretrained_checkpoint:
        raise ValueError("--resume and --pretrained-checkpoint are mutually exclusive")
    if args.best_metric not in BEST_METRIC_CHOICES:
        raise ValueError(f"best_metric must be one of {sorted(BEST_METRIC_CHOICES)}, got {args.best_metric}")
    if args.best_variant not in BEST_VARIANT_CHOICES:
        raise ValueError(f"best_variant must be one of {sorted(BEST_VARIANT_CHOICES)}, got {args.best_variant}")
    _validate_freeze_args(args)
    _validate_probability_args(args)
    _validate_parallel_args(args, distributed)


def _validate_probability_args(args: Namespace) -> None:
    for name in AUGMENT_PROBABILITY_ARGS:
        value = getattr(args, name)
        if value < 0.0 or value > 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}")


def _validate_freeze_args(args: Namespace) -> None:
    if args.freeze_backbone_epochs < 0:
        raise ValueError("freeze_backbone_epochs must be >= 0")
    layers = _freeze_backbone_layers(args)
    if args.freeze_backbone_epochs > 0 and not layers:
        raise ValueError("freeze_backbone_layers must not be empty when freeze_backbone_epochs > 0")
    invalid = set(layers) - FREEZABLE_BACKBONE_LAYERS
    if invalid:
        raise ValueError(f"Unknown freeze_backbone_layers: {sorted(invalid)}")


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch: int, metric_name: str, metric_value: float, dataset, variant: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": _unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    payload.update(_checkpoint_metadata(epoch, metric_name, metric_value, dataset, variant))
    torch.save(payload, path)


def load_checkpoint(path: str, model, optimizer, scheduler) -> int:
    if not path:
        return 0
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    return int(checkpoint["epoch"]) + 1


def run_training(model, loader, optimizer, scheduler, scaler, start_epoch: int, dataset, device, args, distributed, pretrained_count) -> None:
    best_metric_value = 0.0
    output_dir = Path(args.output_dir)
    if distributed.is_main:
        _initialize_metric_files(output_dir, start_epoch)
        _write_run_config(output_dir, args, dataset, loader, distributed, scheduler, pretrained_count)
        print(_training_header(args, loader, distributed))
    for epoch in range(start_epoch, args.epochs):
        rank_zero_print(distributed, f"epoch={epoch + 1}/{args.epochs}")
        metrics = train_one_epoch(model, loader, optimizer, scaler, device, args, epoch, distributed)
        if distributed.is_main:
            _print_epoch(epoch, metrics)
            _write_train_metrics(output_dir, epoch, metrics)
        if (epoch + 1) % args.eval_period == 0 or epoch + 1 == args.epochs:
            best_metric_value = _evaluate_epoch(model, optimizer, scheduler, epoch, dataset, best_metric_value, device, args, distributed)
        scheduler.step()


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
    sketch_context = _build_sketch_context(model, batch, labels, has_sketch, device, args, effective_consistency_weight)
    outputs, sketch_outputs = _forward_training_paths(model, images, sketch_context, device, args)
    _validate_batch_targets(labels, outputs["logits"].size(1), "identity label")
    _validate_batch_clothes_targets(clothes_labels, outputs, effective_cal_weight)
    labels = labels.to(device, non_blocking=args.pin_memory)
    clothes_labels = clothes_labels.to(device, non_blocking=args.pin_memory)
    ce_loss = F.cross_entropy(outputs["logits"].float(), labels)
    triplet = batch_hard_triplet_loss(outputs["features"].float(), labels, args.triplet_margin)
    cal_loss = _cal_loss(outputs, clothes_labels, effective_cal_weight)
    sketch_loss, consistency_loss = _sketch_losses(sketch_context, outputs, sketch_outputs, device, args, effective_consistency_weight)
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


@dataclass(frozen=True)
class SketchContext:
    enabled: bool
    images: torch.Tensor | None = None
    labels: torch.Tensor | None = None
    rgb_mask: torch.Tensor | None = None
    consistency_targets: torch.Tensor | None = None


def _build_sketch_context(model, batch, labels: torch.Tensor, has_sketch: torch.Tensor, device, args, consistency_weight: float) -> SketchContext:
    if not _use_sketch_path(args, has_sketch, consistency_weight):
        return SketchContext(False)
    rgb_mask = has_sketch.to(device, non_blocking=args.pin_memory)
    sketch_images = batch["sketch_image"][has_sketch].to(device, non_blocking=args.pin_memory)
    sketch_labels = labels[has_sketch].to(device, non_blocking=args.pin_memory)
    if args.sketch_loss_weight > NO_SKETCH_LOSS:
        return SketchContext(True, sketch_images, sketch_labels, rgb_mask)
    targets = _extract_sketch_targets(model, sketch_images, device, args)["features"].detach()
    return SketchContext(True, sketch_images, sketch_labels, rgb_mask, targets)


def _forward_training_paths(model, images: torch.Tensor, sketch_context: SketchContext, device, args):
    if not sketch_context.enabled or args.sketch_loss_weight <= NO_SKETCH_LOSS:
        with _autocast_context(args, device):
            return model(images), None
    combined_images = torch.cat([images, sketch_context.images], dim=0)
    with _autocast_context(args, device):
        combined_outputs = model(combined_images)
    image_count = images.size(0)
    return _split_outputs(combined_outputs, image_count)


def _split_outputs(outputs: dict[str, torch.Tensor], first_count: int):
    first = {key: value[:first_count] for key, value in outputs.items()}
    second = {key: value[first_count:] for key, value in outputs.items()}
    return first, second


def _sketch_losses(sketch_context: SketchContext, rgb_outputs, sketch_outputs, device, args, consistency_weight: float):
    if not sketch_context.enabled:
        return _zero_loss(device), _zero_loss(device)
    sketch_loss = _sketch_identity_loss(sketch_outputs, sketch_context.labels, device, args)
    consistency = _sketch_consistency_loss(sketch_context, rgb_outputs["features"], sketch_outputs, device, consistency_weight)
    return sketch_loss, consistency


def _use_sketch_path(args: Namespace, has_sketch: torch.Tensor, consistency_weight: float) -> bool:
    if not args.use_prcc_sketch:
        return False
    if args.sketch_loss_weight <= NO_SKETCH_LOSS and consistency_weight <= NO_SKETCH_LOSS:
        return False
    return bool(has_sketch.any().item())


def _sketch_identity_loss(sketch_outputs, sketch_labels: torch.Tensor | None, device, args) -> torch.Tensor:
    if args.sketch_loss_weight <= NO_SKETCH_LOSS:
        return _zero_loss(device)
    sketch_ce = F.cross_entropy(sketch_outputs["logits"].float(), sketch_labels)
    sketch_triplet = _optional_triplet(sketch_outputs["features"].float(), sketch_labels, args.triplet_margin)
    return sketch_ce + args.triplet_weight * sketch_triplet


def _sketch_consistency_loss(sketch_context: SketchContext, rgb_features: torch.Tensor, sketch_outputs, device, weight: float) -> torch.Tensor:
    if weight <= NO_SKETCH_LOSS:
        return _zero_loss(device)
    targets = _sketch_consistency_targets(sketch_context, sketch_outputs)
    return _consistency_loss(rgb_features[sketch_context.rgb_mask].float(), targets.float())


def _sketch_consistency_targets(sketch_context: SketchContext, sketch_outputs) -> torch.Tensor:
    if sketch_context.consistency_targets is not None:
        return sketch_context.consistency_targets
    return sketch_outputs["features"].detach()


def _extract_sketch_targets(model, sketch_images: torch.Tensor, device, args) -> dict[str, torch.Tensor]:
    training_modes = _module_training_modes(model)
    model.eval()
    try:
        with torch.no_grad():
            with _autocast_context(args, device):
                return model(sketch_images)
    finally:
        _restore_module_training_modes(training_modes)


def _module_training_modes(model) -> dict[torch.nn.Module, bool]:
    return {module: module.training for module in model.modules()}


def _restore_module_training_modes(training_modes: dict[torch.nn.Module, bool]) -> None:
    for module, training in training_modes.items():
        module.train(training)


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


def _evaluate_and_save(model, optimizer, scheduler, epoch: int, dataset, best: float, device, args) -> float:
    eval_model = _unwrap_model(model)
    eval_metrics = evaluate_enabled_datasets(eval_model, device, args)
    selected_metric = primary_eval_metric(eval_metrics, args.best_metric, args.best_variant)
    output_dir = Path(args.output_dir)
    _write_eval_metrics(output_dir, epoch, eval_metrics)
    save_checkpoint(output_dir / CHECKPOINT_LAST, eval_model, optimizer, scheduler, epoch, args.best_metric, selected_metric, dataset, args.best_variant)
    if selected_metric <= best:
        return best
    save_checkpoint(output_dir / CHECKPOINT_BEST, eval_model, optimizer, scheduler, epoch, args.best_metric, selected_metric, dataset, args.best_variant)
    print(f"new_best {args.best_variant}/{args.best_metric}={selected_metric:.4f}")
    return selected_metric


def _evaluate_epoch(model, optimizer, scheduler, epoch: int, dataset, best: float, device, args, distributed) -> float:
    if distributed.is_main:
        best = _evaluate_and_save(model, optimizer, scheduler, epoch, dataset, best, device, args)
    synchronize_distributed(distributed)
    return best


def synchronize_distributed(distributed: DistributedContext) -> None:
    if distributed.enabled:
        dist.barrier()


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


def _write_run_config(output_dir: Path, args: Namespace, dataset, loader, distributed, scheduler, pretrained_count) -> None:
    config = {
        "args": vars(args),
        "dataset": _dataset_summary(dataset),
        "loader": _loader_summary(loader),
        "distributed": _distributed_summary(distributed),
        "pretrained_parameter_count": pretrained_count,
        "lr_scheduler": _scheduler_summary(args, scheduler),
    }
    with (output_dir / RUN_CONFIG_JSON).open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)


def _dataset_summary(dataset) -> dict:
    valid_samples = [sample for sample in dataset.samples if not sample.is_junk]
    return {
        "samples": len(dataset.samples),
        "valid_samples": len(valid_samples),
        "identities": dataset.num_classes,
        "clothes_classes": dataset.num_clothes_classes,
        "source_samples": _count_by_source(valid_samples),
        "source_identities": _count_identities_by_source(valid_samples),
    }


def _loader_summary(loader) -> dict:
    batch_sampler = getattr(loader, "batch_sampler", None)
    return {
        "batches_per_epoch": len(loader),
        "batch_size": getattr(loader, "batch_size", None),
        "batch_sampler_batch_size": getattr(batch_sampler, "batch_size", None),
        "identities_per_batch": getattr(batch_sampler, "identities_per_batch", None),
        "instances": getattr(batch_sampler, "instances", None),
        "num_workers": getattr(loader, "num_workers", None),
        "pin_memory": getattr(loader, "pin_memory", None),
        "persistent_workers": _loader_has_persistent_workers(loader),
    }


def _distributed_summary(distributed: DistributedContext) -> dict:
    return {
        "enabled": distributed.enabled,
        "rank": distributed.rank,
        "world_size": distributed.world_size,
        "local_rank": distributed.local_rank,
    }


def _scheduler_summary(args: Namespace, scheduler) -> dict:
    return {
        "name": scheduler.__class__.__name__,
        "initial_lr": args.lr,
        "current_lr": scheduler.get_last_lr(),
        "milestones": _lr_milestones(args.lr_milestones),
        "gamma": args.lr_gamma,
    }


def _count_by_source(samples) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample in samples:
        counts[sample.source] = counts.get(sample.source, 0) + 1
    return counts


def _count_identities_by_source(samples) -> dict[str, int]:
    identities: dict[str, set[int]] = {}
    for sample in samples:
        identities.setdefault(sample.source, set()).add(sample.label)
    return {source: len(labels) for source, labels in identities.items()}


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


def _checkpoint_metadata(epoch: int, metric_name: str, metric_value: float, dataset, variant: str) -> dict[str, int | float | str]:
    return {
        "epoch": epoch,
        "best_metric": metric_name,
        "best_variant": variant,
        "best_metric_value": metric_value,
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


def _training_header(args: Namespace, loader, distributed: DistributedContext) -> str:
    parts = [
        f"precision={args.precision}",
        f"lr={scheduler_lrs(args.lr)}",
        f"lr_milestones={args.lr_milestones}",
        f"lr_gamma={args.lr_gamma}",
        f"best_metric={args.best_metric}",
        f"best_variant={args.best_variant}",
        f"feature_key={args.feature_key}",
        f"eval_period={args.eval_period}",
        f"freeze_backbone_epochs={args.freeze_backbone_epochs}",
        f"freeze_backbone_layers={args.freeze_backbone_layers}",
        f"pin_memory={args.pin_memory}",
        f"persistent_workers={_loader_has_persistent_workers(loader)}",
    ]
    if distributed.enabled:
        parts.extend([f"world_size={distributed.world_size}", f"rank={distributed.rank}"])
    return " ".join(parts)


def rank_zero_print(distributed: DistributedContext, message: str) -> None:
    if distributed.is_main:
        print(message)


def scheduler_lrs(lr: float) -> str:
    return f"{lr:.8g}"


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if isinstance(model, (torch.nn.DataParallel, DistributedDataParallel)):
        return model.module
    return model


def _validate_parallel_args(args: Namespace, distributed: DistributedContext) -> None:
    if args.distributed:
        _validate_distributed_args(args, distributed)
    _validate_multi_gpu_args(args)


def _validate_distributed_args(args: Namespace, distributed: DistributedContext) -> None:
    if args.multi_gpu:
        raise ValueError("--distributed and --multi-gpu are mutually exclusive")
    if distributed.world_size <= 1:
        raise ValueError("--distributed requires WORLD_SIZE > 1")
    if args.batch_size % distributed.world_size != 0:
        raise ValueError("batch_size must be divisible by distributed world_size")
    if (args.batch_size // distributed.world_size) % args.instances != 0:
        raise ValueError("local batch_size must be divisible by instances")


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
