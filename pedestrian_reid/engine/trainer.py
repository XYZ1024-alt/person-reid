from __future__ import annotations

from argparse import Namespace
from contextlib import nullcontext
import csv
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import random

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from pedestrian_reid.builders import MODE_MARKET, MODE_PRCC, MODE_PRCC_DEV, build_train_loader, build_training_dataset
from pedestrian_reid.builders import selected_prcc_dev_pids
from pedestrian_reid.data.datasets import PRCC_SOURCE, UNKNOWN_CLOTHES
from pedestrian_reid.engine.evaluator import evaluate_enabled_datasets, load_model, primary_eval_metric
from pedestrian_reid.data.transforms import VARIANT_DARK, VARIANT_OCCLUDED, VARIANT_STANDARD
from pedestrian_reid.modules.losses import batch_hard_triplet_loss
from pedestrian_reid.modules.metrics import COMBINED_FEATURE_KEY
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
    "ce_market",
    "ce_prcc",
    "triplet",
    "cal",
    "sketch",
    "consistency",
    "distill",
    "part_triplet",
    "cloth_invariant",
    "cross_clothes_contrastive",
    "valid_cross_clothes_pairs",
    "effective_cal_weight",
    "effective_prcc_ce_weight",
    "effective_sketch_consistency_weight",
    "effective_distill_weight",
]
EVAL_METRIC_FIELDS = ["rank1", "rank2", "rank3", "rank4", "rank5", "mAP"]
BEST_METRIC_RANK1 = "rank1"
BEST_METRIC_MAP = "mAP"
BEST_METRIC_CHOICES = {BEST_METRIC_RANK1, BEST_METRIC_MAP}
BEST_DATASET_AUTO = "auto"
BEST_DATASET_CHOICES = {BEST_DATASET_AUTO, MODE_MARKET, MODE_PRCC, MODE_PRCC_DEV}
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
BACKBONE_LAYER_ORDER = ("stem", "layer1", "layer2", "layer3", "layer4")
PRETRAIN_SKIP_PREVIEW_LIMIT = 10
MIN_IMAGENET_PRETRAINED_TENSORS = 300
PRCC_EXPECTED_CLOTHES_PER_IDENTITY = 2
MIN_PARTS = 1
NO_PART_LOSS = 0.0
NO_CLOTH_INVARIANT_LOSS = 0.0
NO_DISTILL_LOSS = 0.0
NO_CROSS_CLOTHES_CONTRASTIVE_LOSS = 0.0
NO_PRCC_CE_WEIGHT = 0.0
NO_PRCC_CE_RAMP_EPOCHS = 0
SINGLE_RAMP_EPOCH = 1
NO_PRCC_DEV_IDENTITIES = 0
MIN_CONTRASTIVE_TEMPERATURE = 0.0
MIN_POSITIVE_COUNT = 1
UPPER_TRIANGLE_DIAGONAL = 1
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


@dataclass(frozen=True)
class CheckpointResumeState:
    start_epoch: int = 0
    best_metric_value: float = 0.0


@dataclass(frozen=True, kw_only=True)
class CheckpointTarget:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: object
    scaler: object


@dataclass(frozen=True, kw_only=True)
class CheckpointSaveRequest:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: object
    scaler: object
    epoch: int
    metric_name: str
    metric_value: float
    dataset: object
    variant: str
    best_dataset: str


@dataclass(frozen=True, kw_only=True)
class TrainingRun:
    model: torch.nn.Module
    teacher: torch.nn.Module | None
    loader: object
    optimizer: torch.optim.Optimizer
    scheduler: object
    scaler: object
    resume_state: CheckpointResumeState
    dataset: object
    device: torch.device
    args: Namespace
    distributed: DistributedContext
    pretrained_count: int | None


@dataclass(frozen=True)
class ClassificationLosses:
    total: torch.Tensor
    market: torch.Tensor
    prcc: torch.Tensor


@dataclass(frozen=True)
class AuxiliaryLosses:
    part_triplet: torch.Tensor
    cloth_invariant: torch.Tensor
    cross_clothes_contrastive: torch.Tensor
    valid_cross_clothes_pairs: torch.Tensor


@dataclass(frozen=True, kw_only=True)
class LossComponents:
    classification: ClassificationLosses
    triplet: torch.Tensor
    cal: torch.Tensor
    sketch: torch.Tensor
    consistency: torch.Tensor
    distill: torch.Tensor
    auxiliary: AuxiliaryLosses
    effective_cal_weight: float
    effective_prcc_ce_weight: float
    consistency_weight: float
    distill_weight: float


def train_from_args(args: Namespace) -> None:
    distributed = initialize_distributed(args)
    try:
        set_seed(args.seed + distributed.rank)
        validate_training_args(args, distributed)
        device = training_device(args, distributed)
        dataset = build_training_dataset(args)
        validate_training_dataset(dataset, args)
        loader = build_train_loader(dataset, args, distributed)
        _require_cal_labels(dataset.num_clothes_classes, args.cal_weight)
        model = build_model(dataset, args).to(device)
        teacher = initialize_teacher(args, device)
        pretrained_count = initialize_model_weights(model, args, distributed)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = build_lr_scheduler(optimizer, args)
        scaler = _build_grad_scaler(args, device)
        resume_state = load_checkpoint(args.resume, CheckpointTarget(model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler))
        model = configure_parallel_model(model, args, device, distributed)
        run_training(
            TrainingRun(
                model=model,
                teacher=teacher,
                loader=loader,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                resume_state=resume_state,
                dataset=dataset,
                device=device,
                args=args,
                distributed=distributed,
                pretrained_count=pretrained_count,
            )
        )
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


def build_model(dataset, args: Namespace) -> PedestrianReIDNet:
    return PedestrianReIDNet(
        dataset.num_classes,
        num_clothes_classes=dataset.num_clothes_classes,
        use_part_branch=args.use_part_branch,
        num_parts=args.num_parts,
        part_embedding_dim=args.part_embedding_dim,
        combined_global_weight=args.combined_global_weight,
        combined_part_weight=args.combined_part_weight,
    )


def initialize_teacher(args: Namespace, device: torch.device) -> torch.nn.Module | None:
    if _max_distill_weight(args) <= NO_DISTILL_LOSS:
        return None
    teacher = load_model(args.teacher_checkpoint, device)
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    teacher.eval()
    return teacher


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
    return _cal_has_inactive_epochs(args) or _sketch_path_is_conditional(args) or _backbone_freeze_is_active(args)


def _cal_has_inactive_epochs(args: Namespace) -> bool:
    return args.cal_weight > NO_CAL_LOSS and args.cal_warmup_epochs > 0


def _sketch_path_is_conditional(args: Namespace) -> bool:
    if not args.use_prcc_sketch:
        return False
    return args.sketch_loss_weight > NO_SKETCH_LOSS or args.rgb_sketch_consistency_weight > NO_SKETCH_LOSS


def _backbone_freeze_is_active(args: Namespace) -> bool:
    return args.freeze_backbone_all_epochs or args.freeze_backbone_epochs > 0


def configure_backbone_freeze(model, args: Namespace, epoch: int, *, distributed: DistributedContext) -> None:
    layers = _active_freeze_layers(args)
    if not layers:
        return
    freeze = args.freeze_backbone_all_epochs or epoch < args.freeze_backbone_epochs
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


def _active_freeze_layers(args: Namespace) -> list[str]:
    if args.freeze_backbone_all_epochs:
        return list(BACKBONE_LAYER_ORDER)
    return _freeze_backbone_layers(args)


def train_one_epoch(
    model,
    loader,
    *,
    teacher,
    optimizer,
    scaler,
    device: torch.device,
    args: Namespace,
    epoch: int,
    distributed: DistributedContext,
) -> dict[str, float]:
    model.train()
    configure_backbone_freeze(model, args, epoch, distributed=distributed)
    totals = _empty_epoch_totals()
    effective_cal_weight = _effective_cal_weight(args, epoch)
    effective_prcc_ce_weight = _effective_prcc_ce_weight(args, epoch)
    effective_consistency_weight = _effective_sketch_consistency_weight(args, epoch)
    effective_distill_weight = _effective_distill_weight(args, epoch)
    progress = tqdm(loader, desc="batches", unit="batch", disable=not distributed.is_main)
    for batch in progress:
        losses = _train_batch(
            model,
            batch,
            teacher=teacher,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            args=args,
            effective_cal_weight=effective_cal_weight,
            effective_prcc_ce_weight=effective_prcc_ce_weight,
            effective_consistency_weight=effective_consistency_weight,
            effective_distill_weight=effective_distill_weight,
        )
        _accumulate(totals, losses)
        progress.set_postfix(
            _batch_metrics(
                losses,
                cal_weight=effective_cal_weight,
                prcc_ce_weight=effective_prcc_ce_weight,
                consistency_weight=effective_consistency_weight,
                distill_weight=effective_distill_weight,
            )
        )
    metrics = {key: value / len(loader) for key, value in totals.items()}
    metrics["effective_cal_weight"] = effective_cal_weight
    metrics["effective_prcc_ce_weight"] = effective_prcc_ce_weight
    metrics["effective_sketch_consistency_weight"] = effective_consistency_weight
    metrics["effective_distill_weight"] = effective_distill_weight
    return metrics


def validate_training_dataset(dataset, args: Namespace) -> None:
    _validate_contiguous_targets(_target_values(dataset.samples, "label"), dataset.num_classes, "identity label")
    clothes = _target_values(dataset.samples, "clothes_id")
    known_clothes = [value for value in clothes if value >= 0]
    if known_clothes:
        _validate_contiguous_targets(known_clothes, dataset.num_clothes_classes, "clothes label")
    _validate_prcc_cal_labels(dataset, args)


def _validate_prcc_cal_labels(dataset, args: Namespace) -> None:
    if args.cal_weight <= NO_CAL_LOSS:
        return
    prcc_samples = _valid_prcc_samples(dataset)
    if not prcc_samples:
        return
    _require_known_prcc_clothes(prcc_samples)
    clothes_by_pid = _prcc_clothes_by_pid(prcc_samples)
    _require_prcc_outfit_count(clothes_by_pid)
    _require_prcc_outfit_label_count(prcc_samples, clothes_by_pid)


def _valid_prcc_samples(dataset) -> list:
    return [sample for sample in dataset.samples if sample.source == PRCC_SOURCE and not sample.is_junk]


def _require_known_prcc_clothes(samples: list) -> None:
    unknown_pids = sorted({sample.pid for sample in samples if sample.clothes_id == UNKNOWN_CLOTHES})
    if unknown_pids:
        raise ValueError(f"PRCC CAL requires known clothes labels; unknown_pids={unknown_pids[:10]}")


def _prcc_clothes_by_pid(samples: list) -> dict[int, set[int]]:
    clothes_by_pid: dict[int, set[int]] = {}
    for sample in samples:
        clothes_by_pid.setdefault(sample.pid, set()).add(sample.clothes_id)
    return clothes_by_pid


def _require_prcc_outfit_count(clothes_by_pid: dict[int, set[int]]) -> None:
    invalid = [pid for pid, clothes in clothes_by_pid.items() if len(clothes) != PRCC_EXPECTED_CLOTHES_PER_IDENTITY]
    if invalid:
        raise ValueError(f"PRCC CAL requires 2 outfit labels per identity; invalid_pids={invalid[:10]}")


def _require_prcc_outfit_label_count(samples: list, clothes_by_pid: dict[int, set[int]]) -> None:
    expected = len(clothes_by_pid) * PRCC_EXPECTED_CLOTHES_PER_IDENTITY
    actual = len({sample.clothes_id for sample in samples})
    if actual != expected:
        raise ValueError(f"PRCC clothes labels must be outfit-level; expected={expected} actual={actual}")


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
    if args.part_triplet_weight < 0:
        raise ValueError("part_triplet_weight must be >= 0")
    if args.cloth_invariant_weight < 0:
        raise ValueError("cloth_invariant_weight must be >= 0")
    if args.num_parts < MIN_PARTS:
        raise ValueError(f"num_parts must be >= {MIN_PARTS}")
    if args.part_embedding_dim <= 0:
        raise ValueError("part_embedding_dim must be > 0")
    _validate_combined_weight_args(args)
    if args.part_triplet_weight > NO_PART_LOSS and not args.use_part_branch:
        raise ValueError("part_triplet_weight requires --use-part-branch")
    if args.cloth_invariant_weight > NO_CLOTH_INVARIANT_LOSS and args.mode == MODE_MARKET:
        raise ValueError("cloth_invariant_weight requires PRCC or joint mode")
    _validate_objective_shift_args(args)
    _validate_distill_args(args)
    _validate_feature_key_args(args)
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
    _validate_best_dataset_args(args)
    if args.best_variant not in BEST_VARIANT_CHOICES:
        raise ValueError(f"best_variant must be one of {sorted(BEST_VARIANT_CHOICES)}, got {args.best_variant}")
    _validate_freeze_args(args)
    _validate_probability_args(args)
    _validate_parallel_args(args, distributed)


def _validate_combined_weight_args(args: Namespace) -> None:
    if args.combined_global_weight < 0.0 or args.combined_part_weight < 0.0:
        raise ValueError("combined feature weights must be >= 0")
    if args.combined_global_weight == 0.0 and args.combined_part_weight == 0.0:
        raise ValueError("at least one combined feature weight must be > 0")


def _validate_objective_shift_args(args: Namespace) -> None:
    if args.prcc_dev_identities < NO_PRCC_DEV_IDENTITIES:
        raise ValueError("prcc_dev_identities must be >= 0")
    if args.prcc_ce_weight < NO_PRCC_CE_WEIGHT:
        raise ValueError("prcc_ce_weight must be >= 0")
    if args.prcc_ce_final_weight < NO_PRCC_CE_WEIGHT:
        raise ValueError("prcc_ce_final_weight must be >= 0")
    if args.prcc_ce_ramp_epochs < NO_PRCC_CE_RAMP_EPOCHS:
        raise ValueError("prcc_ce_ramp_epochs must be >= 0")
    if args.cross_clothes_contrastive_weight < NO_CROSS_CLOTHES_CONTRASTIVE_LOSS:
        raise ValueError("cross_clothes_contrastive_weight must be >= 0")
    if args.cross_clothes_contrastive_weight > NO_CROSS_CLOTHES_CONTRASTIVE_LOSS and args.mode == MODE_MARKET:
        raise ValueError("cross_clothes_contrastive_weight requires PRCC or joint mode")
    if args.contrastive_temperature <= MIN_CONTRASTIVE_TEMPERATURE:
        raise ValueError("contrastive_temperature must be > 0")


def _validate_feature_key_args(args: Namespace) -> None:
    if args.feature_key == COMBINED_FEATURE_KEY and not args.use_part_branch:
        raise ValueError("combined_features requires --use-part-branch")
    if args.triplet_feature_key == COMBINED_FEATURE_KEY and not args.use_part_branch:
        raise ValueError("triplet_feature_key=combined_features requires --use-part-branch")


def _validate_best_dataset_args(args: Namespace) -> None:
    if args.best_dataset not in BEST_DATASET_CHOICES:
        raise ValueError(f"best_dataset must be one of {sorted(BEST_DATASET_CHOICES)}, got {args.best_dataset}")
    if args.mode == MODE_MARKET and args.prcc_dev_identities > NO_PRCC_DEV_IDENTITIES:
        raise ValueError("prcc_dev_identities requires PRCC or joint mode")
    if args.best_dataset == MODE_PRCC_DEV and args.prcc_dev_identities <= NO_PRCC_DEV_IDENTITIES:
        raise ValueError("best_dataset=prcc_dev requires --prcc-dev-identities > 0")
    if args.mode == MODE_MARKET and args.best_dataset in {MODE_PRCC, MODE_PRCC_DEV}:
        raise ValueError(f"best_dataset={args.best_dataset} is unavailable in market mode")
    if args.mode == MODE_PRCC and args.best_dataset == MODE_MARKET:
        raise ValueError("best_dataset=market is unavailable in prcc mode")
    if args.prcc_dev_identities > NO_PRCC_DEV_IDENTITIES and args.best_dataset == MODE_PRCC:
        raise ValueError("best_dataset=prcc is not evaluated while --prcc-dev-identities is active")


def _validate_distill_args(args: Namespace) -> None:
    if args.distill_weight < 0:
        raise ValueError("distill_weight must be >= 0")
    if args.distill_final_weight < 0:
        raise ValueError("distill_final_weight must be >= 0")
    if args.distill_hold_epochs < 0:
        raise ValueError("distill_hold_epochs must be >= 0")
    if args.distill_ramp_epochs < 0:
        raise ValueError("distill_ramp_epochs must be >= 0")
    if _max_distill_weight(args) > NO_DISTILL_LOSS and not args.teacher_checkpoint:
        raise ValueError("distill_weight requires --teacher-checkpoint")


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


def save_checkpoint(path: Path, request: CheckpointSaveRequest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base_model = _unwrap_model(request.model)
    payload = {
        "model": base_model.state_dict(),
        "optimizer": request.optimizer.state_dict(),
        "scheduler": request.scheduler.state_dict(),
        "scaler": request.scaler.state_dict(),
        "model_config": _model_config(base_model),
    }
    payload.update(_checkpoint_metadata(request))
    torch.save(payload, path)


def load_checkpoint(path: str, target: CheckpointTarget) -> CheckpointResumeState:
    if not path:
        return CheckpointResumeState()
    checkpoint = torch.load(path, map_location="cpu")
    target.model.load_state_dict(checkpoint["model"])
    target.optimizer.load_state_dict(checkpoint["optimizer"])
    target.scheduler.load_state_dict(checkpoint["scheduler"])
    if "scaler" in checkpoint:
        target.scaler.load_state_dict(checkpoint["scaler"])
    return CheckpointResumeState(
        start_epoch=int(checkpoint["epoch"]) + 1,
        best_metric_value=float(checkpoint["best_metric_value"]),
    )


def run_training(run: TrainingRun) -> None:
    start_epoch = run.resume_state.start_epoch
    best_metric_value = run.resume_state.best_metric_value
    output_dir = Path(run.args.output_dir)
    if run.distributed.is_main:
        _initialize_metric_files(output_dir, start_epoch)
        _write_run_config(output_dir, run.args, run.dataset, run.loader, run.distributed, run.scheduler, run.pretrained_count)
        print(_training_header(run.args, run.loader, run.distributed))
    for epoch in range(start_epoch, run.args.epochs):
        rank_zero_print(run.distributed, f"epoch={epoch + 1}/{run.args.epochs}")
        metrics = train_one_epoch(
            run.model,
            run.loader,
            teacher=run.teacher,
            optimizer=run.optimizer,
            scaler=run.scaler,
            device=run.device,
            args=run.args,
            epoch=epoch,
            distributed=run.distributed,
        )
        if run.distributed.is_main:
            _print_epoch(epoch, metrics)
            _write_train_metrics(output_dir, epoch, metrics)
        run.scheduler.step()
        if (epoch + 1) % run.args.eval_period == 0 or epoch + 1 == run.args.epochs:
            best_metric_value = _evaluate_epoch(
                run.model,
                run.optimizer,
                run.scheduler,
                run.scaler,
                epoch,
                run.dataset,
                best_metric_value,
                run.device,
                run.args,
                run.distributed,
            )


def _train_batch(
    model,
    batch,
    *,
    teacher,
    optimizer,
    scaler,
    device: torch.device,
    args: Namespace,
    effective_cal_weight: float,
    effective_prcc_ce_weight: float,
    effective_consistency_weight: float,
    effective_distill_weight: float,
):
    images = batch["image"].to(device, non_blocking=args.pin_memory)
    labels = batch["label"]
    clothes_labels = batch["clothes_label"]
    sources = batch["source"]
    has_sketch = batch["has_sketch"].bool()
    sketch_context = _build_sketch_context(model, batch, labels, has_sketch, device, args, effective_consistency_weight)
    outputs, sketch_outputs = _forward_training_paths(model, images, sketch_context, device, args)
    _validate_batch_targets(labels, outputs["logits"].size(1), "identity label")
    _validate_batch_clothes_targets(clothes_labels, outputs, effective_cal_weight)
    labels = labels.to(device, non_blocking=args.pin_memory)
    clothes_labels = clothes_labels.to(device, non_blocking=args.pin_memory)
    classification = _classification_losses(outputs, labels, sources, effective_prcc_ce_weight, device)
    triplet_features = _training_feature_output(outputs, args.triplet_feature_key).float()
    triplet = batch_hard_triplet_loss(triplet_features, labels, args.triplet_margin)
    cal_loss = _cal_loss(outputs, clothes_labels, effective_cal_weight)
    sketch_loss, consistency_loss = _sketch_losses(sketch_context, outputs, sketch_outputs, device, args, effective_consistency_weight)
    distill_loss = _distill_loss(
        outputs,
        teacher=teacher,
        images=images,
        device=device,
        args=args,
        distill_weight=effective_distill_weight,
    )
    auxiliary = _auxiliary_losses(outputs, labels, clothes_labels, sources, args, device)
    components = LossComponents(
        classification=classification,
        triplet=triplet,
        cal=cal_loss,
        sketch=sketch_loss,
        consistency=consistency_loss,
        distill=distill_loss,
        auxiliary=auxiliary,
        effective_cal_weight=effective_cal_weight,
        effective_prcc_ce_weight=effective_prcc_ce_weight,
        consistency_weight=effective_consistency_weight,
        distill_weight=effective_distill_weight,
    )
    loss = _total_loss(args, components)
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return _batch_loss_metrics(loss, components)


def _batch_loss_metrics(loss: torch.Tensor, components: LossComponents) -> dict[str, torch.Tensor]:
    return {
        "loss": loss,
        "ce": components.classification.total,
        "ce_market": components.classification.market,
        "ce_prcc": components.classification.prcc,
        "triplet": components.triplet,
        "cal": components.cal,
        "sketch": components.sketch,
        "consistency": components.consistency,
        "distill": components.distill,
        "part_triplet": components.auxiliary.part_triplet,
        "cloth_invariant": components.auxiliary.cloth_invariant,
        "cross_clothes_contrastive": components.auxiliary.cross_clothes_contrastive,
        "valid_cross_clothes_pairs": components.auxiliary.valid_cross_clothes_pairs,
    }


def _classification_losses(outputs, labels: torch.Tensor, sources, prcc_weight: float, device) -> ClassificationLosses:
    prcc_mask = _source_mask(sources, device)
    market_loss = _masked_cross_entropy(outputs["logits"].float(), labels, ~prcc_mask, device)
    prcc_loss = _masked_cross_entropy(outputs["logits"].float(), labels, prcc_mask, device)
    total = market_loss + prcc_weight * prcc_loss
    return ClassificationLosses(total, market_loss, prcc_loss)


def _masked_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, device) -> torch.Tensor:
    if not mask.any().item():
        return _zero_loss(device)
    return F.cross_entropy(logits[mask], labels[mask])


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
    sketch_features = _training_feature_output(sketch_outputs, args.triplet_feature_key).float()
    sketch_triplet = _optional_triplet(sketch_features, sketch_labels, args.triplet_margin)
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


def _training_feature_output(outputs: dict[str, torch.Tensor], feature_key: str) -> torch.Tensor:
    if feature_key not in outputs:
        raise ValueError(f"Model did not produce triplet_feature_key={feature_key}; enable the matching model branch")
    return outputs[feature_key]


def _distill_loss(
    outputs,
    *,
    teacher,
    images: torch.Tensor,
    device: torch.device,
    args: Namespace,
    distill_weight: float,
) -> torch.Tensor:
    if teacher is None or distill_weight <= NO_DISTILL_LOSS:
        return _zero_loss(device)
    with torch.no_grad():
        with _autocast_context(args, device):
            teacher_outputs = teacher(images)
    student_features = outputs["bn_features"].float()
    teacher_features = teacher_outputs["bn_features"].detach().float()
    return 1.0 - F.cosine_similarity(student_features, teacher_features, dim=1).mean()


def _auxiliary_losses(outputs, labels: torch.Tensor, clothes_labels: torch.Tensor, sources, args, device) -> AuxiliaryLosses:
    part_triplet = _part_triplet_loss(outputs, labels, args, device)
    cloth_invariant, pair_count = _cloth_invariant_loss(outputs, labels, clothes_labels, sources, args, device)
    contrastive = _cross_clothes_contrastive_loss(outputs, labels, clothes_labels, sources, args, device)
    return AuxiliaryLosses(part_triplet, cloth_invariant, contrastive, pair_count)


def _part_triplet_loss(outputs, labels: torch.Tensor, args, device) -> torch.Tensor:
    if args.part_triplet_weight <= NO_PART_LOSS:
        return _zero_loss(device)
    if "part_features" not in outputs:
        return _zero_loss(device)
    part_features = F.normalize(outputs["part_features"].flatten(1).float(), dim=1)
    return _optional_triplet(part_features, labels, args.triplet_margin)


def _cloth_invariant_loss(outputs, labels: torch.Tensor, clothes_labels: torch.Tensor, sources, args, device):
    if args.cloth_invariant_weight <= NO_CLOTH_INVARIANT_LOSS:
        return _zero_loss(device), _zero_pair_count(device)
    pair_mask = _cross_clothes_pair_mask(labels, clothes_labels, sources, device)
    pair_count = pair_mask.sum()
    if pair_count.item() == 0:
        return _zero_loss(device), _zero_pair_count(device)
    similarities = _pairwise_cosine(_invariant_features(outputs).float())
    return (1.0 - similarities[pair_mask]).mean(), pair_count.float()


def _cross_clothes_contrastive_loss(outputs, labels: torch.Tensor, clothes_labels: torch.Tensor, sources, args, device):
    if args.cross_clothes_contrastive_weight <= NO_CROSS_CLOTHES_CONTRASTIVE_LOSS:
        return _zero_loss(device)
    features = _training_feature_output(outputs, args.triplet_feature_key).float()
    valid = _source_mask(sources, device) & clothes_labels.ge(0)
    if not valid.any().item():
        raise ValueError("cross-clothes contrastive loss requires PRCC samples with known clothes labels")
    return _supervised_cross_clothes_contrastive(features[valid], labels[valid], clothes_labels[valid], args.contrastive_temperature)


def _supervised_cross_clothes_contrastive(
    features: torch.Tensor,
    labels: torch.Tensor,
    clothes_labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    similarities = _pairwise_cosine(features) / temperature
    same_identity = labels.unsqueeze(0).eq(labels.unsqueeze(1))
    different_clothes = clothes_labels.unsqueeze(0).ne(clothes_labels.unsqueeze(1))
    positive_mask = same_identity & different_clothes
    denominator_mask = positive_mask | ~same_identity
    positive_rows = positive_mask.any(dim=1)
    if not positive_rows.any().item():
        raise ValueError("cross-clothes contrastive loss found no positive cross-clothes pairs")
    log_prob = similarities - similarities.masked_fill(~denominator_mask, float("-inf")).logsumexp(dim=1, keepdim=True)
    positive_count = positive_mask.sum(dim=1).clamp(min=MIN_POSITIVE_COUNT)
    positive_log_prob = (log_prob * positive_mask.float()).sum(dim=1) / positive_count
    return -positive_log_prob[positive_rows].mean()


def _cross_clothes_pair_mask(labels: torch.Tensor, clothes_labels: torch.Tensor, sources, device) -> torch.Tensor:
    source_mask = _source_mask(sources, device)
    same_identity = labels.unsqueeze(0).eq(labels.unsqueeze(1))
    different_clothes = clothes_labels.unsqueeze(0).ne(clothes_labels.unsqueeze(1))
    known_clothes = clothes_labels.ge(0)
    valid_source = source_mask.unsqueeze(0) & source_mask.unsqueeze(1)
    valid_clothes = known_clothes.unsqueeze(0) & known_clothes.unsqueeze(1)
    pair_mask = same_identity & different_clothes & valid_source & valid_clothes
    return torch.triu(pair_mask, diagonal=UPPER_TRIANGLE_DIAGONAL)


def _source_mask(sources, device) -> torch.Tensor:
    return torch.tensor([source == PRCC_SOURCE for source in sources], dtype=torch.bool, device=device)


def _pairwise_cosine(features: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(features, dim=1)
    return normalized @ normalized.t()


def _invariant_features(outputs) -> torch.Tensor:
    if "combined_features" in outputs:
        return outputs["combined_features"]
    return outputs["features"]


def _zero_pair_count(device: torch.device) -> torch.Tensor:
    return torch.zeros((), device=device)


def _consistency_loss(rgb_features: torch.Tensor, sketch_features: torch.Tensor) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(rgb_features, sketch_features, dim=1).mean()


def _total_loss(args, components: LossComponents):
    loss = components.classification.total + args.triplet_weight * components.triplet
    loss = loss + components.effective_cal_weight * components.cal
    loss = loss + args.sketch_loss_weight * components.sketch
    loss = loss + components.consistency_weight * components.consistency
    loss = loss + components.distill_weight * components.distill
    loss = loss + args.part_triplet_weight * components.auxiliary.part_triplet
    loss = loss + args.cloth_invariant_weight * components.auxiliary.cloth_invariant
    return loss + args.cross_clothes_contrastive_weight * components.auxiliary.cross_clothes_contrastive


def _evaluate_and_save(model, optimizer, scheduler, scaler, epoch: int, dataset, best: float, device, args) -> float:
    eval_model = _unwrap_model(model)
    eval_metrics = evaluate_enabled_datasets(eval_model, device, args)
    selected_metric = primary_eval_metric(eval_metrics, args.best_metric, args.best_variant, args.best_dataset)
    output_dir = Path(args.output_dir)
    best_metric_value = max(best, selected_metric)
    _write_eval_metrics(output_dir, epoch, eval_metrics)
    request = CheckpointSaveRequest(
        model=eval_model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=epoch,
        metric_name=args.best_metric,
        metric_value=best_metric_value,
        dataset=dataset,
        variant=args.best_variant,
        best_dataset=args.best_dataset,
    )
    save_checkpoint(output_dir / CHECKPOINT_LAST, request)
    if selected_metric <= best:
        return best
    save_checkpoint(output_dir / CHECKPOINT_BEST, replace(request, metric_value=selected_metric))
    print(f"new_best {args.best_dataset}/{args.best_variant}/{args.best_metric}={selected_metric:.4f}")
    return selected_metric


def _evaluate_epoch(model, optimizer, scheduler, scaler, epoch: int, dataset, best: float, device, args, distributed) -> float:
    if distributed.is_main:
        best = _evaluate_and_save(model, optimizer, scheduler, scaler, epoch, dataset, best, device, args)
    synchronize_distributed(distributed)
    return best


def synchronize_distributed(distributed: DistributedContext) -> None:
    if distributed.enabled:
        dist.barrier()


def _accumulate(totals: dict[str, float], losses: dict[str, torch.Tensor]) -> None:
    for key in totals:
        totals[key] += losses[key].item()


def _empty_epoch_totals() -> dict[str, float]:
    return {
        "loss": 0.0,
        "ce": 0.0,
        "ce_market": 0.0,
        "ce_prcc": 0.0,
        "triplet": 0.0,
        "cal": 0.0,
        "sketch": 0.0,
        "consistency": 0.0,
        "distill": 0.0,
        "part_triplet": 0.0,
        "cloth_invariant": 0.0,
        "cross_clothes_contrastive": 0.0,
        "valid_cross_clothes_pairs": 0.0,
    }


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
        "prcc_dev": _prcc_dev_summary(args),
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
        "source_clothes_classes": _count_clothes_by_source(valid_samples),
    }


def _prcc_dev_summary(args: Namespace) -> dict:
    pids = selected_prcc_dev_pids(args)
    return {
        "identities": len(pids),
        "seed": getattr(args, "prcc_dev_seed", None),
        "pids": pids,
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


def _count_clothes_by_source(samples) -> dict[str, int]:
    clothes: dict[str, set[int]] = {}
    for sample in samples:
        if sample.clothes_id != UNKNOWN_CLOTHES:
            clothes.setdefault(sample.source, set()).add(sample.clothes_id)
    return {source: len(labels) for source, labels in clothes.items()}


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


def _batch_metrics(
    losses: dict[str, torch.Tensor],
    *,
    cal_weight: float,
    prcc_ce_weight: float,
    consistency_weight: float,
    distill_weight: float,
) -> dict[str, str]:
    return {
        "loss": f"{losses['loss'].item():.4f}",
        "ce": f"{losses['ce'].item():.4f}",
        "ce_m": f"{losses['ce_market'].item():.4f}",
        "ce_p": f"{losses['ce_prcc'].item():.4f}",
        "tri": f"{losses['triplet'].item():.4f}",
        "cal": f"{losses['cal'].item():.4f}",
        "sk": f"{losses['sketch'].item():.4f}",
        "con": f"{losses['consistency'].item():.4f}",
        "dis": f"{losses['distill'].item():.4f}",
        "part": f"{losses['part_triplet'].item():.4f}",
        "cloth": f"{losses['cloth_invariant'].item():.4f}",
        "xcloth": f"{losses['cross_clothes_contrastive'].item():.4f}",
        "pairs": f"{losses['valid_cross_clothes_pairs'].item():.0f}",
        "cal_w": f"{cal_weight:.4f}",
        "prcc_ce_w": f"{prcc_ce_weight:.4f}",
        "con_w": f"{consistency_weight:.4f}",
        "dis_w": f"{distill_weight:.4f}",
    }


def _cal_loss(outputs: dict[str, torch.Tensor], clothes_labels: torch.Tensor, cal_weight: float) -> torch.Tensor:
    if cal_weight <= NO_CAL_LOSS:
        return torch.zeros((), device=clothes_labels.device)
    known = clothes_labels.ge(0)
    if not known.any():
        return torch.zeros((), device=clothes_labels.device)
    return F.cross_entropy(outputs["clothes_logits"][known].float(), clothes_labels[known])


def _checkpoint_metadata(request: CheckpointSaveRequest) -> dict[str, int | float | str]:
    return {
        "epoch": request.epoch,
        "best_metric": request.metric_name,
        "best_dataset": request.best_dataset,
        "best_variant": request.variant,
        "best_metric_value": request.metric_value,
        "num_classes": request.dataset.num_classes,
        "num_clothes_classes": request.dataset.num_clothes_classes,
    }


def _model_config(model) -> dict[str, int | float | bool]:
    return {
        "embedding_dim": int(model.embedding_dim),
        "use_part_branch": bool(model.use_part_branch),
        "num_parts": int(model.num_parts),
        "part_embedding_dim": int(model.part_embedding_dim),
        "combined_global_weight": float(model.combined_global_weight),
        "combined_part_weight": float(model.combined_part_weight),
    }


def _print_epoch(epoch: int, metrics: dict[str, float]) -> None:
    print(
        f"epoch={epoch + 1} loss={metrics['loss']:.4f} ce={metrics['ce']:.4f} "
        f"ce_market={metrics['ce_market']:.4f} ce_prcc={metrics['ce_prcc']:.4f} "
        f"triplet={metrics['triplet']:.4f} cal={metrics['cal']:.4f} "
        f"sketch={metrics['sketch']:.4f} consistency={metrics['consistency']:.4f} "
        f"distill={metrics['distill']:.4f} "
        f"part={metrics['part_triplet']:.4f} cloth={metrics['cloth_invariant']:.4f} "
        f"xcloth={metrics['cross_clothes_contrastive']:.4f} "
        f"pairs={metrics['valid_cross_clothes_pairs']:.1f} "
        f"cal_w={metrics['effective_cal_weight']:.4f} "
        f"prcc_ce_w={metrics['effective_prcc_ce_weight']:.4f} "
        f"con_w={metrics['effective_sketch_consistency_weight']:.4f} "
        f"distill_w={metrics['effective_distill_weight']:.4f}"
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


def _effective_prcc_ce_weight(args: Namespace, epoch: int) -> float:
    if args.prcc_ce_ramp_epochs == NO_PRCC_CE_RAMP_EPOCHS:
        return args.prcc_ce_weight
    if args.prcc_ce_ramp_epochs == SINGLE_RAMP_EPOCH:
        return args.prcc_ce_final_weight
    if epoch >= args.prcc_ce_ramp_epochs - 1:
        return args.prcc_ce_final_weight
    progress = epoch / (args.prcc_ce_ramp_epochs - 1)
    return _interpolate(args.prcc_ce_weight, args.prcc_ce_final_weight, progress)


def _effective_sketch_consistency_weight(args: Namespace, epoch: int) -> float:
    if args.rgb_sketch_consistency_weight <= NO_SKETCH_LOSS or epoch < args.sketch_warmup_epochs:
        return NO_SKETCH_LOSS
    ramp_index = epoch - args.sketch_warmup_epochs + 1
    if args.sketch_ramp_epochs == 0 or ramp_index >= args.sketch_ramp_epochs:
        return args.rgb_sketch_consistency_weight
    return args.rgb_sketch_consistency_weight * ramp_index / args.sketch_ramp_epochs


def _effective_distill_weight(args: Namespace, epoch: int) -> float:
    if _max_distill_weight(args) <= NO_DISTILL_LOSS:
        return NO_DISTILL_LOSS
    if epoch < args.distill_hold_epochs:
        return args.distill_weight
    if args.distill_ramp_epochs == 0:
        return args.distill_final_weight
    ramp_index = epoch - args.distill_hold_epochs
    if ramp_index >= args.distill_ramp_epochs:
        return args.distill_final_weight
    progress = ramp_index / args.distill_ramp_epochs
    return _interpolate(args.distill_weight, args.distill_final_weight, progress)


def _interpolate(start: float, stop: float, progress: float) -> float:
    return start + (stop - start) * progress


def _max_distill_weight(args: Namespace) -> float:
    return max(args.distill_weight, args.distill_final_weight)


def _loader_has_persistent_workers(loader) -> bool:
    return bool(getattr(loader, "persistent_workers", False))


def _training_header(args: Namespace, loader, distributed: DistributedContext) -> str:
    parts = [
        f"precision={args.precision}",
        f"lr={scheduler_lrs(args.lr)}",
        f"lr_milestones={args.lr_milestones}",
        f"lr_gamma={args.lr_gamma}",
        f"best_metric={args.best_metric}",
        f"best_dataset={args.best_dataset}",
        f"best_variant={args.best_variant}",
        f"feature_key={args.feature_key}",
        f"triplet_feature_key={args.triplet_feature_key}",
        f"use_part_branch={args.use_part_branch}",
        f"num_parts={args.num_parts}",
        f"part_triplet_weight={args.part_triplet_weight}",
        f"cloth_invariant_weight={args.cloth_invariant_weight}",
        f"combined_global_weight={args.combined_global_weight}",
        f"combined_part_weight={args.combined_part_weight}",
        f"teacher_checkpoint={args.teacher_checkpoint}",
        f"distill_weight={args.distill_weight}",
        f"distill_final_weight={args.distill_final_weight}",
        f"distill_hold_epochs={args.distill_hold_epochs}",
        f"distill_ramp_epochs={args.distill_ramp_epochs}",
        f"prcc_dev_identities={args.prcc_dev_identities}",
        f"prcc_dev_seed={args.prcc_dev_seed}",
        f"prcc_ce_weight={args.prcc_ce_weight}",
        f"prcc_ce_final_weight={args.prcc_ce_final_weight}",
        f"prcc_ce_ramp_epochs={args.prcc_ce_ramp_epochs}",
        f"cross_clothes_contrastive_weight={args.cross_clothes_contrastive_weight}",
        f"contrastive_temperature={args.contrastive_temperature}",
        f"eval_period={args.eval_period}",
        f"freeze_backbone_epochs={args.freeze_backbone_epochs}",
        f"freeze_backbone_layers={args.freeze_backbone_layers}",
        f"freeze_backbone_all_epochs={args.freeze_backbone_all_epochs}",
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
