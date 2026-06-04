from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


INPUT_CHANNELS = 3
STEM_CHANNELS = 64
BOTTLENECK_EXPANSION = 4
EMBEDDING_DIM = 256
REID_FEATURE_DIM = 2048
CONV1_KERNEL = 7
CONV3_KERNEL = 3
POINTWISE_KERNEL = 1
CONV1_PADDING = 3
CONV3_PADDING = 1
STEM_STRIDE = 2
LAST_STRIDE = 1
HALF_RATIO = 2
GRAD_REVERSE_SCALE = 1.0
IMAGENET_LOADED_PREFIX = "Loaded ImageNet pretrained backbone parameters"


@dataclass(frozen=True)
class BlockConfig:
    in_channels: int
    channels: int
    stride: int
    use_ibn: bool
    downsample: nn.Module | None


@dataclass(frozen=True)
class LayerConfig:
    channels: int
    blocks: int
    stride: int
    use_ibn: bool


class IBN(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        half_channels = channels // HALF_RATIO
        self.split = half_channels
        self.instance_norm = nn.InstanceNorm2d(half_channels, affine=True)
        self.batch_norm = nn.BatchNorm2d(channels - half_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        first, second = torch.split(inputs, [self.split, inputs.size(1) - self.split], dim=1)
        return torch.cat((self.instance_norm(first), self.batch_norm(second)), dim=1)


class BottleneckIBN(nn.Module):
    expansion = BOTTLENECK_EXPANSION

    def __init__(self, config: BlockConfig):
        super().__init__()
        out_channels = config.channels * self.expansion
        self.conv1 = _conv1x1(config.in_channels, config.channels)
        self.norm1 = IBN(config.channels) if config.use_ibn else nn.BatchNorm2d(config.channels)
        self.conv2 = _conv3x3(config.channels, config.channels, config.stride)
        self.bn2 = nn.BatchNorm2d(config.channels)
        self.conv3 = _conv1x1(config.channels, out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = config.downsample

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs
        outputs = self.relu(self.norm1(self.conv1(inputs)))
        outputs = self.relu(self.bn2(self.conv2(outputs)))
        outputs = self.bn3(self.conv3(outputs))
        if self.downsample is not None:
            residual = self.downsample(inputs)
        return self.relu(outputs + residual)


class ResNet50IBNBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channels = STEM_CHANNELS
        self.stem = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, STEM_CHANNELS, CONV1_KERNEL, STEM_STRIDE, CONV1_PADDING, bias=False),
            nn.BatchNorm2d(STEM_CHANNELS),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=CONV3_KERNEL, stride=STEM_STRIDE, padding=CONV3_PADDING),
        )
        self.layer1 = self._make_layer(LayerConfig(64, 3, 1, True))
        self.layer2 = self._make_layer(LayerConfig(128, 4, 2, True))
        self.layer3 = self._make_layer(LayerConfig(256, 6, 2, True))
        self.layer4 = self._make_layer(LayerConfig(512, 3, LAST_STRIDE, False))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        features = self.layer1(features)
        features = self.layer2(features)
        features = self.layer3(features)
        return self.layer4(features)

    def _make_layer(self, config: LayerConfig) -> nn.Sequential:
        downsample = _downsample(self.in_channels, config.channels, config.stride)
        blocks = [self._build_block(config, config.stride, downsample)]
        self.in_channels = config.channels * BOTTLENECK_EXPANSION
        for _ in range(1, config.blocks):
            blocks.append(self._build_block(config, 1, None))
        return nn.Sequential(*blocks)

    def _build_block(self, config: LayerConfig, stride: int, downsample: nn.Module | None) -> BottleneckIBN:
        block_config = BlockConfig(self.in_channels, config.channels, stride, config.use_ibn, downsample)
        return BottleneckIBN(block_config)


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = scale
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx, gradients: torch.Tensor):
        return gradients.neg().mul(ctx.scale), None


class RobustPersonReIDNet(nn.Module):
    def __init__(self, num_classes: int, embedding_dim: int = EMBEDDING_DIM, num_clothes_classes: int = 0):
        super().__init__()
        self.backbone = ResNet50IBNBackbone()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embedding = nn.Linear(REID_FEATURE_DIM, embedding_dim, bias=False)
        self.bnneck = nn.BatchNorm1d(embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes, bias=False)
        self.clothes_classifier = _clothes_classifier(embedding_dim, num_clothes_classes)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        pooled = self.pool(self.backbone(images)).flatten(1)
        embedding = self.bnneck(self.embedding(pooled))
        outputs = {"logits": self.classifier(embedding), "features": F.normalize(embedding, dim=1)}
        if self.clothes_classifier is not None:
            reversed_features = GradientReverse.apply(embedding, GRAD_REVERSE_SCALE)
            outputs["clothes_logits"] = self.clothes_classifier(reversed_features)
        return outputs


def _conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, POINTWISE_KERNEL, stride=stride, bias=False)


def _conv3x3(in_channels: int, out_channels: int, stride: int) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, CONV3_KERNEL, stride, CONV3_PADDING, bias=False)


def _downsample(in_channels: int, channels: int, stride: int) -> nn.Module | None:
    out_channels = channels * BOTTLENECK_EXPANSION
    if stride == 1 and in_channels == out_channels:
        return None
    return nn.Sequential(_conv1x1(in_channels, out_channels, stride), nn.BatchNorm2d(out_channels))


def _clothes_classifier(embedding_dim: int, num_clothes_classes: int) -> nn.Linear | None:
    if num_clothes_classes <= 0:
        return None
    return nn.Linear(embedding_dim, num_clothes_classes, bias=True)


def load_imagenet_pretrained_backbone(backbone: ResNet50IBNBackbone, *, verbose: bool = True) -> int:
    from torchvision.models import ResNet50_Weights, resnet50

    source = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).state_dict()
    loaded = _load_stem(backbone, source)
    loaded += _load_resnet_layers(backbone, source)
    if verbose:
        print(f"{IMAGENET_LOADED_PREFIX}: {loaded}")
    return loaded


def _load_stem(backbone: ResNet50IBNBackbone, source: dict[str, torch.Tensor]) -> int:
    _copy_parameter(backbone.stem[0].weight, source["conv1.weight"])
    _load_batch_norm(backbone.stem[1], source, "bn1")
    return 6


def _load_resnet_layers(backbone: ResNet50IBNBackbone, source: dict[str, torch.Tensor]) -> int:
    loaded = 0
    for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
        loaded += _load_layer(getattr(backbone, layer_name), source, layer_name)
    return loaded


def _load_layer(layer: nn.Sequential, source: dict[str, torch.Tensor], layer_name: str) -> int:
    loaded = 0
    for block_index, block in enumerate(layer):
        prefix = f"{layer_name}.{block_index}"
        loaded += _load_block(block, source, prefix)
    return loaded


def _load_block(block: BottleneckIBN, source: dict[str, torch.Tensor], prefix: str) -> int:
    loaded = _load_block_convolutions(block, source, prefix)
    loaded += _load_norm1(block.norm1, source, f"{prefix}.bn1")
    loaded += _load_batch_norm(block.bn2, source, f"{prefix}.bn2")
    loaded += _load_batch_norm(block.bn3, source, f"{prefix}.bn3")
    if block.downsample is not None:
        loaded += _load_downsample(block.downsample, source, f"{prefix}.downsample")
    return loaded


def _load_block_convolutions(block: BottleneckIBN, source: dict[str, torch.Tensor], prefix: str) -> int:
    _copy_parameter(block.conv1.weight, source[f"{prefix}.conv1.weight"])
    _copy_parameter(block.conv2.weight, source[f"{prefix}.conv2.weight"])
    _copy_parameter(block.conv3.weight, source[f"{prefix}.conv3.weight"])
    return 3


def _load_norm1(norm: nn.Module, source: dict[str, torch.Tensor], prefix: str) -> int:
    if isinstance(norm, IBN):
        return _load_ibn(norm, source, prefix)
    return _load_batch_norm(norm, source, prefix)


def _load_ibn(norm: IBN, source: dict[str, torch.Tensor], prefix: str) -> int:
    _load_batch_norm_suffix(norm.batch_norm, source, prefix, norm.split)
    return 5


def _load_downsample(downsample: nn.Sequential, source: dict[str, torch.Tensor], prefix: str) -> int:
    _copy_parameter(downsample[0].weight, source[f"{prefix}.0.weight"])
    return 1 + _load_batch_norm(downsample[1], source, f"{prefix}.1")


def _load_batch_norm(module: nn.BatchNorm2d, source: dict[str, torch.Tensor], prefix: str) -> int:
    _copy_parameter(module.weight, source[f"{prefix}.weight"])
    _copy_parameter(module.bias, source[f"{prefix}.bias"])
    _copy_buffer(module.running_mean, source[f"{prefix}.running_mean"])
    _copy_buffer(module.running_var, source[f"{prefix}.running_var"])
    _copy_buffer(module.num_batches_tracked, source[f"{prefix}.num_batches_tracked"])
    return 5


def _load_batch_norm_suffix(module: nn.BatchNorm2d, source: dict[str, torch.Tensor], prefix: str, start: int) -> None:
    _copy_parameter(module.weight, source[f"{prefix}.weight"][start:])
    _copy_parameter(module.bias, source[f"{prefix}.bias"][start:])
    _copy_buffer(module.running_mean, source[f"{prefix}.running_mean"][start:])
    _copy_buffer(module.running_var, source[f"{prefix}.running_var"][start:])
    _copy_buffer(module.num_batches_tracked, source[f"{prefix}.num_batches_tracked"])


def _copy_parameter(target: nn.Parameter, source: torch.Tensor) -> None:
    target.data.copy_(source)


def _copy_buffer(target: torch.Tensor, source: torch.Tensor) -> None:
    target.copy_(source)
