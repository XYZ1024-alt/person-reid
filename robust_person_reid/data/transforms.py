from __future__ import annotations

import random
from dataclasses import dataclass

from PIL import Image, ImageEnhance
import torch


IMAGE_HEIGHT = 256
IMAGE_WIDTH = 128
RGB_CHANNELS = 3
PIXEL_SCALE = 255.0
FLIP_PROBABILITY = 0.5
DARK_PROBABILITY = 0.25
OCCLUSION_PROBABILITY = 0.5
DARK_FACTOR_MIN = 0.30
DARK_FACTOR_MAX = 0.75
BRIGHTNESS_MIN = 0.85
BRIGHTNESS_MAX = 1.15
CONTRAST_MIN = 0.85
CONTRAST_MAX = 1.15
OCCLUSION_AREA_MIN = 0.08
OCCLUSION_AREA_MAX = 0.22
NORMALIZE_MEAN = (0.485, 0.456, 0.406)
NORMALIZE_STD = (0.229, 0.224, 0.225)
VARIANT_STANDARD = "standard"
VARIANT_DARK = "dark"
VARIANT_OCCLUDED = "occluded"


@dataclass(frozen=True)
class TransformConfig:
    height: int = IMAGE_HEIGHT
    width: int = IMAGE_WIDTH
    train: bool = False
    variant: str = VARIANT_STANDARD


class ReIDTransform:
    def __init__(self, config: TransformConfig):
        self.config = config

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB").resize((self.config.width, self.config.height))
        image = self._apply_image_augments(image)
        tensor = _image_to_tensor(image)
        tensor = self._apply_tensor_augments(tensor)
        return _normalize(tensor)

    def pair(self, image: Image.Image, sketch: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        image = _resize_rgb(image, self.config)
        sketch = _resize_rgb(sketch, self.config)
        if self.config.train and random.random() < FLIP_PROBABILITY:
            image = _flip(image)
            sketch = _flip(sketch)
        image = self._apply_non_geometric_image_augments(image)
        tensor = self._apply_tensor_augments(_image_to_tensor(image))
        sketch_tensor = _image_to_tensor(sketch)
        return _normalize(tensor), _normalize(sketch_tensor)

    def _apply_image_augments(self, image: Image.Image) -> Image.Image:
        if self.config.train and random.random() < FLIP_PROBABILITY:
            image = _flip(image)
        return self._apply_non_geometric_image_augments(image)

    def _apply_non_geometric_image_augments(self, image: Image.Image) -> Image.Image:
        if self.config.train:
            image = _jitter_image(image)
        if self.config.variant == VARIANT_DARK or _train_event(self.config.train, DARK_PROBABILITY):
            image = _darken_image(image)
        return image

    def _apply_tensor_augments(self, tensor: torch.Tensor) -> torch.Tensor:
        should_occlude = self.config.variant == VARIANT_OCCLUDED
        if should_occlude or _train_event(self.config.train, OCCLUSION_PROBABILITY):
            return _occlude_tensor(tensor)
        return tensor


def _image_to_tensor(image: Image.Image) -> torch.Tensor:
    width, height = image.size
    buffer = bytearray(image.tobytes())
    tensor = torch.frombuffer(buffer, dtype=torch.uint8).view(height, width, RGB_CHANNELS)
    return tensor.permute(2, 0, 1).float().div(PIXEL_SCALE)


def _resize_rgb(image: Image.Image, config: TransformConfig) -> Image.Image:
    return image.convert("RGB").resize((config.width, config.height))


def _flip(image: Image.Image) -> Image.Image:
    return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)


def _normalize(tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(NORMALIZE_MEAN).view(RGB_CHANNELS, 1, 1)
    std = torch.tensor(NORMALIZE_STD).view(RGB_CHANNELS, 1, 1)
    return (tensor - mean) / std


def _jitter_image(image: Image.Image) -> Image.Image:
    brightness = random.uniform(BRIGHTNESS_MIN, BRIGHTNESS_MAX)
    contrast = random.uniform(CONTRAST_MIN, CONTRAST_MAX)
    image = ImageEnhance.Brightness(image).enhance(brightness)
    return ImageEnhance.Contrast(image).enhance(contrast)


def _darken_image(image: Image.Image) -> Image.Image:
    factor = random.uniform(DARK_FACTOR_MIN, DARK_FACTOR_MAX)
    return ImageEnhance.Brightness(image).enhance(factor)


def _occlude_tensor(tensor: torch.Tensor) -> torch.Tensor:
    _, height, width = tensor.shape
    area_ratio = random.uniform(OCCLUSION_AREA_MIN, OCCLUSION_AREA_MAX)
    box_height = max(1, int(height * area_ratio))
    box_width = max(1, int(width * area_ratio))
    top = random.randint(0, height - box_height)
    left = random.randint(0, width - box_width)
    tensor[:, top : top + box_height, left : left + box_width] = 0.0
    return tensor


def _train_event(train: bool, probability: float) -> bool:
    return train and random.random() < probability
