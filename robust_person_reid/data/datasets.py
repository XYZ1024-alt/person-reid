from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
MARKET_SOURCE = "market1501"
PRCC_SOURCE = "prcc"
JUNK_LABEL = -1
UNKNOWN_CLOTHES = -1
PRCC_SAME_CLOTHES = 0
PRCC_CHANGED_CLOTHES = 1
PRCC_QUERY_CAMERA = "C"
PRCC_GALLERY_CAMERA = "A"
PRCC_CAMERAS = {"A": 1, "B": 2, "C": 3}
SPLIT_TRAIN = "train"
SPLIT_QUERY = "query"
SPLIT_GALLERY = "gallery"


@dataclass(frozen=True)
class ReidSample:
    source: str
    path: Path
    sketch_path: Path | None
    pid: int
    camid: int
    clothes_id: int
    label: int
    is_junk: bool


class ReIDDataset(Dataset):
    def __init__(self, samples: list[ReidSample], transform: Callable):
        if not samples:
            raise ValueError("ReIDDataset received no samples")
        self.samples = samples
        self.transform = transform
        self.num_classes = len({sample.label for sample in samples if not sample.is_junk})
        self.num_clothes_classes = len({sample.clothes_id for sample in samples if sample.clothes_id != UNKNOWN_CLOTHES})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        tensor, sketch_tensor = self._load_tensors(sample)
        return {
            "image": tensor,
            "sketch_image": sketch_tensor,
            "has_sketch": sample.sketch_path is not None,
            "label": sample.label,
            "pid": sample.pid,
            "camid": sample.camid,
            "clothes_id": sample.clothes_id,
            "clothes_label": sample.clothes_id,
            "is_junk": sample.is_junk,
            "path": str(sample.path),
            "source": sample.source,
        }

    def _load_tensors(self, sample: ReidSample) -> tuple:
        with Image.open(sample.path) as image:
            if sample.sketch_path is None:
                tensor = self.transform(image)
                return tensor, tensor.new_zeros(tensor.shape)
            with Image.open(sample.sketch_path) as sketch:
                return self.transform.pair(image, sketch)


def load_market_samples(root: str | Path, split: str) -> list[ReidSample]:
    split_dir = Path(root) / "pytorch" / split
    _require_dir(split_dir)
    samples = [_market_sample(path) for path in _image_files(split_dir)]
    return _filter_train_junk(samples, split)


def load_prcc_samples(root: str | Path, split: str, use_sketch: bool = False) -> list[ReidSample]:
    split_dir = _prcc_split_dir(Path(root), split, "rgb")
    _require_dir(split_dir)
    sketch_dir = _prcc_split_dir(Path(root), split, "sketch") if use_sketch else None
    samples = [_prcc_sample(path, split_dir, sketch_dir) for path in _image_files(split_dir)]
    return _filter_train_junk(samples, split)


def relabel_samples(samples: list[ReidSample]) -> list[ReidSample]:
    label_by_identity = _label_map((sample.source, sample.pid) for sample in samples if not sample.is_junk)
    label_by_clothes = _label_map(_known_clothes_identity(sample) for sample in samples)
    return [_with_relabel(sample, label_by_identity, label_by_clothes) for sample in samples]


def _with_relabel(
    sample: ReidSample,
    label_by_identity: dict[tuple[str, int], int],
    label_by_clothes: dict[tuple[str, int, int], int],
) -> ReidSample:
    if sample.is_junk:
        return replace(sample, label=JUNK_LABEL)
    clothes_id = _relabel_clothes(sample, label_by_clothes)
    return replace(sample, label=label_by_identity[(sample.source, sample.pid)], clothes_id=clothes_id)


def _label_map(identities) -> dict:
    unique_identities = sorted(set(identity for identity in identities if identity is not None))
    return {identity: label for label, identity in enumerate(unique_identities)}


def _known_clothes_identity(sample: ReidSample) -> tuple[str, int, int] | None:
    if sample.clothes_id == UNKNOWN_CLOTHES:
        return None
    return sample.source, sample.pid, sample.clothes_id


def _relabel_clothes(sample: ReidSample, label_by_clothes: dict[tuple[str, int, int], int]) -> int:
    clothes_identity = _known_clothes_identity(sample)
    if clothes_identity is None:
        return UNKNOWN_CLOTHES
    return label_by_clothes[clothes_identity]


def _market_sample(path: Path) -> ReidSample:
    stem_parts = path.stem.split("_")
    if len(stem_parts) < 2 or not stem_parts[1].startswith("c"):
        raise ValueError(f"Invalid Market-1501 filename: {path.name}")
    pid = int(stem_parts[0])
    camid = int(stem_parts[1][1])
    is_junk = pid <= 0
    return ReidSample(MARKET_SOURCE, path, None, pid, camid, UNKNOWN_CLOTHES, pid, is_junk)


def _prcc_sample(path: Path, rgb_dir: Path, sketch_dir: Path | None) -> ReidSample:
    camera = _prcc_camera(path)
    pid = _prcc_pid(path)
    clothes_id = PRCC_CHANGED_CLOTHES if camera == PRCC_QUERY_CAMERA else PRCC_SAME_CLOTHES
    sketch_path = _matching_sketch_path(path, rgb_dir, sketch_dir)
    return ReidSample(PRCC_SOURCE, path, sketch_path, pid, PRCC_CAMERAS[camera], clothes_id, pid, False)


def _prcc_split_dir(root: Path, split: str, modality: str) -> Path:
    base = _prcc_modality_base(root, modality)
    if split == SPLIT_QUERY:
        return _first_existing([base / SPLIT_QUERY, base / "test" / PRCC_QUERY_CAMERA])
    if split == SPLIT_GALLERY:
        return _first_existing([base / SPLIT_GALLERY, base / "test" / PRCC_GALLERY_CAMERA])
    return base / split


def _prcc_modality_base(root: Path, modality: str) -> Path:
    if (root / modality).exists():
        return root / modality
    if root.name in {"rgb", "sketch"}:
        return root.parent / modality
    return root


def _matching_sketch_path(path: Path, rgb_dir: Path, sketch_dir: Path | None) -> Path | None:
    if sketch_dir is None:
        return None
    sketch_path = sketch_dir / path.relative_to(rgb_dir)
    if not sketch_path.exists():
        raise FileNotFoundError(f"Missing PRCC sketch pair for {path}: {sketch_path}")
    return sketch_path


def _filter_train_junk(samples: list[ReidSample], split: str) -> list[ReidSample]:
    if split != SPLIT_TRAIN:
        return samples
    return [sample for sample in samples if not sample.is_junk]


def _image_files(split_dir: Path) -> list[Path]:
    return sorted(path for path in split_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def _require_dir(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_dir():
        raise NotADirectoryError(path)


def _prcc_camera(path: Path) -> str:
    filename_camera = _camera_from_name(path.stem)
    if filename_camera:
        return filename_camera
    for part in reversed(path.parts):
        camera = part.upper()
        if camera in PRCC_CAMERAS:
            return camera
    raise ValueError(f"Cannot infer PRCC camera A/B/C from path: {path}")


def _camera_from_name(stem: str) -> str | None:
    match = re.match(r"([ABC])[_-]", stem.upper())
    if match:
        return match.group(1)
    return None


def _prcc_pid(path: Path) -> int:
    for parent in path.parents:
        if parent.name.isdigit():
            return int(parent.name)
    match = re.match(r"(\d+)", path.stem)
    if match:
        return int(match.group(1))
    raise ValueError(f"Cannot infer PRCC pid from path: {path}")


def _first_existing(candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]
