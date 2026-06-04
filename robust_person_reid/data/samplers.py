from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass

from torch.utils.data import Sampler

from robust_person_reid.data.datasets import MARKET_SOURCE, PRCC_SOURCE, UNKNOWN_CLOTHES, ReidSample


MIN_INSTANCES_PER_IDENTITY = 2
MIN_SOURCE_RATIO = 0.0
MAX_SOURCE_RATIO = 1.0


@dataclass(frozen=True, kw_only=True)
class SourceBalancedSamplerConfig:
    samples: list[ReidSample]
    batch_size: int
    instances: int
    source_ratio: float
    epoch_batch_size: int | None = None
    balanced_source: str = PRCC_SOURCE
    other_source: str = MARKET_SOURCE


class IdentityBatchSampler(Sampler[list[int]]):
    def __init__(self, samples: list[ReidSample], batch_size: int, instances: int, *, epoch_batch_size: int | None = None):
        if instances < MIN_INSTANCES_PER_IDENTITY:
            raise ValueError("instances must be >= 2 for triplet training")
        if batch_size % instances != 0:
            raise ValueError("batch_size must be divisible by instances")
        self.batch_size = batch_size
        self.instances = instances
        self.identities_per_batch = batch_size // instances
        self.index_by_label = _group_indices(samples)
        self.labels = sorted(self.index_by_label)
        self.num_batches = _num_batches(len(samples), epoch_batch_size or batch_size)
        if len(self.labels) < self.identities_per_batch:
            raise ValueError("Not enough identities for one identity-balanced batch")

    def __iter__(self):
        for _ in range(self.num_batches):
            labels = random.sample(self.labels, self.identities_per_batch)
            yield [index for label in labels for index in self._sample_identity(label)]

    def __len__(self) -> int:
        return self.num_batches

    def _sample_identity(self, label: int) -> list[int]:
        indices = self.index_by_label[label]
        if len(indices) >= self.instances:
            return random.sample(indices, self.instances)
        return random.choices(indices, k=self.instances)


class ClothesAwareIdentityBatchSampler(Sampler[list[int]]):
    def __init__(self, samples: list[ReidSample], batch_size: int, instances: int, *, epoch_batch_size: int | None = None):
        _validate_batch_config(batch_size, instances)
        self.batch_size = batch_size
        self.instances = instances
        self.identities_per_batch = batch_size // instances
        self.index_by_label_clothes = _group_indices_by_label_clothes(samples)
        self.labels = sorted(self.index_by_label_clothes)
        self.num_batches = _num_batches(len(samples), epoch_batch_size or batch_size)
        _validate_identity_count(self.labels, self.identities_per_batch)
        _validate_clothes_aware_groups(self.index_by_label_clothes)

    def __iter__(self):
        for _ in range(self.num_batches):
            labels = random.sample(self.labels, self.identities_per_batch)
            yield [index for label in labels for index in self._sample_identity(label)]

    def __len__(self) -> int:
        return self.num_batches

    def _sample_identity(self, label: int) -> list[int]:
        return _sample_clothes_aware(self.index_by_label_clothes[label], self.instances)


def _group_indices(samples: list[ReidSample]) -> dict[int, list[int]]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        if not sample.is_junk:
            grouped[sample.label].append(index)
    return dict(grouped)


class SourceBalancedIdentityBatchSampler(Sampler[list[int]]):
    def __init__(self, config: SourceBalancedSamplerConfig):
        _validate_batch_config(config.batch_size, config.instances)
        _validate_source_ratio(config.source_ratio)
        self.batch_size = config.batch_size
        self.instances = config.instances
        self.identities_per_batch = config.batch_size // config.instances
        self.index_by_source_label = _group_indices_by_source(config.samples)
        self.index_by_source_label_clothes = _group_indices_by_source_label_clothes(config.samples)
        self.source_counts = _source_identity_counts(self.identities_per_batch, config.source_ratio)
        self.balanced_source = config.balanced_source
        self.other_source = config.other_source
        self.num_batches = _num_batches(len(config.samples), config.epoch_batch_size or config.batch_size)
        self._validate_sources()
        _validate_clothes_aware_groups(self.index_by_source_label_clothes[self.balanced_source])

    def __iter__(self):
        for _ in range(self.num_batches):
            indices = self._sample_source(self.balanced_source) + self._sample_source(self.other_source)
            random.shuffle(indices)
            yield indices

    def __len__(self) -> int:
        return self.num_batches

    def _sample_source(self, source: str) -> list[int]:
        count = self.source_counts[source]
        labels = random.sample(sorted(self.index_by_source_label[source]), count)
        return [index for label in labels for index in self._sample_identity(source, label)]

    def _sample_identity(self, source: str, label: int) -> list[int]:
        if source == self.balanced_source:
            return _sample_clothes_aware(self.index_by_source_label_clothes[source][label], self.instances)
        indices = self.index_by_source_label[source][label]
        if len(indices) >= self.instances:
            return random.sample(indices, self.instances)
        return random.choices(indices, k=self.instances)

    def _validate_sources(self) -> None:
        for source, required in self.source_counts.items():
            available = len(self.index_by_source_label.get(source, {}))
            if available < required:
                raise ValueError(f"{source} needs {required} identities per batch, but only {available} are available")


def _validate_batch_config(batch_size: int, instances: int) -> None:
    if instances < MIN_INSTANCES_PER_IDENTITY:
        raise ValueError("instances must be >= 2 for triplet training")
    if batch_size % instances != 0:
        raise ValueError("batch_size must be divisible by instances")


def _num_batches(sample_count: int, batch_size: int) -> int:
    batches = sample_count // batch_size
    if batches <= 0:
        raise ValueError(f"Not enough samples for one batch: samples={sample_count} batch_size={batch_size}")
    return batches


def _validate_identity_count(labels: list[int], identities_per_batch: int) -> None:
    if len(labels) < identities_per_batch:
        raise ValueError("Not enough identities for one identity-balanced batch")


def _validate_clothes_aware_groups(index_by_label_clothes: dict[int, dict[int, list[int]]]) -> None:
    invalid = [label for label, grouped in index_by_label_clothes.items() if len(grouped) < 2]
    if invalid:
        raise ValueError(f"clothes-aware sampling requires at least 2 clothes labels per identity; invalid={invalid[:10]}")


def _validate_source_ratio(source_ratio: float) -> None:
    if source_ratio <= MIN_SOURCE_RATIO or source_ratio >= MAX_SOURCE_RATIO:
        raise ValueError("source_ratio must be in (0, 1) for source-balanced joint training")


def _source_identity_counts(identities_per_batch: int, source_ratio: float) -> dict[str, int]:
    balanced_count = round(identities_per_batch * source_ratio)
    if balanced_count <= 0 or balanced_count >= identities_per_batch:
        raise ValueError("source_ratio must allocate at least one identity to each source")
    return {PRCC_SOURCE: balanced_count, MARKET_SOURCE: identities_per_batch - balanced_count}


def _group_indices_by_source(samples: list[ReidSample]) -> dict[str, dict[int, list[int]]]:
    grouped: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, sample in enumerate(samples):
        if not sample.is_junk:
            grouped[sample.source][sample.label].append(index)
    return {source: dict(indices_by_label) for source, indices_by_label in grouped.items()}


def _group_indices_by_label_clothes(samples: list[ReidSample]) -> dict[int, dict[int, list[int]]]:
    grouped: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, sample in enumerate(samples):
        if not sample.is_junk and sample.clothes_id != UNKNOWN_CLOTHES:
            grouped[sample.label][sample.clothes_id].append(index)
    return {label: dict(indices_by_clothes) for label, indices_by_clothes in grouped.items()}


def _group_indices_by_source_label_clothes(samples: list[ReidSample]) -> dict[str, dict[int, dict[int, list[int]]]]:
    grouped: dict[str, dict[int, dict[int, list[int]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for index, sample in enumerate(samples):
        if not sample.is_junk and sample.clothes_id != UNKNOWN_CLOTHES:
            grouped[sample.source][sample.label][sample.clothes_id].append(index)
    return {source: _freeze_nested(grouped_by_label) for source, grouped_by_label in grouped.items()}


def _freeze_nested(grouped_by_label) -> dict[int, dict[int, list[int]]]:
    return {label: dict(indices_by_clothes) for label, indices_by_clothes in grouped_by_label.items()}


def _sample_clothes_aware(index_by_clothes: dict[int, list[int]], instances: int) -> list[int]:
    clothes_ids = random.sample(sorted(index_by_clothes), MIN_INSTANCES_PER_IDENTITY)
    quotas = _clothes_quotas(instances)
    sampled = [_sample_from_clothes(index_by_clothes[clothes_id], quota) for clothes_id, quota in zip(clothes_ids, quotas)]
    return [index for indices in sampled for index in indices]


def _clothes_quotas(instances: int) -> list[int]:
    first = instances // MIN_INSTANCES_PER_IDENTITY
    return [first, instances - first]


def _sample_from_clothes(indices: list[int], count: int) -> list[int]:
    if len(indices) >= count:
        return random.sample(indices, count)
    return random.choices(indices, k=count)
