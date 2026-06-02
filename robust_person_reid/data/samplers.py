from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass

from torch.utils.data import Sampler

from robust_person_reid.data.datasets import MARKET_SOURCE, PRCC_SOURCE, ReidSample


MIN_INSTANCES_PER_IDENTITY = 2
MIN_SOURCE_RATIO = 0.0
MAX_SOURCE_RATIO = 1.0


@dataclass(frozen=True, kw_only=True)
class SourceBalancedSamplerConfig:
    samples: list[ReidSample]
    batch_size: int
    instances: int
    source_ratio: float
    balanced_source: str = PRCC_SOURCE
    other_source: str = MARKET_SOURCE


class IdentityBatchSampler(Sampler[list[int]]):
    def __init__(self, samples: list[ReidSample], batch_size: int, instances: int):
        if instances < MIN_INSTANCES_PER_IDENTITY:
            raise ValueError("instances must be >= 2 for triplet training")
        if batch_size % instances != 0:
            raise ValueError("batch_size must be divisible by instances")
        self.batch_size = batch_size
        self.instances = instances
        self.identities_per_batch = batch_size // instances
        self.index_by_label = _group_indices(samples)
        self.labels = sorted(self.index_by_label)
        self.num_batches = len(samples) // batch_size
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
        self.source_counts = _source_identity_counts(self.identities_per_batch, config.source_ratio)
        self.balanced_source = config.balanced_source
        self.other_source = config.other_source
        self.num_batches = len(config.samples) // config.batch_size
        self._validate_sources()

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
