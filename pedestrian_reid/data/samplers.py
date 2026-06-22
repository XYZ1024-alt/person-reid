from __future__ import annotations

import random
from collections import defaultdict

from torch.utils.data import Sampler

from pedestrian_reid.data.datasets import UNKNOWN_CLOTHES, ReidSample


MIN_INSTANCES_PER_IDENTITY = 2


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


def _group_indices_by_label_clothes(samples: list[ReidSample]) -> dict[int, dict[int, list[int]]]:
    grouped: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, sample in enumerate(samples):
        if not sample.is_junk and sample.clothes_id != UNKNOWN_CLOTHES:
            grouped[sample.label][sample.clothes_id].append(index)
    return {label: dict(indices_by_clothes) for label, indices_by_clothes in grouped.items()}


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
