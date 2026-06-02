from __future__ import annotations

import random
from collections import defaultdict

from torch.utils.data import Sampler

from robust_person_reid.data.datasets import ReidSample


MIN_INSTANCES_PER_IDENTITY = 2


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

