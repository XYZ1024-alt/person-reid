from __future__ import annotations

import torch


TORCH_SHARING_STRATEGY = "file_system"


def configure_torch_runtime() -> None:
    torch.multiprocessing.set_sharing_strategy(TORCH_SHARING_STRATEGY)
