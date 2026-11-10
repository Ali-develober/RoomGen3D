"""Shared helpers: reproducible seeds and freeing GPU memory between pipeline stages."""

import gc
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Fix Python, NumPy and PyTorch random seeds so results are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def free_gpu(*objs) -> None:
    """Delete the given models/tensors and release cached GPU memory.

    Call this before loading the next large model (8 GB VRAM limit).
    Note: the caller should also drop its own references (e.g. `model = None`).
    """
    for obj in objs:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
