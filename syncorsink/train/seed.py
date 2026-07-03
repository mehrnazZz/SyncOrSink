from __future__ import annotations

import random

import numpy as np
import torch


def set_global_seeds(seed: int | None) -> None:
    """Seed Python, NumPy, and Torch RNGs for reproducible training runs."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
