from __future__ import annotations

from typing import Dict, Any

import numpy as np

from .base import BasePolicy
from .registry import register


@register("torch")
class TorchPolicy(BasePolicy):
    """
    Adapter for a PyTorch model. Expects a callable `model(obs_dict) -> actions`.
    This is a thin wrapper and leaves batching to the model.
    """

    def __init__(self, model, device: str = "cpu"):
        self.model = model
        self.device = device

    def __call__(self, obs: dict, info: dict, state: dict) -> Dict[int, dict]:
        # Pass through; user model should handle tensors.
        return self.model(obs)
