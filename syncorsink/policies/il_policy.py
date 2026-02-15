from __future__ import annotations

from typing import Dict, Any

from .base import BasePolicy
from .registry import register


@register("il")
class ILPolicy(BasePolicy):
    """
    Adapter for imitation learning models with predict(obs)->action API.
    """

    def __init__(self, model):
        self.model = model

    def __call__(self, obs: dict, info: dict, state: dict) -> Dict[int, dict]:
        actions: Dict[int, dict] = {}
        for agent_id in obs.keys():
            action = self.model.predict(obs[agent_id])
            if isinstance(action, dict):
                actions[agent_id] = action
            else:
                actions[agent_id] = {"action": int(action), "message_tokens": []}
        return actions
