from __future__ import annotations

from typing import Dict

from .base import BasePolicy
from .registry import register


@register("mappo")
class MAPPOPolicy(BasePolicy):
    """
    Adapter for a MAPPO-style policy.

    Expects a model with `act(obs_dict) -> actions` where obs_dict is per-agent,
    or `act(obs, agent_id)` if `per_agent=True`.
    """

    def __init__(self, model, per_agent: bool = False):
        self.model = model
        self.per_agent = per_agent

    def __call__(self, obs: dict, info: dict, state: dict) -> Dict[int, dict]:
        if not self.per_agent:
            return self.model.act(obs)
        actions: Dict[int, dict] = {}
        for agent_id in obs.keys():
            action = self.model.act(obs[agent_id], agent_id)
            if isinstance(action, dict):
                actions[agent_id] = action
            else:
                actions[agent_id] = {"action": int(action), "message_tokens": []}
        return actions
