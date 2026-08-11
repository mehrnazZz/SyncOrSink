from __future__ import annotations

import numpy as np


class RandomPolicy:
    def __init__(self, action_space, num_agents: int):
        self.action_space = action_space
        self.num_agents = int(num_agents)
        self._rng = np.random.default_rng()

    def reset(self, episode=None, seed=None):
        self._rng = np.random.default_rng(seed)

    def __call__(self, obs, info, state):
        actions = {}
        for agent_id in range(self.num_agents):
            action_id = int(self._rng.integers(0, _num_actions(self.action_space)))
            actions[agent_id] = {"action": action_id, "message_tokens": []}
        return actions


def random_policy(action_space, num_agents: int):
    return RandomPolicy(action_space, num_agents)


def _num_actions(action_space) -> int:
    action_branch = getattr(action_space, "spaces", {}).get("action") if action_space is not None else None
    return int(getattr(action_branch, "n", 8))
