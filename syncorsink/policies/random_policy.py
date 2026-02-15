from __future__ import annotations

import numpy as np


def random_policy(action_space, num_agents: int):
    def _policy(obs, info, state):
        actions = {}
        for agent_id in range(num_agents):
            action_id = int(np.random.randint(0, 8))
            actions[agent_id] = {"action": action_id, "message_tokens": []}
        return actions

    return _policy
