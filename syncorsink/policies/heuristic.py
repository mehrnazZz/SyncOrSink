from __future__ import annotations

import numpy as np

# Simple heuristic for signal_hunt: move toward visible clue/target.

def heuristic_policy(env):
    def _policy(obs, info, state):
        actions = {}
        for agent_id in range(env.num_agents):
            local = obs[agent_id]["local_grid"]
            radius = local.shape[0] // 2
            target = _find_first(local, [5, 6])  # clue or target
            if target is None:
                action_id = int(np.random.randint(0, 8))
            else:
                dx = target[0] - radius
                dy = target[1] - radius
                action_id = _move_action(dx, dy, env)
            actions[agent_id] = {"action": action_id, "message_tokens": []}
        return actions

    return _policy


def _find_first(local, values):
    for y in range(local.shape[0]):
        for x in range(local.shape[1]):
            if int(local[y, x]) in values:
                return (x, y)
    return None


def _move_action(dx, dy, env):
    if abs(dx) > abs(dy):
        return env.ACTION_RIGHT if dx > 0 else env.ACTION_LEFT
    if abs(dy) > 0:
        return env.ACTION_DOWN if dy > 0 else env.ACTION_UP
    return env.ACTION_INTERACT
