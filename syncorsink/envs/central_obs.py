from __future__ import annotations

import numpy as np


def build_central_obs(env) -> dict:
    # full grid and agent states
    return {
        "grid": np.array(env.grid, copy=True),
        "agent_positions": np.array(env.agent_positions, dtype=np.int16),
        "inventories": np.array(env.inventories, dtype=np.int16),
        "steps": int(env.steps),
    }
