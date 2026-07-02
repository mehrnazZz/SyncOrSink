from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict

from .spec_validate import validate_spec


@dataclass
class EvalSpec:
    scenario: str
    split: str | None
    episodes: int
    map_variant: int
    policy: str
    mode: str  # "marl" or "llm"
    track: str = "dtde"
    map_size: int = 16
    num_agents: int = 3
    fov_preset: str = "medium"
    max_steps: int = 300
    comm_mode: str = "tokens"
    energy_preset: str = "hard"
    policy_checkpoint: str | None = None
    comm_mat_deterministic: bool = True
    comm_mat_send_threshold: float = 0.5


def load_spec(path: str) -> EvalSpec:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    validate_spec(data)
    return EvalSpec(
        scenario=data["scenario"],
        split=data.get("split"),
        episodes=int(data.get("episodes", 1)),
        map_variant=int(data.get("map_variant", 0)),
        policy=data.get("policy", "random"),
        mode=data.get("mode", "marl"),
        track=data.get("track", "dtde"),
        map_size=int(data.get("map_size", 16)),
        num_agents=int(data.get("agents", data.get("num_agents", 3))),
        fov_preset=data.get("fov_preset", "medium"),
        max_steps=int(data.get("max_steps", 300)),
        comm_mode=data.get("comm_mode", "tokens"),
        energy_preset=data.get("energy_preset", "hard"),
        policy_checkpoint=data.get("policy_checkpoint"),
        comm_mat_deterministic=bool(data.get("comm_mat_deterministic", True)),
        comm_mat_send_threshold=float(data.get("comm_mat_send_threshold", 0.5)),
    )
