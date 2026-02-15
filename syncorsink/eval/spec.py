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
    )
