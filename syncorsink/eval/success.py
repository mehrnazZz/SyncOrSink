from __future__ import annotations

from typing import Any, Mapping


def episode_success(scenario: str | None, done: bool, info: Mapping[str, Any] | None) -> bool:
    """Return the benchmark success flag for a finished episode."""
    if scenario == "energy_grid":
        return bool((info or {}).get("success", False))
    return bool(done)
