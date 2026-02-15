from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class FovPreset:
    name: str
    radius: int


FOV_PRESETS = {
    "hard": FovPreset("hard", 2),
    "medium": FovPreset("medium", 3),
    "easy": FovPreset("easy", 4),
}


def get_rng(seed: int | None) -> np.random.Generator:
    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng(seed)


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
