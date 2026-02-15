from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SplitSpec:
    name: str
    base_seed: int
    count: int


def make_split_seeds(split: SplitSpec) -> list[int]:
    return [split.base_seed + i for i in range(split.count)]


def split_from_name(name: str) -> SplitSpec:
    if name == "train":
        return SplitSpec(name="train", base_seed=1000, count=64)
    if name == "val":
        return SplitSpec(name="val", base_seed=2000, count=16)
    if name == "test":
        return SplitSpec(name="test", base_seed=3000, count=32)
    raise ValueError(f"Unknown split: {name}")


def seed_for_variant(split: str, variant: int) -> int:
    spec = split_from_name(split)
    return spec.base_seed + (variant % spec.count)
