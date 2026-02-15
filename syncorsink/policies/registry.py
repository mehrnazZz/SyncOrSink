from __future__ import annotations

from typing import Callable, Dict

from .base import BasePolicy


_POLICY_REGISTRY: Dict[str, Callable[..., BasePolicy]] = {}


def register(name: str):
    def _wrap(cls):
        _POLICY_REGISTRY[name] = cls
        return cls
    return _wrap


def create(name: str, *args, **kwargs) -> BasePolicy:
    if name not in _POLICY_REGISTRY:
        raise ValueError(f"Unknown policy: {name}")
    return _POLICY_REGISTRY[name](*args, **kwargs)


def list_policies():
    return sorted(_POLICY_REGISTRY.keys())
