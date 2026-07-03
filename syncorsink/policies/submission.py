from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping

from .base import BasePolicy


PolicyFn = Callable[[dict, dict, dict], Dict[int, dict]]


@dataclass(frozen=True)
class LoadedPolicy:
    policy: PolicyFn
    metadata: Dict[str, Any]


class ExternalPolicyAdapter(BasePolicy):
    """Adapter for external policy submissions.

    A submitted policy can be either:
    - a callable with signature `(obs, info, state) -> actions`
    - an object with `act(obs, info, state) -> actions`

    Optional `reset`, `metadata`, and `load_checkpoint` methods are forwarded
    when present.
    """

    def __init__(self, policy: Any):
        self.policy = policy

    def reset(self, *args, **kwargs):
        reset = getattr(self.policy, "reset", None)
        if reset is None:
            return None
        return _call_optional_lifecycle(reset, *args, **kwargs)

    def load_checkpoint(self, path: str):
        loader = getattr(self.policy, "load_checkpoint", None)
        if loader is None:
            return None
        return loader(path)

    def metadata(self) -> Dict[str, Any]:
        metadata = getattr(self.policy, "metadata", None)
        if metadata is None:
            return {}
        data = metadata() if callable(metadata) else metadata
        if data is None:
            return {}
        if not isinstance(data, Mapping):
            raise ValueError("external policy metadata must be a mapping")
        return dict(data)

    def act(self, obs: dict, info: dict, state: dict) -> Dict[int, dict]:
        act = getattr(self.policy, "act", None)
        if act is not None:
            return act(obs, info, state)
        if callable(self.policy):
            return self.policy(obs, info, state)
        raise TypeError("external policy must be callable or expose act(obs, info, state)")


def load_policy_entrypoint(
    entrypoint: str,
    *,
    env: Any,
    spec: Mapping[str, Any],
    checkpoint: str | None = None,
    kwargs: Mapping[str, Any] | None = None,
) -> LoadedPolicy:
    target = import_entrypoint(entrypoint)
    policy, used_kwargs = _call_policy_factory(
        target,
        {
            "env": env,
            "spec": dict(spec),
            "checkpoint": checkpoint,
            **dict(kwargs or {}),
        },
    )
    if policy is None:
        raise ValueError(f"policy entrypoint returned None: {entrypoint}")

    adapter = ExternalPolicyAdapter(policy)
    if checkpoint is not None and "checkpoint" not in used_kwargs:
        adapter.load_checkpoint(checkpoint)
    return LoadedPolicy(policy=adapter, metadata=adapter.metadata())


def import_entrypoint(entrypoint: str) -> Any:
    if ":" not in entrypoint:
        raise ValueError("policy entrypoint must be formatted as 'module.submodule:object'")
    module_name, object_name = entrypoint.split(":", 1)
    if not module_name or not object_name:
        raise ValueError("policy entrypoint must include both module and object")
    module = importlib.import_module(module_name)
    target = module
    for part in object_name.split("."):
        target = getattr(target, part)
    return target


def reset_policy(policy: Any, *, episode: int | None = None, seed: int | None = None) -> None:
    reset = getattr(policy, "reset", None)
    if reset is None:
        return None
    _call_optional_lifecycle(reset, episode=episode, seed=seed)
    return None


def _call_policy_factory(factory: Any, available_kwargs: Mapping[str, Any]) -> tuple[Any, set[str]]:
    if not callable(factory):
        raise TypeError("policy entrypoint object must be callable")
    kwargs, used = _select_supported_kwargs(factory, available_kwargs)
    try:
        return factory(**kwargs), used
    except TypeError as exc:
        if kwargs:
            try:
                return factory(), set()
            except TypeError:
                pass
        raise TypeError(f"failed to construct external policy from entrypoint: {exc}") from exc


def _select_supported_kwargs(factory: Any, available_kwargs: Mapping[str, Any]) -> tuple[dict[str, Any], set[str]]:
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return {}, set()

    accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
    kwargs: dict[str, Any] = {}
    for name, value in available_kwargs.items():
        if value is None:
            continue
        if accepts_kwargs or name in signature.parameters:
            kwargs[name] = value
    return kwargs, set(kwargs)


def _call_optional_lifecycle(fn: Callable[..., Any], *args, **kwargs):
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn()

    accepts_args = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in signature.parameters.values())
    accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
    selected_kwargs = {
        key: value
        for key, value in kwargs.items()
        if value is not None and (accepts_kwargs or key in signature.parameters)
    }
    if accepts_args:
        return fn(*args, **selected_kwargs)
    return fn(**selected_kwargs)
