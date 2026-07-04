from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, Dict, Mapping

from .base import BasePolicy


PolicyFn = Callable[[dict, dict, dict], Dict[int, dict]]


@dataclass(frozen=True)
class LoadedPolicy:
    policy: PolicyFn
    metadata: Dict[str, Any]


class ExternalPolicyAdapter(BasePolicy):
    """Adapter for centralized/debug external policy submissions.

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


class DecentralizedEnvView:
    """Read-only environment view for decentralized external policy factories."""

    _ACTION_ATTRS = (
        "ACTION_UP",
        "ACTION_DOWN",
        "ACTION_LEFT",
        "ACTION_RIGHT",
        "ACTION_STAY",
        "ACTION_INTERACT",
        "ACTION_PICKUP",
        "ACTION_DROP",
    )

    def __init__(self, env: Any):
        self.num_agents = int(getattr(env, "num_agents"))
        self.map_size = int(getattr(env, "map_size"))
        self.action_space = getattr(env, "action_space", None)
        self.observation_space = getattr(env, "observation_space", None)
        self.config = _safe_config_view(getattr(env, "config", None))
        for attr in self._ACTION_ATTRS:
            setattr(self, attr, int(getattr(env, attr)))


class DecentralizedExternalPolicyAdapter(ExternalPolicyAdapter):
    """Adapter that calls an external policy once per agent with local context only."""

    def __init__(self, policy: Any, num_agents: int):
        super().__init__(policy)
        self.num_agents = int(num_agents)
        self._reset_agent_states()

    def reset(self, *args, **kwargs):
        self._reset_agent_states()
        return super().reset(*args, **kwargs)

    def _reset_agent_states(self):
        self._agent_states = {agent_id: {"agent_id": agent_id} for agent_id in range(self.num_agents)}

    def metadata(self) -> Dict[str, Any]:
        data = super().metadata()
        data.setdefault("execution", "decentralized")
        return data

    def act(self, obs: dict, info: dict, state: dict) -> Dict[int, dict]:
        agent_callable = getattr(self.policy, "act_agent", None)
        if agent_callable is None:
            agent_callable = getattr(self.policy, "act", None)
            if agent_callable is not None and _looks_like_team_act(agent_callable):
                agent_callable = None
        if agent_callable is None and callable(self.policy) and not _looks_like_team_act(self.policy):
            agent_callable = self.policy
        if agent_callable is None:
            raise TypeError(
                "decentralized external policies must expose act_agent(agent_id, obs, info, state) "
                "or be an agent-level callable"
            )

        actions = {}
        for agent_id, agent_obs in obs.items():
            aid = int(agent_id)
            agent_state = self._agent_states.setdefault(aid, {"agent_id": aid})
            agent_state.update(_agent_state_view(state, aid))
            agent_action = _call_agent_policy(
                agent_callable,
                agent_id=aid,
                obs=agent_obs,
                info=_agent_info_view(info, aid),
                state=agent_state,
            )
            actions[aid] = _unwrap_agent_action(agent_action, aid)
        return actions


def load_policy_entrypoint(
    entrypoint: str,
    *,
    env: Any,
    spec: Mapping[str, Any],
    checkpoint: str | None = None,
    kwargs: Mapping[str, Any] | None = None,
    decentralized: bool = False,
) -> LoadedPolicy:
    target = import_entrypoint(entrypoint)
    factory_env = DecentralizedEnvView(env) if decentralized else env
    policy, used_kwargs = _call_policy_factory(
        target,
        {
            "env": factory_env,
            "spec": dict(spec),
            "checkpoint": checkpoint,
            **dict(kwargs or {}),
        },
    )
    if policy is None:
        raise ValueError(f"policy entrypoint returned None: {entrypoint}")

    adapter = (
        DecentralizedExternalPolicyAdapter(policy, num_agents=getattr(env, "num_agents"))
        if decentralized
        else ExternalPolicyAdapter(policy)
    )
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


def _safe_config_view(config: Any) -> Any:
    if config is None:
        return SimpleNamespace()
    safe = {}
    for key in (
        "scenario",
        "map_size",
        "num_agents",
        "fov_preset",
        "comm_mode",
        "comm_token_limit",
        "max_messages",
        "token_vocab_size",
        "max_steps",
        "comm_radius",
        "comm_cost",
        "comm_len_cost",
        "energy_preset",
        "energy_private_monitor",
        "split",
        "map_variant",
        "track",
    ):
        if hasattr(config, key):
            safe[key] = getattr(config, key)
    return SimpleNamespace(**safe)


def _agent_info_view(info: Mapping[str, Any] | None, agent_id: int) -> dict:
    if not info:
        return {}
    out: dict[str, Any] = {}
    for key, value in info.items():
        if key == "central_obs":
            continue
        if key in {"messages_text", "messages_with_sender", "events", "comm_tokens"} and isinstance(value, Mapping):
            out[key] = {agent_id: value.get(agent_id, [] if key != "comm_tokens" else 0)}
        else:
            out[key] = value
    return out


def _agent_state_view(state: Mapping[str, Any] | None, agent_id: int) -> dict:
    out = {"agent_id": int(agent_id)}
    if state:
        for key, value in state.items():
            if key.startswith("_"):
                continue
            out[key] = value
    return out


def _looks_like_team_act(fn: Callable[..., Any]) -> bool:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    params = [
        p
        for p in signature.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    names = [p.name for p in params]
    return names[:3] == ["obs", "info", "state"] or (len(params) == 3 and "agent_id" not in names)


def _call_agent_policy(fn: Callable[..., Any], *, agent_id: int, obs: dict, info: dict, state: dict) -> Any:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(agent_id, obs, info, state)

    accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
    if accepts_kwargs:
        return fn(agent_id=agent_id, obs=obs, info=info, state=state)
    params = [
        p
        for p in signature.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    names = [p.name for p in params]
    if "agent_id" in names:
        kwargs = {}
        for name, value in {"agent_id": agent_id, "obs": obs, "info": info, "state": state}.items():
            if name in names:
                kwargs[name] = value
        if len(kwargs) == len(params):
            return fn(**kwargs)
        return fn(agent_id, obs, info, state)
    if len(params) >= 4:
        return fn(agent_id, obs, info, state)
    return fn(obs, info, state)


def _unwrap_agent_action(action: Any, agent_id: int) -> dict:
    if not isinstance(action, Mapping):
        raise TypeError("agent-level policy must return an action mapping")
    if "action" in action:
        return dict(action)
    if agent_id in action:
        nested = action[agent_id]
    else:
        nested = action.get(str(agent_id))
    if isinstance(nested, Mapping) and "action" in nested:
        return dict(nested)
    raise TypeError("agent-level policy returned neither an action nor an agent-indexed action")
