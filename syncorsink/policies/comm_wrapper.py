from __future__ import annotations

from typing import Callable, Dict, Any


def _clip_tokens(tokens: list[int], vocab_size: int, limit: int) -> list[int]:
    clipped = [max(0, min(vocab_size - 1, int(t))) for t in tokens]
    return clipped[:limit]


def _oracle_message(env) -> list[int]:
    scenario = env.config.scenario
    if scenario == "pipeline_assembly":
        stages = env.scenario_state.data.get("stages", [])
        open_stages = [s for s in stages if not s["done"]]
        stage = open_stages[0] if open_stages else None
        if stage is None:
            return [1, 0]
        sx, sy = stage["station"]
        required = stage["required"]
        tokens = [1, stage["stage"], sx, sy, len(required)]
        tokens.extend(required)
        return tokens
    if scenario == "energy_grid":
        node_energy = env.scenario_state.data.get("node_energy", {})
        node_types = env.scenario_state.data.get("node_types", {})
        if not node_energy:
            return [2, 0]
        target_node = sorted(node_energy.items(), key=lambda kv: kv[1])[0][0]
        tx, ty = target_node
        ttype = node_types.get(target_node, 0)
        recharge = env.scenario_state.data.get("recharge_count", 0)
        target = env.scenario_state.data.get("success_recharges", 0)
        return [2, tx, ty, ttype, recharge, target]
    if scenario == "signal_hunt":
        target = env.scenario_state.data.get("target")
        if target is None:
            return [3, 0]
        tx, ty = target
        return [3, tx, ty]
    return [0]


def wrap_oracle_with_comm(policy_fn: Callable[[dict, dict, dict], Dict[int, dict]], env):
    last_tokens: list[int] | None = None

    def _policy(obs: dict, info: dict, state: dict) -> Dict[int, dict]:
        nonlocal last_tokens
        actions = policy_fn(obs, info, state)
        tokens = _oracle_message(env)
        tokens = _clip_tokens(tokens, env.token_vocab_size, env.comm_token_limit)
        send = tokens != (last_tokens or [])
        if send:
            last_tokens = tokens
        for aid in range(env.num_agents):
            act = actions.get(aid, {"action": env.ACTION_STAY})
            act["message_tokens"] = tokens if send else []
            actions[aid] = act
        return actions

    return _policy
