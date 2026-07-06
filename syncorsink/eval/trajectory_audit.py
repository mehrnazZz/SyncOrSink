"""Trajectory-level diagnostics for SyncOrSink policies.

The benchmark score tells us whether a policy solved the task. This module
adds the next layer down: did the policy collect clues, scan decoys, synchronize
near the target, send messages, or simply time out?
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from syncorsink.envs import SyncOrSinkConfig, SyncOrSinkEnv
from syncorsink.envs.utils import manhattan
from syncorsink.eval.metrics import EpisodeStats, summarize
from syncorsink.eval.success import episode_success
from syncorsink.policies.submission import reset_policy


PolicyFn = Callable[[dict, dict, dict], dict[int, dict]]
PolicyFactory = Callable[[SyncOrSinkEnv], PolicyFn]


@dataclass(frozen=True)
class AuditPolicySpec:
    label: str
    factory: PolicyFactory
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class MAPPODecodeConfig:
    deterministic: bool = False
    action_mode: str = "sample"
    action_temperature: float = 1.0
    send_mode: str = "threshold"
    send_threshold: float = 0.25
    token_mode: str = "argmax"
    token_temperature: float = 1.0
    length_mode: str = "argmax"
    length_temperature: float = 1.0


def run_trajectory_audit(
    env_config: SyncOrSinkConfig,
    policies: list[AuditPolicySpec] | tuple[AuditPolicySpec, ...],
    *,
    episodes: int = 100,
    seed: int = 3000,
) -> dict[str, Any]:
    if episodes < 1:
        raise ValueError("episodes must be >= 1")
    if not policies:
        raise ValueError("at least one policy is required")

    policy_results = []
    for spec in policies:
        policy_results.append(
            audit_policy(
                env_config,
                spec,
                episodes=episodes,
                seed=seed,
            )
        )

    return {
        "status": "complete",
        "config": {
            "env": asdict(env_config),
            "episodes": episodes,
            "seed": seed,
        },
        "policies": policy_results,
        "comparison": _compare_by_seed(policy_results),
    }


def audit_policy(
    env_config: SyncOrSinkConfig,
    policy_spec: AuditPolicySpec,
    *,
    episodes: int,
    seed: int,
) -> dict[str, Any]:
    from syncorsink.train.seed import set_global_seeds

    set_global_seeds(seed)
    env = SyncOrSinkEnv(env_config)
    policy = policy_spec.factory(env)
    episode_rows: list[dict[str, Any]] = []
    stats: list[EpisodeStats] = []

    for ep in range(episodes):
        ep_seed = seed + ep
        row, ep_stats = _run_single_episode(env, policy, ep, ep_seed)
        episode_rows.append(row)
        stats.append(ep_stats)

    summary = summarize(stats)
    return {
        "label": policy_spec.label,
        "metadata": dict(policy_spec.metadata or {}),
        "summary": asdict(summary),
        "diagnostics": _summarize_episode_rows(episode_rows),
        "episodes": episode_rows,
    }


def make_oracle_policy_factory(scenario: str, oracle_type: str = "oracle_strong_comm") -> PolicyFactory:
    from syncorsink.policies.comm_wrapper import wrap_oracle_with_comm
    from syncorsink.policies.local_oracle import local_signal_policy
    from syncorsink.policies.oracle import (
        energy_oracle,
        energy_oracle_strong,
        pipeline_oracle,
        pipeline_oracle_strong,
        signal_hunt_oracle,
        signal_hunt_oracle_strong,
    )

    oracle_map = {
        "signal_hunt": {
            "oracle": signal_hunt_oracle,
            "oracle_strong": signal_hunt_oracle_strong,
        },
        "energy_grid": {
            "oracle": energy_oracle,
            "oracle_strong": energy_oracle_strong,
        },
        "pipeline_assembly": {
            "oracle": pipeline_oracle,
            "oracle_strong": pipeline_oracle_strong,
        },
    }
    base_type = oracle_type.removesuffix("_comm")
    if scenario not in oracle_map or base_type not in oracle_map[scenario]:
        if scenario == "signal_hunt" and oracle_type == "signal_hint_comm":
            return lambda env: local_signal_policy(env)
        raise ValueError(f"unknown oracle policy for scenario={scenario!r}: {oracle_type!r}")

    def _factory(env: SyncOrSinkEnv) -> PolicyFn:
        policy = oracle_map[scenario][base_type](env)
        if oracle_type.endswith("_comm"):
            policy = wrap_oracle_with_comm(policy, env)
        return policy

    return _factory


def make_bc_checkpoint_policy_factory(
    checkpoint: str | Path,
    *,
    deterministic: bool = True,
    device: str = "cpu",
) -> PolicyFactory:
    def _factory(env: SyncOrSinkEnv) -> PolicyFn:
        del env
        from syncorsink.train.curriculum import _load_bc_policy

        return _load_bc_policy(Path(checkpoint), deterministic=deterministic, device=device)

    return _factory


def make_mappo_checkpoint_policy_factory(
    checkpoint: str | Path,
    *,
    decode: MAPPODecodeConfig | None = None,
    device: str = "cpu",
    sample_seed: int = 0,
) -> PolicyFactory:
    decode = decode or MAPPODecodeConfig()

    def _factory(env: SyncOrSinkEnv) -> PolicyFn:
        from syncorsink.train.mappo import load_mappo_checkpoint_policy

        return load_mappo_checkpoint_policy(
            Path(checkpoint),
            env,
            deterministic=decode.deterministic,
            device=device,
            sample_seed=sample_seed,
            send_threshold=decode.send_threshold,
            action_mode=decode.action_mode,
            action_temperature=decode.action_temperature,
            send_mode=decode.send_mode,
            token_mode=decode.token_mode,
            token_temperature=decode.token_temperature,
            length_mode=decode.length_mode,
            length_temperature=decode.length_temperature,
        )

    return _factory


def make_recurrent_checkpoint_policy_factory(
    checkpoint: str | Path,
    *,
    device: str = "cpu",
    eval_send_threshold: float = 0.25,
) -> PolicyFactory:
    def _factory(env: SyncOrSinkEnv) -> PolicyFn:
        del env
        from syncorsink.train.recurrent_bc_rl import load_recurrent_checkpoint_policy

        return load_recurrent_checkpoint_policy(
            Path(checkpoint),
            device=device,
            eval_send_threshold=eval_send_threshold,
        )

    return _factory


def signal_failure_type(row: Mapping[str, Any]) -> str:
    if row.get("success"):
        return "success"
    signal = row.get("signal") or {}
    if int(signal.get("decoy_scans", 0)) > 0:
        return "decoy_scan"
    if int(signal.get("clues_found", 0)) == 0 and int(signal.get("target_scans", 0)) == 0:
        return "no_clue_or_target_scan"
    if int(signal.get("target_scans", 0)) == 0:
        return "no_target_scan"
    if int(signal.get("unique_target_scanners", 0)) < 2:
        return "solo_target_scan"
    return "unsynchronized_target_scan"


def generic_failure_type(row: Mapping[str, Any]) -> str:
    if row.get("success"):
        return "success"
    if row.get("truncated"):
        return "timeout"
    events = row.get("event_counts") or {}
    if int(events.get("node_depleted", 0)) > 0:
        return "node_depleted"
    return "terminated_failure"


def _run_single_episode(
    env: SyncOrSinkEnv,
    policy: PolicyFn,
    episode: int,
    seed: int,
) -> tuple[dict[str, Any], EpisodeStats]:
    obs, info = env.reset(seed=seed)
    reset_policy(policy, episode=episode, seed=seed)
    done = False
    truncated = False
    steps = 0
    total_reward = 0.0
    comm_tokens = 0
    per_agent_reward = {i: 0.0 for i in range(env.num_agents)}
    per_agent_comm = {i: 0 for i in range(env.num_agents)}
    event_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    signal = _new_signal_episode_state(env)
    last_info: dict[str, Any] = {}

    while not (done or truncated):
        actions = policy(obs, info, {"step": steps})
        _record_actions(env, actions, action_counts, signal)
        obs, rewards, done, truncated, info = env.step(actions)
        last_info = info or {}
        _record_events(last_info, event_counts, signal)
        _record_post_step(env, signal)

        steps += 1
        total_reward += sum(rewards.values())
        for aid, reward in rewards.items():
            per_agent_reward[aid] += reward
        if "comm_tokens" in last_info:
            comm_tokens += sum(last_info["comm_tokens"].values())
            for aid, count in last_info["comm_tokens"].items():
                per_agent_comm[aid] += count

    scenario = getattr(env.config, "scenario", None)
    success = episode_success(scenario, done, last_info)
    ep_stats = EpisodeStats(
        total_reward=total_reward,
        steps=steps,
        success=success,
        comm_tokens=comm_tokens,
        per_agent_reward=per_agent_reward,
        per_agent_comm=per_agent_comm,
    )
    row = {
        "episode": episode,
        "seed": seed,
        "success": success,
        "done": done,
        "truncated": truncated,
        "steps": steps,
        "total_reward": total_reward,
        "comm_tokens": comm_tokens,
        "per_agent_reward": per_agent_reward,
        "per_agent_comm": per_agent_comm,
        "action_counts": dict(sorted(action_counts.items())),
        "event_counts": dict(sorted(event_counts.items())),
    }
    if scenario == "signal_hunt":
        row["signal"] = _finalize_signal_episode_state(env, signal)
        row["failure_type"] = signal_failure_type(row)
    else:
        row["failure_type"] = generic_failure_type(row)
    return row, ep_stats


def _record_actions(
    env: SyncOrSinkEnv,
    actions: Mapping[int, Any],
    action_counts: Counter[str],
    signal: dict[str, Any],
) -> None:
    target = signal.get("target")
    for agent_id, action in actions.items():
        action_id = _action_id(action)
        action_counts[str(action_id)] += 1
        message_tokens = _message_tokens(action)
        if message_tokens:
            signal["message_steps"].add(signal["step"])
            signal["message_tokens"] += len(message_tokens)
        if target is not None and action_id == env.ACTION_INTERACT:
            if env.agent_positions[int(agent_id)] == target:
                signal["target_scans"].append((signal["step"], int(agent_id)))
    signal["step"] += 1


def _record_events(
    info: Mapping[str, Any],
    event_counts: Counter[str],
    signal: dict[str, Any],
) -> None:
    for event in _iter_events(info):
        name = str(event.get("event", "unknown"))
        event_counts[name] += 1
        if name == "clue_found":
            signal["clues_found"] += 1
        elif name == "decoy_scan":
            signal["decoy_scans"] += 1


def _record_post_step(env: SyncOrSinkEnv, signal: dict[str, Any]) -> None:
    target = signal.get("target")
    if target is None:
        return
    distances = [manhattan(pos, target) for pos in env.agent_positions]
    signal["min_target_distances"].append(min(distances))
    signal["avg_target_distances"].append(sum(distances) / len(distances))
    radius = int(getattr(env.config, "signal_colocation_radius", 2))
    if sum(1 for dist in distances if dist <= radius) >= 2:
        signal["both_near_target_steps"] += 1


def _new_signal_episode_state(env: SyncOrSinkEnv) -> dict[str, Any]:
    target = None
    if getattr(env.config, "scenario", None) == "signal_hunt":
        target = env.scenario_state.data.get("target")
    return {
        "target": target,
        "step": 0,
        "clues_found": 0,
        "decoy_scans": 0,
        "target_scans": [],
        "message_steps": set(),
        "message_tokens": 0,
        "min_target_distances": [],
        "avg_target_distances": [],
        "both_near_target_steps": 0,
    }


def _finalize_signal_episode_state(env: SyncOrSinkEnv, signal: Mapping[str, Any]) -> dict[str, Any]:
    target_scans = list(signal.get("target_scans", []))
    unique_scanners = sorted({agent_id for _, agent_id in target_scans})
    scan_steps = [step for step, _ in target_scans]
    min_gap = None
    if len(scan_steps) >= 2:
        ordered = sorted(scan_steps)
        min_gap = min(b - a for a, b in zip(ordered, ordered[1:]))
    final_distances = {}
    target = signal.get("target")
    if target is not None:
        final_distances = {
            int(agent_id): manhattan(pos, target)
            for agent_id, pos in enumerate(env.agent_positions)
        }
    return {
        "target": list(target) if target is not None else None,
        "clues_found": int(signal.get("clues_found", 0)),
        "decoy_scans": int(signal.get("decoy_scans", 0)),
        "target_scans": len(target_scans),
        "unique_target_scanners": len(unique_scanners),
        "target_scan_steps": scan_steps,
        "target_scan_agents": unique_scanners,
        "min_target_scan_gap": min_gap,
        "message_steps": len(signal.get("message_steps", set())),
        "message_tokens": int(signal.get("message_tokens", 0)),
        "both_near_target_steps": int(signal.get("both_near_target_steps", 0)),
        "final_target_distance": final_distances,
        "min_target_distance": _safe_min(signal.get("min_target_distances", [])),
        "avg_target_distance": _safe_avg(signal.get("avg_target_distances", [])),
    }


def _summarize_episode_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    failure_counts = Counter(str(row.get("failure_type", "unknown")) for row in rows)
    event_counts = Counter()
    action_counts = Counter()
    for row in rows:
        event_counts.update(row.get("event_counts", {}))
        action_counts.update(row.get("action_counts", {}))

    signal_rows = [row["signal"] for row in rows if "signal" in row]
    diagnostics = {
        "failure_type_counts": dict(sorted(failure_counts.items())),
        "event_counts": dict(sorted(event_counts.items())),
        "action_counts": dict(sorted(action_counts.items())),
    }
    if signal_rows:
        diagnostics["signal"] = {
            "avg_clues_found": _avg_key(signal_rows, "clues_found"),
            "avg_decoy_scans": _avg_key(signal_rows, "decoy_scans"),
            "avg_target_scans": _avg_key(signal_rows, "target_scans"),
            "avg_unique_target_scanners": _avg_key(signal_rows, "unique_target_scanners"),
            "avg_message_steps": _avg_key(signal_rows, "message_steps"),
            "avg_message_tokens": _avg_key(signal_rows, "message_tokens"),
            "avg_both_near_target_steps": _avg_key(signal_rows, "both_near_target_steps"),
            "avg_min_target_distance": _avg_key(signal_rows, "min_target_distance"),
            "avg_target_distance": _avg_key(signal_rows, "avg_target_distance"),
        }
    return diagnostics


def _compare_by_seed(policy_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_seed: dict[int, dict[str, Any]] = {}
    for result in policy_results:
        label = result["label"]
        for row in result["episodes"]:
            seed = int(row["seed"])
            by_seed.setdefault(seed, {"seed": seed})
            by_seed[seed][label] = {
                "success": bool(row["success"]),
                "failure_type": row.get("failure_type"),
                "steps": int(row["steps"]),
                "comm_tokens": int(row["comm_tokens"]),
            }
    return [by_seed[seed] for seed in sorted(by_seed)]


def _iter_events(info: Mapping[str, Any]):
    events = info.get("events", {})
    if isinstance(events, Mapping):
        for agent_events in events.values():
            if isinstance(agent_events, list):
                for event in agent_events:
                    if isinstance(event, Mapping):
                        yield event


def _action_id(action: Any) -> int:
    if isinstance(action, Mapping):
        return int(action.get("action", 0))
    return int(action)


def _message_tokens(action: Any) -> list[int]:
    if not isinstance(action, Mapping):
        return []
    tokens = action.get("message_tokens") or []
    return list(tokens)


def _safe_min(values) -> float | None:
    values = [float(value) for value in values if value is not None]
    return min(values) if values else None


def _safe_avg(values) -> float | None:
    values = [float(value) for value in values if value is not None]
    return sum(values) / len(values) if values else None


def _avg_key(rows: list[Mapping[str, Any]], key: str) -> float:
    values = [row.get(key) for row in rows if row.get(key) is not None]
    if not values:
        return 0.0
    return float(sum(values) / len(values))
