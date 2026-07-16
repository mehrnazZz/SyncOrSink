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
    env_config: SyncOrSinkConfig | None = None


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
    include_signal_trace: bool = False,
) -> dict[str, Any]:
    if episodes < 1:
        raise ValueError("episodes must be >= 1")
    if not policies:
        raise ValueError("at least one policy is required")

    policy_results = []
    for spec in policies:
        policy_results.append(
            audit_policy(
                spec.env_config or env_config,
                spec,
                episodes=episodes,
                seed=seed,
                include_signal_trace=include_signal_trace,
            )
        )

    return {
        "status": "complete",
        "config": {
            "env": asdict(env_config),
            "episodes": episodes,
            "seed": seed,
            "include_signal_trace": include_signal_trace,
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
    include_signal_trace: bool = False,
) -> dict[str, Any]:
    from syncorsink.train.seed import set_global_seeds

    set_global_seeds(seed)
    env = SyncOrSinkEnv(env_config)
    policy = policy_spec.factory(env)
    episode_rows: list[dict[str, Any]] = []
    stats: list[EpisodeStats] = []

    for ep in range(episodes):
        ep_seed = seed + ep
        row, ep_stats = _run_single_episode(
            env,
            policy,
            ep,
            ep_seed,
            include_signal_trace=include_signal_trace,
        )
        episode_rows.append(row)
        stats.append(ep_stats)

    summary = summarize(stats)
    return {
        "label": policy_spec.label,
        "metadata": dict(policy_spec.metadata or {}),
        "env_config": asdict(env_config),
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
    eval_signal_scan_gate_threshold: float | None = None,
    eval_signal_scan_gate_suppress: bool | None = None,
    eval_signal_target_validity_threshold: float | None = None,
    eval_signal_target_decision_threshold: float | None = None,
    eval_signal_target_decision_suppress: bool | None = None,
    eval_signal_scan_sync_assist: bool | None = None,
    eval_signal_scan_sync_force_first: bool | None = None,
    eval_signal_scan_broadcast_assist: bool | None = None,
    eval_signal_exact_target_message_guard: bool | None = None,
    eval_signal_exact_target_navigation_assist: bool | None = None,
    eval_signal_exact_target_memory_steps: int | None = None,
    eval_signal_scan_refresh_assist: bool | None = None,
    eval_signal_scan_refresh_threshold: float | None = None,
) -> PolicyFactory:
    def _factory(env: SyncOrSinkEnv) -> PolicyFn:
        del env
        from syncorsink.train.recurrent_bc_rl import load_recurrent_checkpoint_policy

        return load_recurrent_checkpoint_policy(
            Path(checkpoint),
            device=device,
            eval_send_threshold=eval_send_threshold,
            eval_signal_scan_gate_threshold=eval_signal_scan_gate_threshold,
            eval_signal_scan_gate_suppress=eval_signal_scan_gate_suppress,
            eval_signal_target_validity_threshold=eval_signal_target_validity_threshold,
            eval_signal_target_decision_threshold=eval_signal_target_decision_threshold,
            eval_signal_target_decision_suppress=eval_signal_target_decision_suppress,
            eval_signal_scan_sync_assist=eval_signal_scan_sync_assist,
            eval_signal_scan_sync_force_first=eval_signal_scan_sync_force_first,
            eval_signal_scan_broadcast_assist=eval_signal_scan_broadcast_assist,
            eval_signal_exact_target_message_guard=eval_signal_exact_target_message_guard,
            eval_signal_exact_target_navigation_assist=eval_signal_exact_target_navigation_assist,
            eval_signal_exact_target_memory_steps=eval_signal_exact_target_memory_steps,
            eval_signal_scan_refresh_assist=eval_signal_scan_refresh_assist,
            eval_signal_scan_refresh_threshold=eval_signal_scan_refresh_threshold,
        )

    return _factory


def recurrent_checkpoint_env_config(
    checkpoint: str | Path,
    base_config: SyncOrSinkConfig | None = None,
) -> SyncOrSinkConfig:
    """Build the environment surface expected by a recurrent checkpoint."""
    import torch

    ckpt = torch.load(Path(checkpoint), map_location="cpu")
    raw_cfg = ckpt.get("config", {}) if isinstance(ckpt, Mapping) else {}
    if not isinstance(raw_cfg, Mapping):
        raw_cfg = {}
    env_kwargs = asdict(base_config or SyncOrSinkConfig())
    field_map = {
        "scenario": "scenario",
        "map_size": "map_size",
        "agents": "num_agents",
        "fov_preset": "fov_preset",
        "max_steps": "max_steps",
        "energy_preset": "energy_preset",
        "energy_private_monitor": "energy_private_monitor",
        "signal_decoy_count": "signal_decoy_count",
        "decoy_penalty": "decoy_penalty",
        "scan_window": "scan_window",
        "pipeline_shaping": "pipeline_shaping",
        "pipeline_shaping_scale": "pipeline_shaping_scale",
        "energy_shaping": "energy_shaping",
        "energy_shaping_scale": "energy_shaping_scale",
        "signal_shaping": "signal_shaping",
        "signal_shaping_scale": "signal_shaping_scale",
        "signal_scan_bonus": "signal_scan_bonus",
        "signal_joint_scan_bonus": "signal_joint_scan_bonus",
        "signal_colocation_bonus": "signal_colocation_bonus",
        "signal_colocation_radius": "signal_colocation_radius",
        "signal_comm_utility": "signal_comm_utility",
        "signal_target_visit_bonus": "signal_target_visit_bonus",
        "signal_decoy_visit_penalty": "signal_decoy_visit_penalty",
        "signal_unique_target_scan_bonus": "signal_unique_target_scan_bonus",
        "comm_token_limit": "comm_token_limit",
        "comm_vocab_size": "token_vocab_size",
        "comm_max_messages": "max_messages",
        "comm_cost": "comm_cost",
        "comm_len_cost": "comm_len_cost",
        "obs_exploration_memory": "obs_exploration_memory",
        "obs_exploration_age": "obs_exploration_age",
    }
    base_preserved = {"map_size", "max_steps"} if base_config is not None else set()
    for cfg_key, env_key in field_map.items():
        if env_key in base_preserved:
            continue
        if cfg_key in raw_cfg and raw_cfg[cfg_key] is not None:
            env_kwargs[env_key] = raw_cfg[cfg_key]
    return SyncOrSinkConfig(**env_kwargs)


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
    *,
    include_signal_trace: bool = False,
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
    signal = _new_signal_episode_state(env, include_trace=include_signal_trace)
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
    step = int(signal.get("step", 0))
    if target is not None:
        signal.setdefault("trace", []).append(_signal_trace_pre_step(env, actions, step))
    for agent_id, action in actions.items():
        action_id = _action_id(action)
        action_counts[str(action_id)] += 1
        message_tokens = _message_tokens(action)
        if message_tokens:
            signal["message_steps"].add(step)
            signal["message_tokens"] += len(message_tokens)
        if target is not None and action_id == env.ACTION_INTERACT:
            if env.agent_positions[int(agent_id)] == target:
                signal["target_scans"].append((step, int(agent_id)))
    signal["step"] = step + 1


def _record_events(
    info: Mapping[str, Any],
    event_counts: Counter[str],
    signal: dict[str, Any],
) -> None:
    if signal.get("target") is not None and signal.get("trace"):
        signal["trace"][-1]["events"] = _signal_events_by_agent(info)
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
    if signal.get("trace"):
        signal["trace"][-1].update(_signal_trace_post_step(env))
    distances = [manhattan(pos, target) for pos in env.agent_positions]
    signal["min_target_distances"].append(min(distances))
    signal["avg_target_distances"].append(sum(distances) / len(distances))
    radius = int(getattr(env.config, "signal_colocation_radius", 2))
    if sum(1 for dist in distances if dist <= radius) >= 2:
        signal["both_near_target_steps"] += 1


def _new_signal_episode_state(env: SyncOrSinkEnv, *, include_trace: bool = False) -> dict[str, Any]:
    target = None
    if getattr(env.config, "scenario", None) == "signal_hunt":
        target = env.scenario_state.data.get("target")
    return {
        "target": target,
        "include_trace": include_trace,
        "step": 0,
        "clues_found": 0,
        "decoy_scans": 0,
        "target_scans": [],
        "trace": [],
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
    trace = list(signal.get("trace", []))
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
    result = {
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
        "lifecycle": _signal_lifecycle_from_trace(env, trace),
    }
    if bool(signal.get("include_trace", False)):
        result["trace"] = trace
    return result


def _signal_trace_pre_step(
    env: SyncOrSinkEnv,
    actions: Mapping[int, Any],
    step: int,
) -> dict[str, Any]:
    positions = _positions_by_agent(env)
    target = _coerce_pos(env.scenario_state.data.get("target"))
    decoys = _signal_decoys(env)
    return {
        "step": int(step),
        "env_step_before": int(getattr(env, "steps", 0)),
        "positions_before": positions,
        "target_distance_before": _target_distances(positions, target),
        "on_target_before": _agents_at_position(positions, target),
        "on_decoy_before": _agents_at_any_position(positions, decoys),
        "scan_log_before": _signal_scan_log(env),
        "actions": {
            int(agent_id): _trace_action(
                env,
                actions.get(agent_id, actions.get(str(agent_id), {"action": env.ACTION_STAY})),
            )
            for agent_id in range(env.num_agents)
        },
    }


def _signal_trace_post_step(env: SyncOrSinkEnv) -> dict[str, Any]:
    positions = _positions_by_agent(env)
    target = _coerce_pos(env.scenario_state.data.get("target"))
    decoys = _signal_decoys(env)
    return {
        "env_step_after": int(getattr(env, "steps", 0)),
        "positions_after": positions,
        "target_distance_after": _target_distances(positions, target),
        "on_target_after": _agents_at_position(positions, target),
        "on_decoy_after": _agents_at_any_position(positions, decoys),
        "scan_log_after": _signal_scan_log(env),
    }


def _signal_lifecycle_from_trace(env: SyncOrSinkEnv, trace: list[Mapping[str, Any]]) -> dict[str, Any]:
    scan_window = int(env.scenario_state.data.get("scan_window", getattr(env.config, "scan_window", 3)))
    target = _coerce_pos(env.scenario_state.data.get("target"))

    first_target_reach_step: int | None = None
    first_target_reach_agent: int | None = None
    first_target_scan_step: int | None = None
    first_target_scan_agent: int | None = None
    first_joint_target_scan_step: int | None = None
    first_joint_target_scan_agents: list[int] = []
    first_teammate_move_step: int | None = None
    first_teammate_reach_step: int | None = None
    first_teammate_scan_step: int | None = None

    target_scan_events = 0
    joint_target_scan_events = 0
    decoy_scan_events = 0
    redundant_active_target_scans = 0
    refresh_target_scans = 0
    target_reach_without_scan_agent_steps = 0
    message_steps_before_first_scan = 0
    message_steps_after_first_scan = 0
    message_steps_at_first_scan = 0

    for row in trace:
        step = int(row.get("step", 0))
        on_target_before = _int_list(row.get("on_target_before", []))
        on_target_after = _int_list(row.get("on_target_after", []))
        for agent_id in sorted(on_target_before + on_target_after):
            if first_target_reach_step is None:
                first_target_reach_step = step
                first_target_reach_agent = int(agent_id)
                break

        events = _trace_events_by_agent(row)
        target_scanners_this_step = [
            agent_id
            for agent_id, names in events.items()
            if "target_scan" in names
        ]
        joint_scanners_this_step = [
            agent_id
            for agent_id, names in events.items()
            if "joint_target_scan" in names
        ]
        for names in events.values():
            target_scan_events += names.count("target_scan")
            joint_target_scan_events += names.count("joint_target_scan")
            decoy_scan_events += names.count("decoy_scan")

        for agent_id in on_target_before:
            if int(agent_id) not in target_scanners_this_step:
                target_reach_without_scan_agent_steps += 1

        scan_log_before = row.get("scan_log_before", {})
        env_step_after = _maybe_int(row.get("env_step_after"))
        for agent_id in target_scanners_this_step:
            if first_target_scan_step is None:
                first_target_scan_step = step
                first_target_scan_agent = int(agent_id)
            last_scan = _mapping_get(scan_log_before, int(agent_id))
            if last_scan is None or env_step_after is None:
                continue
            age = env_step_after - int(last_scan)
            if 0 <= age < scan_window:
                redundant_active_target_scans += 1
            elif age >= scan_window:
                refresh_target_scans += 1

        if joint_scanners_this_step and first_joint_target_scan_step is None:
            first_joint_target_scan_step = step
            first_joint_target_scan_agents = sorted(int(aid) for aid in joint_scanners_this_step)

        if first_target_scan_step is not None and first_target_scan_agent is not None:
            teammate_ids = [aid for aid in range(env.num_agents) if int(aid) != int(first_target_scan_agent)]
            if step > first_target_scan_step:
                for agent_id in teammate_ids:
                    before = _mapping_get(row.get("target_distance_before", {}), int(agent_id))
                    after = _mapping_get(row.get("target_distance_after", {}), int(agent_id))
                    if (
                        first_teammate_move_step is None
                        and before is not None
                        and after is not None
                        and float(after) < float(before)
                    ):
                        first_teammate_move_step = step
                    if first_teammate_reach_step is None and int(agent_id) in on_target_after:
                        first_teammate_reach_step = step
            if first_teammate_scan_step is None:
                for agent_id in teammate_ids:
                    if int(agent_id) in target_scanners_this_step and step >= first_target_scan_step:
                        first_teammate_scan_step = step
                        break

    for row in trace:
        step = int(row.get("step", 0))
        if not _trace_has_message(row):
            continue
        if first_target_scan_step is None or step < first_target_scan_step:
            message_steps_before_first_scan += 1
        elif step == first_target_scan_step:
            message_steps_at_first_scan += 1
        else:
            message_steps_after_first_scan += 1

    diagnoses: list[str] = []
    if first_target_reach_step is None:
        diagnoses.append("never_reached_target")
    elif first_target_scan_step is None:
        diagnoses.append("reached_target_without_scan")
    elif first_joint_target_scan_step is None:
        if first_teammate_move_step is None and first_teammate_scan_step is None:
            diagnoses.append("no_teammate_response_after_first_scan")
        elif first_teammate_scan_step is None:
            diagnoses.append("teammate_responded_but_no_target_scan")
        else:
            diagnoses.append("unsynchronized_scan_window_miss")
    else:
        diagnoses.append("joint_scan_completed")

    if redundant_active_target_scans > 0:
        diagnoses.append("redundant_active_rescans")
    if refresh_target_scans > 0:
        diagnoses.append("refresh_scans")
    if decoy_scan_events > 0:
        diagnoses.append("decoy_scans")
    if not diagnoses:
        diagnoses.append("no_issue_detected")

    return {
        "target": list(target) if target is not None else None,
        "scan_window": scan_window,
        "first_target_reach_step": first_target_reach_step,
        "first_target_reach_agent": first_target_reach_agent,
        "first_target_scan_step": first_target_scan_step,
        "first_target_scan_agent": first_target_scan_agent,
        "first_joint_target_scan_step": first_joint_target_scan_step,
        "first_joint_target_scan_agents": first_joint_target_scan_agents,
        "first_teammate_move_toward_target_step_after_first_scan": first_teammate_move_step,
        "first_teammate_target_reach_step_after_first_scan": first_teammate_reach_step,
        "first_teammate_target_scan_step_after_first_scan": first_teammate_scan_step,
        "steps_reach_to_first_scan": _step_delta(first_target_reach_step, first_target_scan_step),
        "steps_first_scan_to_joint": _step_delta(first_target_scan_step, first_joint_target_scan_step),
        "target_scan_events": target_scan_events,
        "joint_target_scan_events": joint_target_scan_events,
        "decoy_scan_events": decoy_scan_events,
        "redundant_active_target_scans": redundant_active_target_scans,
        "refresh_target_scans": refresh_target_scans,
        "target_reach_without_scan_agent_steps": target_reach_without_scan_agent_steps,
        "message_steps_before_first_scan": message_steps_before_first_scan,
        "message_steps_at_first_scan": message_steps_at_first_scan,
        "message_steps_after_first_scan": message_steps_after_first_scan,
        "diagnoses": diagnoses,
    }


def _positions_by_agent(env: SyncOrSinkEnv) -> dict[int, list[int]]:
    return {
        int(agent_id): [int(pos[0]), int(pos[1])]
        for agent_id, pos in enumerate(env.agent_positions)
    }


def _signal_decoys(env: SyncOrSinkEnv) -> list[tuple[int, int]]:
    return [
        pos
        for raw_pos in env.scenario_state.data.get("decoys", [])
        if (pos := _coerce_pos(raw_pos)) is not None
    ]


def _signal_scan_log(env: SyncOrSinkEnv) -> dict[int, int]:
    raw_log = env.scenario_state.data.get("scan_log") or {}
    scan_log: dict[int, int] = {}
    if not isinstance(raw_log, Mapping):
        return scan_log
    for agent_id, step in raw_log.items():
        try:
            scan_log[int(agent_id)] = int(step)
        except (TypeError, ValueError):
            continue
    return dict(sorted(scan_log.items()))


def _target_distances(
    positions: Mapping[int, list[int]],
    target: tuple[int, int] | None,
) -> dict[int, int]:
    if target is None:
        return {}
    return {
        int(agent_id): int(manhattan((int(pos[0]), int(pos[1])), target))
        for agent_id, pos in positions.items()
    }


def _agents_at_position(
    positions: Mapping[int, list[int]],
    target: tuple[int, int] | None,
) -> list[int]:
    if target is None:
        return []
    return sorted(
        int(agent_id)
        for agent_id, pos in positions.items()
        if (int(pos[0]), int(pos[1])) == target
    )


def _agents_at_any_position(
    positions: Mapping[int, list[int]],
    targets: list[tuple[int, int]],
) -> list[int]:
    target_set = set(targets)
    if not target_set:
        return []
    return sorted(
        int(agent_id)
        for agent_id, pos in positions.items()
        if (int(pos[0]), int(pos[1])) in target_set
    )


def _trace_action(env: SyncOrSinkEnv, action: Any) -> dict[str, Any]:
    action_id = _action_id(action)
    message_tokens = _message_tokens(action)
    trace = {
        "action": action_id,
        "name": _action_name(env, action_id),
        "message_len": len(message_tokens),
        "message_tokens": message_tokens,
    }
    if isinstance(action, Mapping) and action.get("message_text"):
        trace["message_text"] = str(action.get("message_text"))
    return trace


def _action_name(env: SyncOrSinkEnv, action_id: int) -> str:
    names = {
        int(env.ACTION_UP): "up",
        int(env.ACTION_DOWN): "down",
        int(env.ACTION_LEFT): "left",
        int(env.ACTION_RIGHT): "right",
        int(env.ACTION_STAY): "stay",
        int(env.ACTION_INTERACT): "interact",
        int(env.ACTION_PICKUP): "pickup",
        int(env.ACTION_DROP): "drop",
    }
    return names.get(int(action_id), f"unknown_{int(action_id)}")


def _signal_events_by_agent(info: Mapping[str, Any]) -> dict[int, list[str]]:
    raw_events = info.get("events", {})
    events_by_agent: dict[int, list[str]] = {}
    if not isinstance(raw_events, Mapping):
        return events_by_agent
    for agent_id, events in raw_events.items():
        if not isinstance(events, list):
            continue
        names = [
            str(event.get("event", "unknown"))
            for event in events
            if isinstance(event, Mapping)
        ]
        if names:
            events_by_agent[int(agent_id)] = names
    return dict(sorted(events_by_agent.items()))


def _trace_events_by_agent(row: Mapping[str, Any]) -> dict[int, list[str]]:
    raw_events = row.get("events", {})
    events_by_agent: dict[int, list[str]] = {}
    if not isinstance(raw_events, Mapping):
        return events_by_agent
    for agent_id, names in raw_events.items():
        if isinstance(names, list):
            events_by_agent[int(agent_id)] = [str(name) for name in names]
    return events_by_agent


def _trace_has_message(row: Mapping[str, Any]) -> bool:
    actions = row.get("actions", {})
    if not isinstance(actions, Mapping):
        return False
    for action in actions.values():
        if isinstance(action, Mapping) and int(action.get("message_len", 0)) > 0:
            return True
    return False


def _compact_signal_lifecycle(lifecycle: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "first_target_reach_step",
        "first_target_scan_step",
        "first_joint_target_scan_step",
        "steps_reach_to_first_scan",
        "steps_first_scan_to_joint",
        "target_scan_events",
        "redundant_active_target_scans",
        "refresh_target_scans",
        "target_reach_without_scan_agent_steps",
        "diagnoses",
    )
    return {key: lifecycle.get(key) for key in keys if key in lifecycle}


def _coerce_pos(pos: Any) -> tuple[int, int] | None:
    if pos is None:
        return None
    try:
        x, y = pos
    except (TypeError, ValueError):
        return None
    return int(x), int(y)


def _int_list(values: Any) -> list[int]:
    if not isinstance(values, list):
        return []
    return [int(value) for value in values]


def _mapping_get(mapping: Any, key: int, default: Any = None) -> Any:
    if not isinstance(mapping, Mapping):
        return default
    return mapping.get(key, mapping.get(str(key), default))


def _maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _step_delta(start: int | None, end: int | None) -> int | None:
    if start is None or end is None:
        return None
    return int(end) - int(start)


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
        lifecycle_rows = [
            signal["lifecycle"]
            for signal in signal_rows
            if isinstance(signal.get("lifecycle"), Mapping)
        ]
        if lifecycle_rows:
            diagnosis_counts = Counter(
                str(diagnosis)
                for lifecycle in lifecycle_rows
                for diagnosis in lifecycle.get("diagnoses", [])
            )
            diagnostics["signal_lifecycle"] = {
                "avg_first_target_reach_step": _avg_key(lifecycle_rows, "first_target_reach_step"),
                "avg_first_target_scan_step": _avg_key(lifecycle_rows, "first_target_scan_step"),
                "avg_first_joint_target_scan_step": _avg_key(lifecycle_rows, "first_joint_target_scan_step"),
                "avg_steps_reach_to_first_scan": _avg_key(lifecycle_rows, "steps_reach_to_first_scan"),
                "avg_steps_first_scan_to_joint": _avg_key(lifecycle_rows, "steps_first_scan_to_joint"),
                "avg_target_scan_events": _avg_key(lifecycle_rows, "target_scan_events"),
                "avg_joint_target_scan_events": _avg_key(lifecycle_rows, "joint_target_scan_events"),
                "avg_redundant_active_target_scans": _avg_key(lifecycle_rows, "redundant_active_target_scans"),
                "avg_refresh_target_scans": _avg_key(lifecycle_rows, "refresh_target_scans"),
                "avg_target_reach_without_scan_agent_steps": _avg_key(
                    lifecycle_rows,
                    "target_reach_without_scan_agent_steps",
                ),
                "avg_message_steps_before_first_scan": _avg_key(lifecycle_rows, "message_steps_before_first_scan"),
                "avg_message_steps_at_first_scan": _avg_key(lifecycle_rows, "message_steps_at_first_scan"),
                "avg_message_steps_after_first_scan": _avg_key(lifecycle_rows, "message_steps_after_first_scan"),
                "episodes_with_joint_target_scan": sum(
                    1
                    for lifecycle in lifecycle_rows
                    if lifecycle.get("first_joint_target_scan_step") is not None
                ),
                "diagnosis_counts": dict(sorted(diagnosis_counts.items())),
            }
    return diagnostics


def _compare_by_seed(policy_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_seed: dict[int, dict[str, Any]] = {}
    for result in policy_results:
        label = result["label"]
        for row in result["episodes"]:
            seed = int(row["seed"])
            by_seed.setdefault(seed, {"seed": seed})
            policy_row = {
                "success": bool(row["success"]),
                "failure_type": row.get("failure_type"),
                "steps": int(row["steps"]),
                "comm_tokens": int(row["comm_tokens"]),
            }
            signal = row.get("signal") or {}
            lifecycle = signal.get("lifecycle") if isinstance(signal, Mapping) else None
            if isinstance(lifecycle, Mapping):
                policy_row["signal_lifecycle"] = _compact_signal_lifecycle(lifecycle)
            by_seed[seed][label] = policy_row
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
