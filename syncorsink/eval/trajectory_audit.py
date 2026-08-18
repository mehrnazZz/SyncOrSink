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
    include_pipeline_assist_trace: bool = False,
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
                include_pipeline_assist_trace=include_pipeline_assist_trace,
            )
        )

    return {
        "status": "complete",
        "config": {
            "env": asdict(env_config),
            "policy_envs": [
                {
                    "label": result["label"],
                    "env": result["env_config"],
                }
                for result in policy_results
            ],
            "episodes": episodes,
            "seed": seed,
            "include_signal_trace": include_signal_trace,
            "include_pipeline_assist_trace": include_pipeline_assist_trace,
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
    include_pipeline_assist_trace: bool = False,
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
            include_pipeline_assist_trace=include_pipeline_assist_trace,
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
    from syncorsink.policies.planner_comm import (
        energy_planner_comm,
        pipeline_planner_comm,
        signal_hunt_planner_comm,
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
    planner_map = {
        "signal_hunt": signal_hunt_planner_comm,
        "energy_grid": energy_planner_comm,
        "pipeline_assembly": pipeline_planner_comm,
    }
    planner_aliases = {
        "signal_hunt": "signal_hunt_planner_comm",
        "energy_grid": "energy_planner_comm",
        "pipeline_assembly": "pipeline_planner_comm",
    }
    if oracle_type == "planner_comm":
        return lambda env: planner_map[scenario](env)
    if oracle_type in planner_aliases.values():
        expected = planner_aliases.get(scenario)
        if oracle_type != expected:
            raise ValueError(
                f"{oracle_type} is not valid for scenario={scenario!r}; "
                f"use {expected!r} or 'planner_comm'"
            )
        return lambda env: planner_map[scenario](env)
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
    eval_send_threshold: float | None = None,
    eval_signal_target_scan_threshold: float | None = None,
    eval_signal_scan_gate_threshold: float | None = None,
    eval_signal_scan_gate_suppress: bool | None = None,
    eval_signal_target_validity_threshold: float | None = None,
    eval_signal_target_decision_threshold: float | None = None,
    eval_signal_target_decision_suppress: bool | None = None,
    eval_signal_target_scan_lock: bool | None = None,
    eval_signal_exact_target_scan_lock: bool | None = None,
    eval_signal_compatible_target_scan_assist: bool | None = None,
    eval_signal_compatible_target_scan_min_strength: int | None = None,
    eval_signal_negative_memory_scan_guard: bool | None = None,
    eval_signal_target_probe_assist: bool | None = None,
    eval_signal_scan_sync_assist: bool | None = None,
    eval_signal_scan_sync_force_first: bool | None = None,
    eval_signal_scan_broadcast_assist: bool | None = None,
    eval_signal_constraint_message_copy_assist: bool | None = None,
    eval_signal_constraint_message_guard: bool | None = None,
    eval_signal_exact_target_message_guard: bool | None = None,
    eval_signal_initial_exact_message_copy_assist: bool | None = None,
    eval_signal_exact_target_message_copy_assist: bool | None = None,
    eval_signal_exact_target_navigation_assist: bool | None = None,
    eval_signal_exact_target_memory_steps: int | None = None,
    eval_signal_scan_refresh_assist: bool | None = None,
    eval_signal_scan_refresh_threshold: float | None = None,
    eval_signal_evidence_sweep_assist: bool | None = None,
    eval_signal_frontier_exploration_assist: bool | None = None,
    eval_pipeline_navigation_assist: bool | None = None,
    eval_pipeline_navigation_assist_trust_messages: bool | None = None,
    eval_pipeline_station_interact_guard: bool | None = None,
    eval_pipeline_plan_broadcast_assist: bool | None = None,
    eval_pipeline_pickup_gate_suppress: bool | None = None,
    eval_pipeline_frontier_exploration_assist: bool | None = None,
    eval_pipeline_interact_gate_threshold: float | None = None,
    eval_pipeline_interact_gate_promote: bool | None = None,
    eval_pipeline_event_head_threshold: float | None = None,
    eval_pipeline_navigation_head_threshold: float | None = None,
    eval_pipeline_plan_head_threshold: float | None = None,
    eval_pipeline_option_threshold: float | None = None,
    eval_pipeline_option_allow_interact: bool | None = None,
) -> PolicyFactory:
    def _factory(env: SyncOrSinkEnv) -> PolicyFn:
        del env
        from syncorsink.train.recurrent_bc_rl import load_recurrent_checkpoint_policy

        return load_recurrent_checkpoint_policy(
            Path(checkpoint),
            device=device,
            eval_send_threshold=eval_send_threshold,
            eval_signal_target_scan_threshold=eval_signal_target_scan_threshold,
            eval_signal_scan_gate_threshold=eval_signal_scan_gate_threshold,
            eval_signal_scan_gate_suppress=eval_signal_scan_gate_suppress,
            eval_signal_target_validity_threshold=eval_signal_target_validity_threshold,
            eval_signal_target_decision_threshold=eval_signal_target_decision_threshold,
            eval_signal_target_decision_suppress=eval_signal_target_decision_suppress,
            eval_signal_target_scan_lock=eval_signal_target_scan_lock,
            eval_signal_exact_target_scan_lock=eval_signal_exact_target_scan_lock,
            eval_signal_compatible_target_scan_assist=eval_signal_compatible_target_scan_assist,
            eval_signal_compatible_target_scan_min_strength=(
                eval_signal_compatible_target_scan_min_strength
            ),
            eval_signal_negative_memory_scan_guard=eval_signal_negative_memory_scan_guard,
            eval_signal_target_probe_assist=eval_signal_target_probe_assist,
            eval_signal_scan_sync_assist=eval_signal_scan_sync_assist,
            eval_signal_scan_sync_force_first=eval_signal_scan_sync_force_first,
            eval_signal_scan_broadcast_assist=eval_signal_scan_broadcast_assist,
            eval_signal_constraint_message_copy_assist=eval_signal_constraint_message_copy_assist,
            eval_signal_constraint_message_guard=eval_signal_constraint_message_guard,
            eval_signal_exact_target_message_guard=eval_signal_exact_target_message_guard,
            eval_signal_initial_exact_message_copy_assist=(
                eval_signal_initial_exact_message_copy_assist
            ),
            eval_signal_exact_target_message_copy_assist=(
                eval_signal_exact_target_message_copy_assist
            ),
            eval_signal_exact_target_navigation_assist=eval_signal_exact_target_navigation_assist,
            eval_signal_exact_target_memory_steps=eval_signal_exact_target_memory_steps,
            eval_signal_scan_refresh_assist=eval_signal_scan_refresh_assist,
            eval_signal_scan_refresh_threshold=eval_signal_scan_refresh_threshold,
            eval_signal_evidence_sweep_assist=eval_signal_evidence_sweep_assist,
            eval_signal_frontier_exploration_assist=eval_signal_frontier_exploration_assist,
            eval_pipeline_navigation_assist=eval_pipeline_navigation_assist,
            eval_pipeline_navigation_assist_trust_messages=eval_pipeline_navigation_assist_trust_messages,
            eval_pipeline_station_interact_guard=eval_pipeline_station_interact_guard,
            eval_pipeline_plan_broadcast_assist=eval_pipeline_plan_broadcast_assist,
            eval_pipeline_pickup_gate_suppress=eval_pipeline_pickup_gate_suppress,
            eval_pipeline_frontier_exploration_assist=eval_pipeline_frontier_exploration_assist,
            eval_pipeline_interact_gate_threshold=eval_pipeline_interact_gate_threshold,
            eval_pipeline_interact_gate_promote=eval_pipeline_interact_gate_promote,
            eval_pipeline_event_head_threshold=eval_pipeline_event_head_threshold,
            eval_pipeline_navigation_head_threshold=(
                eval_pipeline_navigation_head_threshold
            ),
            eval_pipeline_plan_head_threshold=eval_pipeline_plan_head_threshold,
            eval_pipeline_option_threshold=eval_pipeline_option_threshold,
            eval_pipeline_option_allow_interact=eval_pipeline_option_allow_interact,
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
        "pipeline_stage_count": "pipeline_stage_count",
        "pipeline_required_per_stage_min": "pipeline_required_per_stage_min",
        "pipeline_required_per_stage_max": "pipeline_required_per_stage_max",
        "pipeline_sync_probability": "pipeline_sync_probability",
        "pipeline_dependency_probability": "pipeline_dependency_probability",
        "pipeline_wrong_delivery_penalty": "pipeline_wrong_delivery_penalty",
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


def pipeline_failure_type(row: Mapping[str, Any]) -> str:
    if row.get("success"):
        return "success"
    pipeline = row.get("pipeline") or {}
    events = row.get("event_counts") or {}
    if int(events.get("pipeline_wrong_delivery", 0)) > 0:
        return "wrong_delivery"
    if int(events.get("pipeline_dependency_blocked", 0)) > 0:
        return "dependency_blocked"
    if int(pipeline.get("missed_sync_interacts", 0)) > 0 or int(pipeline.get("sync_ready_stages", 0)) > 0:
        return "sync_wait"
    if int(pipeline.get("max_delivered_resources", 0)) == 0:
        if int(pipeline.get("missed_pickup_opportunities", 0)) > 0:
            return "missed_pickup"
        return "no_delivery"
    if int(pipeline.get("missed_delivery_opportunities", 0)) > 0:
        return "missed_delivery"
    if int(pipeline.get("missed_pickup_opportunities", 0)) > 0:
        return "missed_pickup"
    if int(pipeline.get("max_stages_completed", 0)) == 0:
        return "no_stage_completed"
    if int(pipeline.get("stages_completed", 0)) < int(pipeline.get("stages_total", 0)):
        return "partial_pipeline"
    if row.get("truncated"):
        return "timeout"
    return "terminated_failure"


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
    include_pipeline_assist_trace: bool = False,
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
    pipeline = _new_pipeline_episode_state(env, include_assist_trace=include_pipeline_assist_trace)
    pipeline_state = _initial_pipeline_assist_state(env)
    last_info: dict[str, Any] = {}

    while not (done or truncated):
        actions = policy(obs, info, {"step": steps})
        _record_pipeline_assist_corrections(env, obs, actions, pipeline, pipeline_state)
        _record_actions(env, actions, action_counts, signal, pipeline)
        obs, rewards, done, truncated, info = env.step(actions)
        last_info = info or {}
        _record_events(last_info, event_counts, signal, pipeline)
        _record_post_step(env, signal, pipeline)
        pipeline_state = _update_pipeline_assist_state(env, pipeline_state, last_info)

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
    elif scenario == "pipeline_assembly":
        row["pipeline"] = _finalize_pipeline_episode_state(env, pipeline)
        row["failure_type"] = pipeline_failure_type(row)
    else:
        row["failure_type"] = generic_failure_type(row)
    return row, ep_stats


def _record_actions(
    env: SyncOrSinkEnv,
    actions: Mapping[int, Any],
    action_counts: Counter[str],
    signal: dict[str, Any],
    pipeline: dict[str, Any],
) -> None:
    target = signal.get("target")
    step = int(signal.get("step", 0))
    if target is not None:
        signal.setdefault("trace", []).append(_signal_trace_pre_step(env, actions, step))
    if pipeline.get("active"):
        _record_pipeline_action_opportunities(env, actions, pipeline)
    for agent_id, action in actions.items():
        action_id = _action_id(action)
        action_counts[str(action_id)] += 1
        message_tokens = _message_tokens(action)
        if message_tokens:
            signal["message_steps"].add(step)
            signal["message_tokens"] += len(message_tokens)
            if pipeline.get("active"):
                pipeline["message_steps"].add(step)
                pipeline["message_tokens"] += len(message_tokens)
        if target is not None and action_id == env.ACTION_INTERACT:
            if env.agent_positions[int(agent_id)] == target:
                signal["target_scans"].append((step, int(agent_id)))
    signal["step"] = step + 1


def _record_events(
    info: Mapping[str, Any],
    event_counts: Counter[str],
    signal: dict[str, Any],
    pipeline: dict[str, Any],
) -> None:
    if signal.get("target") is not None and signal.get("trace"):
        signal["trace"][-1]["events"] = _signal_events_by_agent(info)
    for agent_id, event in _iter_agent_events(info):
        name = str(event.get("event", "unknown"))
        event_counts[name] += 1
        if name == "clue_found":
            signal["clues_found"] += 1
        elif name == "decoy_scan":
            signal["decoy_scans"] += 1
        if pipeline.get("active"):
            pipeline["event_counts"][name] += 1
            _record_pipeline_event(pipeline, agent_id, name, event)


def _record_post_step(env: SyncOrSinkEnv, signal: dict[str, Any], pipeline: dict[str, Any]) -> None:
    target = signal.get("target")
    if target is not None:
        if signal.get("trace"):
            signal["trace"][-1].update(_signal_trace_post_step(env))
        distances = [manhattan(pos, target) for pos in env.agent_positions]
        signal["min_target_distances"].append(min(distances))
        signal["avg_target_distances"].append(sum(distances) / len(distances))
        radius = int(getattr(env.config, "signal_colocation_radius", 2))
        if sum(1 for dist in distances if dist <= radius) >= 2:
            signal["both_near_target_steps"] += 1
    if pipeline.get("active"):
        _record_pipeline_progress(env, pipeline)


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


def _new_pipeline_episode_state(
    env: SyncOrSinkEnv,
    *,
    include_assist_trace: bool = False,
) -> dict[str, Any]:
    active = getattr(env.config, "scenario", None) == "pipeline_assembly"
    return {
        "active": active,
        "include_assist_trace": bool(include_assist_trace),
        "message_steps": set(),
        "message_tokens": 0,
        "event_counts": Counter(),
        "progress": [],
        "pickup_opportunities": 0,
        "missed_pickup_opportunities": 0,
        "delivery_opportunities": 0,
        "missed_delivery_opportunities": 0,
        "sync_interact_opportunities": 0,
        "missed_sync_interacts": 0,
        "drop_actions": 0,
        "stall_near_resource_steps": 0,
        "stall_near_station_steps": 0,
        "max_stages_completed": 0,
        "max_delivered_resources": 0,
        "pickup_status_counts": Counter(),
        "delivery_decision_counts": Counter(),
        "wrong_delivery_provenance_counts": Counter(),
        "wrong_delivery_decision_counts": Counter(),
        "agent_carry_source": {},
        "last_delivery_decision_by_agent": {},
        "delivery_ready_station_distances": [],
        "wrong_delivery_ready_station_distances": [],
        "wrong_delivery_events": 0,
        "assist_opportunities": 0,
        "assist_correction_steps": 0,
        "assist_correction_agents": 0,
        "assist_correction_action_counts": Counter(),
        "assist_first_corrections": [],
    }


def _initial_pipeline_assist_state(env: SyncOrSinkEnv) -> dict[str, Any] | None:
    if getattr(env.config, "scenario", None) != "pipeline_assembly":
        return None
    try:
        from syncorsink.train.recurrent_bc_rl import RecurrentConfig, _initial_pipeline_state
    except Exception:
        return None
    return dict(_initial_pipeline_state(_pipeline_recurrent_assist_config(env, RecurrentConfig)))


def _update_pipeline_assist_state(
    env: SyncOrSinkEnv,
    pipeline_state: dict[str, Any] | None,
    info: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if getattr(env.config, "scenario", None) != "pipeline_assembly":
        return pipeline_state
    try:
        from syncorsink.train.recurrent_bc_rl import RecurrentConfig, _update_pipeline_state_from_info
    except Exception:
        return pipeline_state
    cfg = _pipeline_recurrent_assist_config(env, RecurrentConfig)
    updated = _update_pipeline_state_from_info(
        cfg,
        pipeline_state,
        dict(info or {}),
        env.num_agents,
    )
    return dict(updated)


def _pipeline_recurrent_assist_config(env: SyncOrSinkEnv, recurrent_config_cls):
    cfg = env.config
    return recurrent_config_cls(
        scenario="pipeline_assembly",
        map_size=int(getattr(cfg, "map_size", env.map_size)),
        agents=int(env.num_agents),
        fov_preset=str(getattr(cfg, "fov_preset", "easy")),
        max_steps=int(getattr(cfg, "max_steps", 300)),
        comm=bool(getattr(cfg, "comm_token_limit", 0) > 0),
        comm_token_limit=int(getattr(cfg, "comm_token_limit", 8)),
        comm_vocab_size=int(getattr(cfg, "token_vocab_size", 32)),
        comm_max_messages=int(getattr(cfg, "max_messages", 8)),
        pipeline_stage_count=getattr(cfg, "pipeline_stage_count", None),
        pipeline_required_per_stage_min=int(getattr(cfg, "pipeline_required_per_stage_min", 1)),
        pipeline_required_per_stage_max=int(getattr(cfg, "pipeline_required_per_stage_max", 2)),
        pipeline_sync_probability=float(getattr(cfg, "pipeline_sync_probability", 0.5)),
        pipeline_dependency_probability=float(getattr(cfg, "pipeline_dependency_probability", 0.7)),
        pipeline_wrong_delivery_penalty=float(getattr(cfg, "pipeline_wrong_delivery_penalty", 0.25)),
        eval_pipeline_navigation_assist=True,
        eval_pipeline_navigation_assist_trust_messages=True,
        eval_pipeline_station_interact_guard=True,
    )


def _record_pipeline_assist_corrections(
    env: SyncOrSinkEnv,
    obs: Mapping[int, Any],
    actions: Mapping[int, Any],
    pipeline: dict[str, Any],
    pipeline_state: Mapping[str, Any] | None,
) -> None:
    if not pipeline.get("active"):
        return
    try:
        import torch
        from syncorsink.train.recurrent_bc_rl import (
            RecurrentConfig,
            _apply_pipeline_navigation_assist,
            _apply_pipeline_station_interact_guard,
        )
    except Exception:
        return

    cfg = _pipeline_recurrent_assist_config(env, RecurrentConfig)
    action_ids = [
        _action_id(actions.get(aid, actions.get(str(aid), {"action": env.ACTION_STAY})))
        for aid in range(env.num_agents)
    ]
    acts = torch.tensor(action_ids, dtype=torch.long)
    corrected = _apply_pipeline_station_interact_guard(
        cfg,
        dict(obs),
        acts,
        pipeline_state=pipeline_state,
    )
    corrected = _apply_pipeline_navigation_assist(
        cfg,
        dict(obs),
        corrected,
        pipeline_state=pipeline_state,
    )
    corrected_ids = [int(value) for value in corrected.detach().cpu().tolist()]
    pipeline["assist_opportunities"] += int(env.num_agents)
    changed_agents = [
        aid
        for aid, (before, after) in enumerate(zip(action_ids, corrected_ids))
        if int(before) != int(after)
    ]
    if not changed_agents:
        return
    pipeline["assist_correction_steps"] += 1
    pipeline["assist_correction_agents"] += len(changed_agents)
    counts = pipeline.setdefault("assist_correction_action_counts", Counter())
    samples = pipeline.setdefault("assist_first_corrections", [])
    for aid in changed_agents:
        before = int(action_ids[aid])
        after = int(corrected_ids[aid])
        kind = _pipeline_assist_correction_kind(env, before, after)
        key = f"{_action_name(env, before)}->{_action_name(env, after)}"
        counts[key] += 1
        if bool(pipeline.get("include_assist_trace", False)) and len(samples) < 16:
            samples.append(_pipeline_assist_correction_sample(env, aid, before, after, kind))


def _pipeline_assist_correction_kind(env: SyncOrSinkEnv, before: int, after: int) -> str:
    if int(after) == int(env.ACTION_PICKUP):
        return "missed_pickup"
    if int(before) == int(env.ACTION_PICKUP):
        return "bad_pickup"
    if int(after) == int(env.ACTION_INTERACT):
        return "missed_interact"
    if int(before) == int(env.ACTION_INTERACT):
        return "bad_interact"
    if int(after) == int(env.ACTION_DROP):
        return "inventory_recovery"
    move_actions = {
        int(env.ACTION_UP),
        int(env.ACTION_DOWN),
        int(env.ACTION_LEFT),
        int(env.ACTION_RIGHT),
    }
    if int(after) in move_actions:
        return "navigation"
    return "other"


def _pipeline_assist_correction_sample(
    env: SyncOrSinkEnv,
    agent_id: int,
    before: int,
    after: int,
    kind: str,
) -> dict[str, Any]:
    stage = _pipeline_target_stage(env)
    station = _coerce_pos(stage.get("station")) if isinstance(stage, Mapping) else None
    required = _pipeline_stage_needs(stage) if isinstance(stage, Mapping) else []
    pos = tuple(env.agent_positions[int(agent_id)])
    resource_type = int((env.scenario_state.data.get("resource_types") or {}).get(pos, 0))
    return {
        "step": int(getattr(env, "steps", 0)),
        "agent": int(agent_id),
        "kind": kind,
        "from_action": int(before),
        "from_name": _action_name(env, before),
        "to_action": int(after),
        "to_name": _action_name(env, after),
        "position": [int(pos[0]), int(pos[1])],
        "inventory": int(env.inventories[int(agent_id)]),
        "resource_type": resource_type,
        "stage": int(stage.get("stage", -1)) if isinstance(stage, Mapping) else None,
        "station": list(station) if station is not None else None,
        "remaining_required": [int(value) for value in required],
    }


def _record_pipeline_action_opportunities(
    env: SyncOrSinkEnv,
    actions: Mapping[int, Any],
    pipeline: dict[str, Any],
) -> None:
    stage = _pipeline_target_stage(env)
    if stage is None:
        return
    station = _coerce_pos(stage.get("station"))
    if station is None:
        return
    needs = _pipeline_stage_needs(stage)
    resources = env.scenario_state.data.get("resource_types", {})
    for agent_id in range(env.num_agents):
        action = actions.get(agent_id, actions.get(str(agent_id), {"action": env.ACTION_STAY}))
        action_id = _action_id(action)
        pos = tuple(env.agent_positions[int(agent_id)])
        inv = int(env.inventories[int(agent_id)])
        if action_id == env.ACTION_DROP:
            pipeline["drop_actions"] += 1
        if action_id == env.ACTION_PICKUP:
            _record_pipeline_pickup_decision(env, int(agent_id), pos, inv, pipeline)
        if action_id == env.ACTION_INTERACT:
            _record_pipeline_delivery_decision(env, int(agent_id), pos, inv, pipeline)
        if inv == 0 and int(resources.get(pos, 0)) in needs:
            pipeline["pickup_opportunities"] += 1
            if action_id != env.ACTION_PICKUP:
                pipeline["missed_pickup_opportunities"] += 1
            if action_id == env.ACTION_STAY:
                pipeline["stall_near_resource_steps"] += 1
        if inv in needs and pos == station:
            pipeline["delivery_opportunities"] += 1
            if action_id != env.ACTION_INTERACT:
                pipeline["missed_delivery_opportunities"] += 1
        if inv in needs and manhattan(pos, station) <= 1 and action_id == env.ACTION_STAY:
            pipeline["stall_near_station_steps"] += 1
        if not needs and bool(stage.get("sync")) and pos == station:
            pipeline["sync_interact_opportunities"] += 1
            if action_id != env.ACTION_INTERACT:
                pipeline["missed_sync_interacts"] += 1


def _record_pipeline_progress(env: SyncOrSinkEnv, pipeline: dict[str, Any]) -> None:
    summary = _pipeline_progress_summary(env)
    pipeline["progress"].append(summary)
    pipeline["max_stages_completed"] = max(
        int(pipeline.get("max_stages_completed", 0)),
        int(summary.get("stages_completed", 0)),
    )
    pipeline["max_delivered_resources"] = max(
        int(pipeline.get("max_delivered_resources", 0)),
        int(summary.get("delivered_resources", 0)),
    )


def _finalize_pipeline_episode_state(env: SyncOrSinkEnv, pipeline: Mapping[str, Any]) -> dict[str, Any]:
    summary = _pipeline_progress_summary(env)
    required = int(summary.get("required_resources", 0))
    total = int(summary.get("stages_total", 0))
    result = {
        **summary,
        "completion_ratio": (
            float(summary["stages_completed"]) / float(total)
            if total > 0 else 0.0
        ),
        "delivery_ratio": (
            float(summary["delivered_resources"]) / float(required)
            if required > 0 else 0.0
        ),
        "max_stages_completed": int(pipeline.get("max_stages_completed", summary["stages_completed"])),
        "max_delivered_resources": int(pipeline.get("max_delivered_resources", summary["delivered_resources"])),
        "message_steps": len(pipeline.get("message_steps", set())),
        "message_tokens": int(pipeline.get("message_tokens", 0)),
        "pickup_opportunities": int(pipeline.get("pickup_opportunities", 0)),
        "missed_pickup_opportunities": int(pipeline.get("missed_pickup_opportunities", 0)),
        "delivery_opportunities": int(pipeline.get("delivery_opportunities", 0)),
        "missed_delivery_opportunities": int(pipeline.get("missed_delivery_opportunities", 0)),
        "sync_interact_opportunities": int(pipeline.get("sync_interact_opportunities", 0)),
        "missed_sync_interacts": int(pipeline.get("missed_sync_interacts", 0)),
        "drop_actions": int(pipeline.get("drop_actions", 0)),
        "stall_near_resource_steps": int(pipeline.get("stall_near_resource_steps", 0)),
        "stall_near_station_steps": int(pipeline.get("stall_near_station_steps", 0)),
        "event_counts": dict(sorted((pipeline.get("event_counts") or {}).items())),
        "stage_summaries": _pipeline_stage_summaries(env),
        "stage_details": _pipeline_stage_details(env),
        "final_agent_positions": [
            [int(pos[0]), int(pos[1])]
            for pos in getattr(env, "agent_positions", [])
        ],
        "final_agent_inventories": [
            int(inv) for inv in getattr(env, "inventories", [])
        ],
        "final_resource_positions": _pipeline_resource_positions(env),
        "pickup_status_counts": dict(sorted((pipeline.get("pickup_status_counts") or {}).items())),
        "delivery_decision_counts": dict(sorted((pipeline.get("delivery_decision_counts") or {}).items())),
        "wrong_delivery_provenance_counts": dict(sorted(
            (pipeline.get("wrong_delivery_provenance_counts") or {}).items()
        )),
        "wrong_delivery_decision_counts": dict(sorted(
            (pipeline.get("wrong_delivery_decision_counts") or {}).items()
        )),
        "pickup_attempts": sum((pipeline.get("pickup_status_counts") or {}).values()),
        "pickup_needed_ready_attempts": int((pipeline.get("pickup_status_counts") or {}).get("needed_ready", 0)),
        "pickup_needed_blocked_attempts": int((pipeline.get("pickup_status_counts") or {}).get("needed_blocked", 0)),
        "pickup_unneeded_attempts": (
            int((pipeline.get("pickup_status_counts") or {}).get("not_required", 0))
            + int((pipeline.get("pickup_status_counts") or {}).get("already_satisfied", 0))
        ),
        "delivery_interact_attempts": sum((pipeline.get("delivery_decision_counts") or {}).values()),
        "delivery_ready_match_interacts": int(
            (pipeline.get("delivery_decision_counts") or {}).get("ready_station_resource_match", 0)
        ),
        "delivery_blocked_match_interacts": int(
            (pipeline.get("delivery_decision_counts") or {}).get("blocked_station_resource_match", 0)
        ),
        "delivery_wrong_resource_interacts": int(
            (pipeline.get("delivery_decision_counts") or {}).get("wrong_resource_for_station", 0)
        ),
        "delivery_non_station_interacts": int(
            (pipeline.get("delivery_decision_counts") or {}).get("not_station", 0)
        ),
        "wrong_delivery_events": int(pipeline.get("wrong_delivery_events", 0)),
        "wrong_delivery_after_unneeded_pickup": (
            int((pipeline.get("wrong_delivery_provenance_counts") or {}).get("not_required", 0))
            + int((pipeline.get("wrong_delivery_provenance_counts") or {}).get("already_satisfied", 0))
        ),
        "wrong_delivery_after_needed_ready_pickup": int(
            (pipeline.get("wrong_delivery_provenance_counts") or {}).get("needed_ready", 0)
        ),
        "wrong_delivery_after_needed_blocked_pickup": int(
            (pipeline.get("wrong_delivery_provenance_counts") or {}).get("needed_blocked", 0)
        ),
        "wrong_delivery_without_pickup_trace": int(
            (pipeline.get("wrong_delivery_provenance_counts") or {}).get("unknown_pickup", 0)
        ),
        "avg_delivery_ready_station_distance": _safe_avg(
            pipeline.get("delivery_ready_station_distances", [])
        ),
        "avg_wrong_delivery_ready_station_distance": _safe_avg(
            pipeline.get("wrong_delivery_ready_station_distances", [])
        ),
        "assist_opportunities": int(pipeline.get("assist_opportunities", 0)),
        "assist_correction_steps": int(pipeline.get("assist_correction_steps", 0)),
        "assist_correction_agents": int(pipeline.get("assist_correction_agents", 0)),
        "assist_correction_rate": (
            float(pipeline.get("assist_correction_agents", 0))
            / float(max(1, int(pipeline.get("assist_opportunities", 0))))
        ),
        "assist_correction_action_counts": dict(sorted(
            (pipeline.get("assist_correction_action_counts") or {}).items()
        )),
    }
    if bool(pipeline.get("include_assist_trace", False)):
        result["assist_first_corrections"] = list(pipeline.get("assist_first_corrections", []))
    return result


def _record_pipeline_pickup_decision(
    env: SyncOrSinkEnv,
    agent_id: int,
    pos: tuple[int, int],
    inventory: int,
    pipeline: dict[str, Any],
) -> None:
    counts = pipeline.setdefault("pickup_status_counts", Counter())
    if inventory != 0:
        counts["already_carrying"] += 1
        return
    resource_type = int((env.scenario_state.data.get("resource_types") or {}).get(pos, 0))
    if resource_type == 0:
        counts["no_resource"] += 1
        return

    context = _pipeline_resource_need_context(env, resource_type)
    status = str(context["status"])
    counts[status] += 1
    pipeline.setdefault("agent_carry_source", {})[int(agent_id)] = {
        "step": len(pipeline.get("progress", [])),
        "position": [int(pos[0]), int(pos[1])],
        "resource_type": int(resource_type),
        "need_status": status,
        "ready_stage_ids": context["ready_stage_ids"],
        "blocked_stage_ids": context["blocked_stage_ids"],
    }


def _record_pipeline_delivery_decision(
    env: SyncOrSinkEnv,
    agent_id: int,
    pos: tuple[int, int],
    inventory: int,
    pipeline: dict[str, Any],
) -> None:
    if inventory == 0:
        stages = list(env.scenario_state.data.get("stages", []))
        open_station_stages = _pipeline_open_station_stages_at(env, pos)
        ready_sync_stage_ids = [
            int(stage.get("stage", idx))
            for idx, stage in enumerate(open_station_stages)
            if bool(stage.get("sync", False))
            and _pipeline_stage_deps_done(stages, stage)
            and not _pipeline_stage_needs(stage)
        ]
        premature_sync_stage_ids = [
            int(stage.get("stage", idx))
            for idx, stage in enumerate(open_station_stages)
            if bool(stage.get("sync", False))
            and int(stage.get("stage", idx)) not in set(ready_sync_stage_ids)
        ]
        if ready_sync_stage_ids:
            label = "ready_sync_station"
        elif premature_sync_stage_ids:
            label = "premature_sync_station"
        elif open_station_stages:
            label = "empty_inventory_station"
        else:
            return
        context = {
            "label": label,
            "nearest_ready_station_distance": None,
            "open_station_stage_ids": [
                int(stage.get("stage", idx))
                for idx, stage in enumerate(open_station_stages)
            ],
            "ready_sync_stage_ids": ready_sync_stage_ids,
            "premature_sync_stage_ids": premature_sync_stage_ids,
        }
    else:
        context = _pipeline_delivery_decision_context(env, pos, inventory)
        label = str(context["label"])

    pipeline.setdefault("delivery_decision_counts", Counter())[label] += 1
    distance = context.get("nearest_ready_station_distance")
    if distance is not None:
        pipeline.setdefault("delivery_ready_station_distances", []).append(int(distance))
        if label not in {"ready_station_resource_match", "blocked_station_resource_match"}:
            pipeline.setdefault("wrong_delivery_ready_station_distances", []).append(int(distance))
    pipeline.setdefault("last_delivery_decision_by_agent", {})[int(agent_id)] = {
        "step": len(pipeline.get("progress", [])),
        "position": [int(pos[0]), int(pos[1])],
        "resource_type": int(inventory),
        "label": label,
        "nearest_ready_station_distance": distance,
        "open_station_stage_ids": context.get("open_station_stage_ids", []),
        "ready_match_stage_ids": context.get("ready_match_stage_ids", []),
        "blocked_match_stage_ids": context.get("blocked_match_stage_ids", []),
        "ready_sync_stage_ids": context.get("ready_sync_stage_ids", []),
        "premature_sync_stage_ids": context.get("premature_sync_stage_ids", []),
    }


def _record_pipeline_event(
    pipeline: dict[str, Any],
    agent_id: int,
    name: str,
    event: Mapping[str, Any],
) -> None:
    carry_sources = pipeline.setdefault("agent_carry_source", {})
    if name == "picked_resource":
        if int(agent_id) not in carry_sources:
            resource_type = _maybe_int(event.get("resource_type")) or 0
            carry_sources[int(agent_id)] = {
                "step": len(pipeline.get("progress", [])),
                "position": None,
                "resource_type": int(resource_type),
                "need_status": "unknown_pickup",
                "ready_stage_ids": [],
                "blocked_stage_ids": [],
            }
    elif name == "pipeline_wrong_delivery":
        pipeline["wrong_delivery_events"] = int(pipeline.get("wrong_delivery_events", 0)) + 1
        source = carry_sources.get(int(agent_id))
        status = "unknown_pickup"
        if isinstance(source, Mapping):
            status = str(source.get("need_status", "unknown_pickup"))
        pipeline.setdefault("wrong_delivery_provenance_counts", Counter())[status] += 1
        decision = pipeline.setdefault("last_delivery_decision_by_agent", {}).get(int(agent_id), {})
        label = str(decision.get("label", "unknown_decision")) if isinstance(decision, Mapping) else "unknown_decision"
        pipeline.setdefault("wrong_delivery_decision_counts", Counter())[label] += 1
    elif name in {"delivered", "dropped_resource"}:
        carry_sources.pop(int(agent_id), None)


def _pipeline_resource_need_context(env: SyncOrSinkEnv, resource_type: int) -> dict[str, Any]:
    stages = list(env.scenario_state.data.get("stages", []))
    ready_stage_ids: list[int] = []
    blocked_stage_ids: list[int] = []
    satisfied_stage_ids: list[int] = []
    required_anywhere = False

    for stage in stages:
        if bool(stage.get("done")):
            continue
        required = [int(value) for value in stage.get("required", [])]
        req_count = required.count(int(resource_type))
        if req_count <= 0:
            continue
        required_anywhere = True
        delivered_count = [int(value) for value in stage.get("delivered", [])].count(int(resource_type))
        stage_id = int(stage.get("stage", len(ready_stage_ids) + len(blocked_stage_ids)))
        if delivered_count >= req_count:
            satisfied_stage_ids.append(stage_id)
        elif _pipeline_stage_deps_done(stages, stage):
            ready_stage_ids.append(stage_id)
        else:
            blocked_stage_ids.append(stage_id)

    if ready_stage_ids:
        status = "needed_ready"
    elif blocked_stage_ids:
        status = "needed_blocked"
    elif required_anywhere:
        status = "already_satisfied"
    else:
        status = "not_required"
    return {
        "status": status,
        "ready_stage_ids": ready_stage_ids,
        "blocked_stage_ids": blocked_stage_ids,
        "satisfied_stage_ids": satisfied_stage_ids,
    }


def _pipeline_delivery_decision_context(
    env: SyncOrSinkEnv,
    pos: tuple[int, int],
    resource_type: int,
) -> dict[str, Any]:
    stages = list(env.scenario_state.data.get("stages", []))
    open_station_stages = _pipeline_open_station_stages_at(env, pos)
    done_station_stage_ids = [
        int(stage.get("stage", idx))
        for idx, stage in enumerate(stages)
        if bool(stage.get("done")) and _coerce_pos(stage.get("station")) == pos
    ]
    ready_match_stage_ids: list[int] = []
    blocked_match_stage_ids: list[int] = []
    already_satisfied_stage_ids: list[int] = []
    for stage in open_station_stages:
        stage_id = int(stage.get("stage", len(ready_match_stage_ids) + len(blocked_match_stage_ids)))
        required = [int(value) for value in stage.get("required", [])]
        req_count = required.count(int(resource_type))
        delivered_count = [int(value) for value in stage.get("delivered", [])].count(int(resource_type))
        if req_count > 0 and delivered_count < req_count:
            if _pipeline_stage_deps_done(stages, stage):
                ready_match_stage_ids.append(stage_id)
            else:
                blocked_match_stage_ids.append(stage_id)
        elif req_count > 0:
            already_satisfied_stage_ids.append(stage_id)

    ready_stations = [
        station
        for stage in stages
        if not bool(stage.get("done"))
        and _pipeline_stage_deps_done(stages, stage)
        and int(resource_type) in _pipeline_stage_needs(stage)
        if (station := _coerce_pos(stage.get("station"))) is not None
    ]
    nearest_ready_station_distance = (
        min(manhattan(pos, station) for station in ready_stations)
        if ready_stations else None
    )

    if ready_match_stage_ids:
        label = "ready_station_resource_match"
    elif blocked_match_stage_ids:
        label = "blocked_station_resource_match"
    elif already_satisfied_stage_ids:
        label = "already_satisfied_station"
    elif open_station_stages:
        label = "wrong_resource_for_station"
    elif done_station_stage_ids:
        label = "completed_station"
    else:
        label = "not_station"
    return {
        "label": label,
        "nearest_ready_station_distance": nearest_ready_station_distance,
        "open_station_stage_ids": [int(stage.get("stage", idx)) for idx, stage in enumerate(open_station_stages)],
        "done_station_stage_ids": done_station_stage_ids,
        "ready_match_stage_ids": ready_match_stage_ids,
        "blocked_match_stage_ids": blocked_match_stage_ids,
        "already_satisfied_stage_ids": already_satisfied_stage_ids,
    }


def _pipeline_open_station_stages_at(
    env: SyncOrSinkEnv,
    pos: tuple[int, int],
) -> list[Mapping[str, Any]]:
    return [
        stage
        for stage in env.scenario_state.data.get("stages", [])
        if not bool(stage.get("done")) and _coerce_pos(stage.get("station")) == pos
    ]


def _pipeline_stage_deps_done(stages: list[Mapping[str, Any]], stage: Mapping[str, Any]) -> bool:
    for dep in stage.get("deps", []):
        try:
            dep_stage = stages[int(dep)]
        except (IndexError, TypeError, ValueError):
            return False
        if not bool(dep_stage.get("done")):
            return False
    return True


def _pipeline_target_stage(env: SyncOrSinkEnv) -> Mapping[str, Any] | None:
    stages = env.scenario_state.data.get("stages", [])
    open_stages = [stage for stage in stages if not bool(stage.get("done"))]
    if not open_stages:
        return None
    available = [
        stage
        for stage in open_stages
        if all(bool(stages[d].get("done")) for d in stage.get("deps", []))
    ]
    return (available or open_stages)[0]


def _pipeline_stage_needs(stage: Mapping[str, Any]) -> list[int]:
    required = [int(value) for value in stage.get("required", [])]
    delivered = [int(value) for value in stage.get("delivered", [])]
    needs: list[int] = []
    for value in required:
        if value in delivered:
            delivered.remove(value)
        else:
            needs.append(value)
    return needs


def _pipeline_progress_summary(env: SyncOrSinkEnv) -> dict[str, int]:
    stages = env.scenario_state.data.get("stages", [])
    stages_total = len(stages)
    stages_completed = sum(1 for stage in stages if bool(stage.get("done")))
    required_resources = sum(len(stage.get("required", [])) for stage in stages)
    delivered_resources = sum(len(stage.get("delivered", [])) for stage in stages)
    ready_stages = 0
    dependency_blocked_stages = 0
    sync_ready_stages = 0
    for stage in stages:
        if bool(stage.get("done")):
            continue
        deps_done = all(bool(stages[d].get("done")) for d in stage.get("deps", []))
        requirements_met = not _pipeline_stage_needs(stage)
        if deps_done:
            ready_stages += 1
        elif requirements_met:
            dependency_blocked_stages += 1
        if deps_done and requirements_met and bool(stage.get("sync")):
            sync_ready_stages += 1
    return {
        "stages_total": int(stages_total),
        "stages_completed": int(stages_completed),
        "required_resources": int(required_resources),
        "delivered_resources": int(delivered_resources),
        "ready_stages": int(ready_stages),
        "dependency_blocked_stages": int(dependency_blocked_stages),
        "sync_ready_stages": int(sync_ready_stages),
    }


def _pipeline_stage_summaries(env: SyncOrSinkEnv) -> list[dict[str, Any]]:
    rows = []
    for stage in env.scenario_state.data.get("stages", []):
        rows.append({
            "stage": int(stage.get("stage", len(rows))),
            "done": bool(stage.get("done")),
            "required_count": len(stage.get("required", [])),
            "delivered_count": len(stage.get("delivered", [])),
            "dependency_count": len(stage.get("deps", [])),
            "sync": bool(stage.get("sync")),
        })
    return rows


def _pipeline_stage_details(env: SyncOrSinkEnv) -> list[dict[str, Any]]:
    rows = []
    for stage in env.scenario_state.data.get("stages", []):
        station = _coerce_pos(stage.get("station"))
        rows.append({
            "stage": int(stage.get("stage", len(rows))),
            "station": [int(station[0]), int(station[1])] if station is not None else None,
            "required": [int(value) for value in stage.get("required", [])],
            "delivered": [int(value) for value in stage.get("delivered", [])],
            "deps": [int(value) for value in stage.get("deps", [])],
            "done": bool(stage.get("done")),
            "sync": bool(stage.get("sync")),
        })
    return rows


def _pipeline_resource_positions(env: SyncOrSinkEnv) -> list[dict[str, Any]]:
    resources = env.scenario_state.data.get("resource_types", {}) or {}
    rows = []
    for raw_pos, raw_type in resources.items():
        pos = _coerce_pos(raw_pos)
        if pos is None:
            continue
        rows.append({
            "position": [int(pos[0]), int(pos[1])],
            "resource_type": int(raw_type),
        })
    return sorted(rows, key=lambda row: (row["position"][1], row["position"][0], row["resource_type"]))


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
    first_exact_target_message_step: int | None = None
    first_exact_target_message_agents: list[int] = []
    first_teammate_move_after_exact_message_step: int | None = None
    first_teammate_reach_after_exact_message_step: int | None = None
    first_teammate_scan_after_exact_message_step: int | None = None

    target_scan_events = 0
    joint_target_scan_events = 0
    decoy_scan_events = 0
    redundant_active_target_scans = 0
    refresh_target_scans = 0
    target_reach_without_scan_agent_steps = 0
    message_steps_before_first_scan = 0
    message_steps_after_first_scan = 0
    message_steps_at_first_scan = 0
    exact_target_message_steps_before_first_scan = 0
    exact_target_message_steps_after_first_scan = 0
    exact_target_message_steps_at_first_scan = 0

    for row in trace:
        step = int(row.get("step", 0))
        exact_message_agents = _trace_exact_target_message_agents(row, target)
        if exact_message_agents and first_exact_target_message_step is None:
            first_exact_target_message_step = step
            first_exact_target_message_agents = sorted(exact_message_agents)

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

        if first_exact_target_message_step is not None and step > first_exact_target_message_step:
            messenger_ids = set(first_exact_target_message_agents)
            receiver_ids = [aid for aid in range(env.num_agents) if int(aid) not in messenger_ids]
            for agent_id in receiver_ids:
                before = _mapping_get(row.get("target_distance_before", {}), int(agent_id))
                after = _mapping_get(row.get("target_distance_after", {}), int(agent_id))
                if (
                    first_teammate_move_after_exact_message_step is None
                    and before is not None
                    and after is not None
                    and float(after) < float(before)
                ):
                    first_teammate_move_after_exact_message_step = step
                if first_teammate_reach_after_exact_message_step is None and int(agent_id) in on_target_after:
                    first_teammate_reach_after_exact_message_step = step
                if first_teammate_scan_after_exact_message_step is None:
                    for agent_id in receiver_ids:
                        if int(agent_id) in target_scanners_this_step:
                            first_teammate_scan_after_exact_message_step = step
                            break

    for row in trace:
        step = int(row.get("step", 0))
        has_message = _trace_has_message(row)
        has_exact_target_message = bool(_trace_exact_target_message_agents(row, target))
        if has_message:
            if first_target_scan_step is None or step < first_target_scan_step:
                message_steps_before_first_scan += 1
            elif step == first_target_scan_step:
                message_steps_at_first_scan += 1
            else:
                message_steps_after_first_scan += 1
        if has_exact_target_message:
            if first_target_scan_step is None or step < first_target_scan_step:
                exact_target_message_steps_before_first_scan += 1
            elif step == first_target_scan_step:
                exact_target_message_steps_at_first_scan += 1
            else:
                exact_target_message_steps_after_first_scan += 1

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

    rendezvous_diagnoses = _signal_rendezvous_diagnoses(
        success=bool(first_joint_target_scan_step is not None),
        first_exact_target_message_step=first_exact_target_message_step,
        first_target_reach_step=first_target_reach_step,
        first_target_scan_step=first_target_scan_step,
        first_joint_target_scan_step=first_joint_target_scan_step,
        first_teammate_move_after_exact_message_step=(
            first_teammate_move_after_exact_message_step
        ),
        first_teammate_reach_after_exact_message_step=(
            first_teammate_reach_after_exact_message_step
        ),
        first_teammate_scan_after_exact_message_step=(
            first_teammate_scan_after_exact_message_step
        ),
        target_reach_without_scan_agent_steps=target_reach_without_scan_agent_steps,
    )

    return {
        "target": list(target) if target is not None else None,
        "scan_window": scan_window,
        "first_target_reach_step": first_target_reach_step,
        "first_target_reach_agent": first_target_reach_agent,
        "first_target_scan_step": first_target_scan_step,
        "first_target_scan_agent": first_target_scan_agent,
        "first_joint_target_scan_step": first_joint_target_scan_step,
        "first_joint_target_scan_agents": first_joint_target_scan_agents,
        "first_exact_target_message_step": first_exact_target_message_step,
        "first_exact_target_message_agents": first_exact_target_message_agents,
        "first_teammate_move_toward_target_step_after_first_scan": first_teammate_move_step,
        "first_teammate_target_reach_step_after_first_scan": first_teammate_reach_step,
        "first_teammate_target_scan_step_after_first_scan": first_teammate_scan_step,
        "first_teammate_move_toward_target_step_after_exact_message": (
            first_teammate_move_after_exact_message_step
        ),
        "first_teammate_target_reach_step_after_exact_message": (
            first_teammate_reach_after_exact_message_step
        ),
        "first_teammate_target_scan_step_after_exact_message": (
            first_teammate_scan_after_exact_message_step
        ),
        "steps_reach_to_first_scan": _step_delta(first_target_reach_step, first_target_scan_step),
        "steps_first_scan_to_joint": _step_delta(first_target_scan_step, first_joint_target_scan_step),
        "steps_exact_message_to_teammate_move": _step_delta(
            first_exact_target_message_step,
            first_teammate_move_after_exact_message_step,
        ),
        "steps_exact_message_to_teammate_reach": _step_delta(
            first_exact_target_message_step,
            first_teammate_reach_after_exact_message_step,
        ),
        "steps_exact_message_to_teammate_scan": _step_delta(
            first_exact_target_message_step,
            first_teammate_scan_after_exact_message_step,
        ),
        "target_scan_events": target_scan_events,
        "joint_target_scan_events": joint_target_scan_events,
        "decoy_scan_events": decoy_scan_events,
        "redundant_active_target_scans": redundant_active_target_scans,
        "refresh_target_scans": refresh_target_scans,
        "target_reach_without_scan_agent_steps": target_reach_without_scan_agent_steps,
        "message_steps_before_first_scan": message_steps_before_first_scan,
        "message_steps_at_first_scan": message_steps_at_first_scan,
        "message_steps_after_first_scan": message_steps_after_first_scan,
        "exact_target_message_steps_before_first_scan": (
            exact_target_message_steps_before_first_scan
        ),
        "exact_target_message_steps_at_first_scan": exact_target_message_steps_at_first_scan,
        "exact_target_message_steps_after_first_scan": (
            exact_target_message_steps_after_first_scan
        ),
        "diagnoses": diagnoses,
        "rendezvous_diagnoses": rendezvous_diagnoses,
    }


def _signal_rendezvous_diagnoses(
    *,
    success: bool,
    first_exact_target_message_step: int | None,
    first_target_reach_step: int | None,
    first_target_scan_step: int | None,
    first_joint_target_scan_step: int | None,
    first_teammate_move_after_exact_message_step: int | None,
    first_teammate_reach_after_exact_message_step: int | None,
    first_teammate_scan_after_exact_message_step: int | None,
    target_reach_without_scan_agent_steps: int,
) -> list[str]:
    if success:
        return ["success"]
    diagnoses: list[str] = []
    if first_exact_target_message_step is None:
        diagnoses.append("no_exact_target_evidence_message")
    elif first_target_scan_step is not None and first_exact_target_message_step >= first_target_scan_step:
        diagnoses.append("exact_target_message_sent_too_late")

    if first_exact_target_message_step is not None:
        if (
            first_teammate_move_after_exact_message_step is None
            and first_teammate_scan_after_exact_message_step is None
        ):
            diagnoses.append("teammate_received_exact_target_but_did_not_route")
        elif first_teammate_reach_after_exact_message_step is None:
            diagnoses.append("teammate_routed_but_arrived_too_late")
        elif first_teammate_scan_after_exact_message_step is None:
            diagnoses.append("teammate_arrived_but_did_not_scan")
    if (
        first_target_reach_step is not None
        and first_target_scan_step is not None
        and first_joint_target_scan_step is None
        and int(target_reach_without_scan_agent_steps) <= 0
    ):
        diagnoses.append("solo_scanner_coordination_gap")
    if not diagnoses:
        diagnoses.append("unclassified_rendezvous_gap")
    return diagnoses


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


def _trace_exact_target_message_agents(
    row: Mapping[str, Any],
    target: tuple[int, int] | None,
) -> list[int]:
    if target is None:
        return []
    actions = row.get("actions", {})
    if not isinstance(actions, Mapping):
        return []
    target_x, target_y = int(target[0]), int(target[1])
    agents: list[int] = []
    for raw_agent_id, action in actions.items():
        if not isinstance(action, Mapping):
            continue
        tokens = action.get("message_tokens", [])
        if not isinstance(tokens, list):
            continue
        for idx in range(max(0, len(tokens) - 2)):
            try:
                code = int(tokens[idx])
                tx = int(tokens[idx + 1])
                ty = int(tokens[idx + 2])
            except (TypeError, ValueError):
                continue
            if code == 26 and tx == target_x and ty == target_y:
                try:
                    agents.append(int(raw_agent_id))
                except (TypeError, ValueError):
                    pass
                break
    return sorted(set(agents))


def _compact_signal_lifecycle(lifecycle: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "first_target_reach_step",
        "first_target_scan_step",
        "first_joint_target_scan_step",
        "first_exact_target_message_step",
        "steps_reach_to_first_scan",
        "steps_first_scan_to_joint",
        "steps_exact_message_to_teammate_move",
        "steps_exact_message_to_teammate_reach",
        "steps_exact_message_to_teammate_scan",
        "target_scan_events",
        "redundant_active_target_scans",
        "refresh_target_scans",
        "target_reach_without_scan_agent_steps",
        "diagnoses",
        "rendezvous_diagnoses",
    )
    return {key: lifecycle.get(key) for key in keys if key in lifecycle}


def _compact_pipeline_summary(pipeline: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "stages_completed",
        "stages_total",
        "completion_ratio",
        "delivered_resources",
        "required_resources",
        "delivery_ratio",
        "missed_pickup_opportunities",
        "missed_delivery_opportunities",
        "missed_sync_interacts",
        "drop_actions",
        "pickup_status_counts",
        "delivery_decision_counts",
        "wrong_delivery_provenance_counts",
        "wrong_delivery_events",
        "episodes_with_wrong_delivery_events",
        "wrong_delivery_after_unneeded_pickup",
        "avg_wrong_delivery_ready_station_distance",
        "assist_correction_rate",
        "assist_correction_agents",
        "assist_correction_action_counts",
    )
    return {key: pipeline.get(key) for key in keys if key in pipeline}


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
            rendezvous_diagnosis_counts = Counter(
                str(diagnosis)
                for lifecycle in lifecycle_rows
                for diagnosis in lifecycle.get("rendezvous_diagnoses", [])
            )
            diagnostics["signal_lifecycle"] = {
                "avg_first_target_reach_step": _avg_key(lifecycle_rows, "first_target_reach_step"),
                "avg_first_target_scan_step": _avg_key(lifecycle_rows, "first_target_scan_step"),
                "avg_first_joint_target_scan_step": _avg_key(lifecycle_rows, "first_joint_target_scan_step"),
                "avg_first_exact_target_message_step": _avg_key(
                    lifecycle_rows,
                    "first_exact_target_message_step",
                ),
                "avg_steps_reach_to_first_scan": _avg_key(lifecycle_rows, "steps_reach_to_first_scan"),
                "avg_steps_first_scan_to_joint": _avg_key(lifecycle_rows, "steps_first_scan_to_joint"),
                "avg_steps_exact_message_to_teammate_move": _avg_key(
                    lifecycle_rows,
                    "steps_exact_message_to_teammate_move",
                ),
                "avg_steps_exact_message_to_teammate_reach": _avg_key(
                    lifecycle_rows,
                    "steps_exact_message_to_teammate_reach",
                ),
                "avg_steps_exact_message_to_teammate_scan": _avg_key(
                    lifecycle_rows,
                    "steps_exact_message_to_teammate_scan",
                ),
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
                "avg_exact_target_message_steps_before_first_scan": _avg_key(
                    lifecycle_rows,
                    "exact_target_message_steps_before_first_scan",
                ),
                "avg_exact_target_message_steps_at_first_scan": _avg_key(
                    lifecycle_rows,
                    "exact_target_message_steps_at_first_scan",
                ),
                "avg_exact_target_message_steps_after_first_scan": _avg_key(
                    lifecycle_rows,
                    "exact_target_message_steps_after_first_scan",
                ),
                "episodes_with_joint_target_scan": sum(
                    1
                    for lifecycle in lifecycle_rows
                    if lifecycle.get("first_joint_target_scan_step") is not None
                ),
                "episodes_with_exact_target_message": sum(
                    1
                    for lifecycle in lifecycle_rows
                    if lifecycle.get("first_exact_target_message_step") is not None
                ),
                "diagnosis_counts": dict(sorted(diagnosis_counts.items())),
                "rendezvous_diagnosis_counts": dict(sorted(rendezvous_diagnosis_counts.items())),
            }
    pipeline_rows = [row["pipeline"] for row in rows if "pipeline" in row]
    if pipeline_rows:
        pickup_status_counts = Counter()
        delivery_decision_counts = Counter()
        wrong_delivery_provenance_counts = Counter()
        wrong_delivery_decision_counts = Counter()
        assist_correction_action_counts = Counter()
        for row in pipeline_rows:
            pickup_status_counts.update(row.get("pickup_status_counts", {}))
            delivery_decision_counts.update(row.get("delivery_decision_counts", {}))
            wrong_delivery_provenance_counts.update(row.get("wrong_delivery_provenance_counts", {}))
            wrong_delivery_decision_counts.update(row.get("wrong_delivery_decision_counts", {}))
            assist_correction_action_counts.update(row.get("assist_correction_action_counts", {}))
        pickup_attempts = sum(pickup_status_counts.values())
        delivery_interacts = sum(delivery_decision_counts.values())
        delivery_matches = int(delivery_decision_counts.get("ready_station_resource_match", 0))
        wrong_delivery_events = sum(int(row.get("wrong_delivery_events", 0)) for row in pipeline_rows)
        assist_opportunities = sum(int(row.get("assist_opportunities", 0)) for row in pipeline_rows)
        assist_correction_agents = sum(int(row.get("assist_correction_agents", 0)) for row in pipeline_rows)
        diagnostics["pipeline"] = {
            "avg_stages_completed": _avg_key(pipeline_rows, "stages_completed"),
            "avg_completion_ratio": _avg_key(pipeline_rows, "completion_ratio"),
            "avg_delivered_resources": _avg_key(pipeline_rows, "delivered_resources"),
            "avg_delivery_ratio": _avg_key(pipeline_rows, "delivery_ratio"),
            "avg_message_steps": _avg_key(pipeline_rows, "message_steps"),
            "avg_message_tokens": _avg_key(pipeline_rows, "message_tokens"),
            "avg_pickup_opportunities": _avg_key(pipeline_rows, "pickup_opportunities"),
            "avg_missed_pickup_opportunities": _avg_key(pipeline_rows, "missed_pickup_opportunities"),
            "avg_delivery_opportunities": _avg_key(pipeline_rows, "delivery_opportunities"),
            "avg_missed_delivery_opportunities": _avg_key(pipeline_rows, "missed_delivery_opportunities"),
            "avg_sync_interact_opportunities": _avg_key(pipeline_rows, "sync_interact_opportunities"),
            "avg_missed_sync_interacts": _avg_key(pipeline_rows, "missed_sync_interacts"),
            "avg_drop_actions": _avg_key(pipeline_rows, "drop_actions"),
            "avg_stall_near_resource_steps": _avg_key(pipeline_rows, "stall_near_resource_steps"),
            "avg_stall_near_station_steps": _avg_key(pipeline_rows, "stall_near_station_steps"),
            "total_pickup_attempts": int(pickup_attempts),
            "pickup_status_counts": dict(sorted(pickup_status_counts.items())),
            "pickup_needed_ready_rate": (
                float(pickup_status_counts.get("needed_ready", 0)) / float(pickup_attempts)
                if pickup_attempts else 0.0
            ),
            "pickup_unneeded_rate": (
                float(
                    pickup_status_counts.get("not_required", 0)
                    + pickup_status_counts.get("already_satisfied", 0)
                ) / float(pickup_attempts)
                if pickup_attempts else 0.0
            ),
            "total_delivery_interact_attempts": int(delivery_interacts),
            "delivery_decision_counts": dict(sorted(delivery_decision_counts.items())),
            "delivery_ready_match_rate": (
                float(delivery_matches) / float(delivery_interacts)
                if delivery_interacts else 0.0
            ),
            "avg_delivery_ready_station_distance": _avg_key(
                pipeline_rows,
                "avg_delivery_ready_station_distance",
            ),
            "avg_wrong_delivery_ready_station_distance": _avg_key(
                pipeline_rows,
                "avg_wrong_delivery_ready_station_distance",
            ),
            "wrong_delivery_events": int(wrong_delivery_events),
            "avg_wrong_delivery_events": _avg_key(pipeline_rows, "wrong_delivery_events"),
            "episodes_with_wrong_delivery_events": sum(
                1 for row in pipeline_rows if int(row.get("wrong_delivery_events", 0)) > 0
            ),
            "wrong_delivery_provenance_counts": dict(sorted(wrong_delivery_provenance_counts.items())),
            "wrong_delivery_decision_counts": dict(sorted(wrong_delivery_decision_counts.items())),
            "wrong_delivery_after_unneeded_pickup": sum(
                int(row.get("wrong_delivery_after_unneeded_pickup", 0)) for row in pipeline_rows
            ),
            "wrong_delivery_after_needed_ready_pickup": sum(
                int(row.get("wrong_delivery_after_needed_ready_pickup", 0)) for row in pipeline_rows
            ),
            "wrong_delivery_after_needed_blocked_pickup": sum(
                int(row.get("wrong_delivery_after_needed_blocked_pickup", 0)) for row in pipeline_rows
            ),
            "wrong_delivery_without_pickup_trace": sum(
                int(row.get("wrong_delivery_without_pickup_trace", 0)) for row in pipeline_rows
            ),
            "assist_opportunities": int(assist_opportunities),
            "assist_correction_agents": int(assist_correction_agents),
            "assist_correction_steps": sum(
                int(row.get("assist_correction_steps", 0)) for row in pipeline_rows
            ),
            "assist_correction_rate": (
                float(assist_correction_agents) / float(assist_opportunities)
                if assist_opportunities else 0.0
            ),
            "avg_assist_correction_agents": _avg_key(pipeline_rows, "assist_correction_agents"),
            "episodes_with_assist_corrections": sum(
                1 for row in pipeline_rows if int(row.get("assist_correction_agents", 0)) > 0
            ),
            "assist_correction_action_counts": dict(sorted(assist_correction_action_counts.items())),
            "episodes_with_all_stages_completed": sum(
                1
                for row in pipeline_rows
                if int(row.get("stages_completed", 0)) >= int(row.get("stages_total", 0))
            ),
            "episodes_with_all_resources_delivered": sum(
                1
                for row in pipeline_rows
                if int(row.get("delivered_resources", 0)) >= int(row.get("required_resources", 0))
            ),
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
            pipeline = row.get("pipeline") or {}
            if isinstance(pipeline, Mapping):
                policy_row["pipeline"] = _compact_pipeline_summary(pipeline)
            by_seed[seed][label] = policy_row
    return [by_seed[seed] for seed in sorted(by_seed)]


def _iter_events(info: Mapping[str, Any]):
    for _agent_id, event in _iter_agent_events(info):
        yield event


def _iter_agent_events(info: Mapping[str, Any]):
    events = info.get("events", {})
    if isinstance(events, Mapping):
        for agent_id, agent_events in events.items():
            if isinstance(agent_events, list):
                for event in agent_events:
                    if isinstance(event, Mapping):
                        yield int(agent_id), event


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
