"""BC -> RL curriculum runner for core SyncOrSink scenarios.

This module composes the existing centralized experts, BC/DAgger trainer,
MAPPO BC warmstart, and checkpoint evaluation into one reusable workflow.
It is intentionally trainer-facing: generated demos and checkpoints live
under ``logs/`` by default and are not benchmark artifacts.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from syncorsink.envs import SyncOrSinkConfig, SyncOrSinkEnv
from syncorsink.eval.runner import run_episodes
from syncorsink.policies.mappo_models import MAPPOActor
from syncorsink.train.bc import BCConfig, collect_demos, train_bc, train_bc_dagger
from syncorsink.train.mappo import (
    MAPPOConfig,
    action_mask_from_flat_obs,
    flatten_obs,
    load_mappo_checkpoint_policy,
    mask_action_logits,
    train_mappo,
)


SCENARIO_DEFAULTS: dict[str, dict[str, Any]] = {
    "signal_hunt": {
        "agents": 2,
        "max_steps": 60,
        "demo_episodes": 100,
        "dagger_episodes": 30,
    },
    "energy_grid": {
        "agents": 3,
        "max_steps": 80,
        "demo_episodes": 100,
        "dagger_episodes": 30,
    },
    "pipeline_assembly": {
        "agents": 3,
        "max_steps": 120,
        "demo_episodes": 200,
        "dagger_episodes": 30,
    },
}


@dataclass
class BCRLCurriculumConfig:
    scenario: str = "signal_hunt"
    map_size: int = 8
    agents: int | None = None
    fov_preset: str = "easy"
    max_steps: int | None = None
    energy_preset: str = "easy"
    energy_private_monitor: bool = True
    obs_exploration_memory: bool = False
    obs_exploration_age: bool = False

    # Communication / model shape. BC and MAPPO must agree on these.
    comm: bool = True
    comm_token_limit: int = 8
    comm_vocab_size: int = 32
    comm_max_messages: int = 8
    comm_cost: float = 0.001
    comm_len_cost: float = 0.0
    comm_send_target: float = 0.25
    comm_send_target_coeff: float = 0.05

    # Demonstrations and imitation learning.
    demo_episodes: int | None = None
    oracle: str = "oracle_strong_comm"
    dagger_rounds: int = 3
    dagger_episodes: int | None = None
    bc_epochs: int = 30
    bc_batch_size: int = 256
    bc_lr: float = 1e-3
    bc_comm_loss_weight: float = 0.1
    bc_comm_send_pos_weight: float = -1.0
    bc_two_phase: bool = False
    bc_phase1_epochs: int = 30
    bc_phase2_epochs: int = 20
    bc_phase2_lr: float = 3e-4

    # MAPPO fine-tuning.
    rl_updates: int = 3000
    rl_rollout_steps: int = 512
    rl_epochs: int = 2
    rl_minibatch: int = 256
    rl_lr: float = 3e-5
    rl_entropy: float = 0.01
    rl_anneal_lr: bool = True
    bc_kl_coeff: float = 0.5
    bc_freeze_encoder: bool = True
    critic_mode: str = "local"
    shared_actor: bool = False
    hidden_dim: int = 128
    train_eval_every: int = 50
    train_eval_episodes: int = 10

    # Final evaluation.
    eval_episodes: int = 10
    eval_seed: int = 1000
    eval_stochastic: bool = True
    eval_action_mode: str = "sample"
    eval_action_temperature: float = 1.0
    eval_send_mode: str = "threshold"
    eval_send_threshold: float = 0.25
    eval_token_mode: str = "argmax"
    eval_token_temperature: float = 1.0
    eval_length_mode: str = "argmax"
    eval_length_temperature: float = 1.0

    # Runtime.
    output_dir: str = "logs/curriculum"
    run_name: str | None = None
    seed: int = 0
    device: str = "cpu"
    dry_run: bool = False

    # Optional W&B. The BC and RL stages use separate run names with suffixes.
    wandb: bool = False
    wandb_project: str = "syncorsink-curriculum"
    wandb_run: str | None = None
    wandb_mode: str = "offline"


def run_bc_rl_curriculum(cfg: BCRLCurriculumConfig) -> dict[str, Any]:
    cfg = _resolve_defaults(cfg)
    run_dir = _make_run_dir(cfg)
    paths = _artifact_paths(run_dir, cfg.scenario)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "demos").mkdir(exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)

    result: dict[str, Any] = {
        "status": "dry_run" if cfg.dry_run else "running",
        "run_dir": str(run_dir),
        "summary_path": str(paths["summary"]),
        "config": asdict(cfg),
        "paths": {key: str(value) for key, value in paths.items()},
        "stages": _planned_stages(cfg, paths),
        "wandb_stage_runs": _planned_wandb_stage_runs(cfg),
        "wandb_summary": {"enabled": False},
    }
    _write_json(paths["summary"], result)
    if cfg.dry_run:
        return result

    _apply_wandb_mode(cfg, run_dir)

    collect_cfg = _bc_config(
        cfg,
        demo_path=paths["demo"],
        save=None,
        wandb_run_name=None,
    )
    collect_demos(collect_cfg)
    if not paths["demo"].exists():
        raise RuntimeError(f"Demo collection did not create {paths['demo']}")
    result["demo"] = _demo_summary(paths["demo"])
    _write_json(paths["summary"], result)

    bc_cfg = _bc_config(
        cfg,
        demo_path=paths["demo"],
        save=paths["bc_checkpoint"],
        wandb_run_name=_stage_wandb_name(cfg, "bc"),
    )
    if cfg.dagger_rounds > 0:
        train_bc_dagger(bc_cfg)
        result["bc_stage"] = "dagger"
    else:
        train_bc(bc_cfg)
        result["bc_stage"] = "bc"
    if not paths["bc_checkpoint"].exists():
        raise RuntimeError(f"BC stage did not create {paths['bc_checkpoint']}")

    result["bc_diagnostics"] = _bc_checkpoint_diagnostics(paths["bc_checkpoint"], paths["demo"])
    result["eval_bc"] = _eval_bc_checkpoint(cfg, paths["bc_checkpoint"], deterministic=True)
    _write_json(paths["summary"], result)

    train_cfg = _mappo_config(cfg, bc_checkpoint=paths["bc_checkpoint"], save=paths["rl_checkpoint"])
    train_mappo(train_cfg)
    if not paths["rl_checkpoint"].exists():
        raise RuntimeError(f"MAPPO fine-tuning did not create {paths['rl_checkpoint']}")

    result["eval_rl_deterministic"] = _eval_mappo_checkpoint(
        cfg,
        paths["rl_checkpoint"],
        train_cfg=train_cfg,
        deterministic=True,
    )
    if cfg.eval_stochastic:
        result["eval_rl_stochastic"] = _eval_mappo_checkpoint(
            cfg,
            paths["rl_checkpoint"],
            train_cfg=train_cfg,
            deterministic=False,
            decode_kwargs=_eval_decode_kwargs(cfg),
        )
    if paths["rl_best_checkpoint"].exists():
        result["best_eval_checkpoint"] = _checkpoint_eval_metadata(paths["rl_best_checkpoint"])
        result["eval_rl_best_deterministic"] = _eval_mappo_checkpoint(
            cfg,
            paths["rl_best_checkpoint"],
            train_cfg=train_cfg,
            deterministic=True,
        )
        if cfg.eval_stochastic:
            result["eval_rl_best_stochastic"] = _eval_mappo_checkpoint(
                cfg,
                paths["rl_best_checkpoint"],
                train_cfg=train_cfg,
                deterministic=False,
                decode_kwargs=_eval_decode_kwargs(cfg),
            )

    result["status"] = "complete"
    _write_json(paths["summary"], result)
    result["wandb_summary"] = _log_curriculum_summary(cfg, run_dir, paths["summary"], result)
    _write_json(paths["summary"], result)
    return result


def _resolve_defaults(cfg: BCRLCurriculumConfig) -> BCRLCurriculumConfig:
    if cfg.scenario not in SCENARIO_DEFAULTS:
        raise ValueError(f"Unknown curriculum scenario: {cfg.scenario}")
    defaults = SCENARIO_DEFAULTS[cfg.scenario]
    oracle = cfg.oracle
    if cfg.comm and not oracle.endswith("_comm"):
        oracle = f"{oracle}_comm"
    if not cfg.comm and oracle.endswith("_comm"):
        oracle = oracle.removesuffix("_comm")
    return replace(
        cfg,
        agents=cfg.agents if cfg.agents is not None else int(defaults["agents"]),
        max_steps=cfg.max_steps if cfg.max_steps is not None else int(defaults["max_steps"]),
        demo_episodes=(
            cfg.demo_episodes
            if cfg.demo_episodes is not None
            else int(defaults["demo_episodes"])
        ),
        dagger_episodes=(
            cfg.dagger_episodes
            if cfg.dagger_episodes is not None
            else int(defaults["dagger_episodes"])
        ),
        oracle=oracle,
    )


def _make_run_dir(cfg: BCRLCurriculumConfig) -> Path:
    if cfg.run_name:
        name = cfg.run_name
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        name = f"{cfg.scenario}_{cfg.map_size}x{cfg.map_size}_seed{cfg.seed}_{stamp}"
    return Path(cfg.output_dir) / name


def _artifact_paths(run_dir: Path, scenario: str) -> dict[str, Path]:
    return {
        "demo": run_dir / "demos" / f"{scenario}_oracle.npz",
        "bc_checkpoint": run_dir / "checkpoints" / "bc_dagger.pt",
        "rl_checkpoint": run_dir / "checkpoints" / "mappo_bc_rl.pt",
        "rl_best_checkpoint": run_dir / "checkpoints" / "mappo_bc_rl_best.pt",
        "summary": run_dir / "summary.json",
    }


def _planned_stages(cfg: BCRLCurriculumConfig, paths: dict[str, Path]) -> list[dict[str, Any]]:
    return [
        {
            "name": "collect_demos",
            "oracle": cfg.oracle,
            "episodes": cfg.demo_episodes,
            "output": str(paths["demo"]),
        },
        {
            "name": "dagger" if cfg.dagger_rounds > 0 else "bc",
            "rounds": cfg.dagger_rounds,
            "epochs": cfg.bc_epochs,
            "checkpoint": str(paths["bc_checkpoint"]),
        },
        {
            "name": "eval_bc",
            "episodes": cfg.eval_episodes,
            "deterministic": True,
        },
        {
            "name": "mappo_bc_rl",
            "updates": cfg.rl_updates,
            "bc_kl_coeff": cfg.bc_kl_coeff,
            "checkpoint": str(paths["rl_checkpoint"]),
            "best_checkpoint": str(paths["rl_best_checkpoint"]),
        },
        {
            "name": "eval_rl",
            "episodes": cfg.eval_episodes,
            "deterministic": True,
            "stochastic": cfg.eval_stochastic,
            "stochastic_decode": _eval_decode_kwargs(cfg),
        },
    ]


def _planned_wandb_stage_runs(cfg: BCRLCurriculumConfig) -> dict[str, Any]:
    if not cfg.wandb:
        return {}
    return {
        "bc": _stage_wandb_name(cfg, "bc") or "",
        "rl": _stage_wandb_name(cfg, "rl") or "",
        "summary": _stage_wandb_name(cfg, "summary") or "",
        "project": cfg.wandb_project,
        "mode": cfg.wandb_mode,
    }


def _bc_config(
    cfg: BCRLCurriculumConfig,
    *,
    demo_path: Path,
    save: Path | None,
    wandb_run_name: str | None,
) -> BCConfig:
    return BCConfig(
        scenario=cfg.scenario,
        map_size=cfg.map_size,
        agents=int(cfg.agents),
        fov_preset=cfg.fov_preset,
        max_steps=int(cfg.max_steps),
        energy_preset=cfg.energy_preset,
        obs_exploration_memory=cfg.obs_exploration_memory,
        obs_exploration_age=cfg.obs_exploration_age,
        demo_episodes=int(cfg.demo_episodes),
        oracle_type=cfg.oracle,
        demo_path=str(demo_path),
        epochs=cfg.bc_epochs,
        batch_size=cfg.bc_batch_size,
        lr=cfg.bc_lr,
        hidden_dim=cfg.hidden_dim,
        comm=cfg.comm,
        comm_token_limit=cfg.comm_token_limit,
        comm_vocab_size=cfg.comm_vocab_size,
        comm_loss_weight=cfg.bc_comm_loss_weight,
        comm_send_pos_weight=cfg.bc_comm_send_pos_weight,
        two_phase=cfg.bc_two_phase,
        phase1_epochs=cfg.bc_phase1_epochs,
        phase2_epochs=cfg.bc_phase2_epochs,
        phase2_lr=cfg.bc_phase2_lr,
        dagger_rounds=cfg.dagger_rounds,
        dagger_episodes=int(cfg.dagger_episodes),
        device=cfg.device,
        seed=cfg.seed,
        save=str(save) if save is not None else None,
        wandb=cfg.wandb,
        wandb_project=cfg.wandb_project,
        wandb_run=wandb_run_name,
    )


def _mappo_config(
    cfg: BCRLCurriculumConfig,
    *,
    bc_checkpoint: Path,
    save: Path,
) -> MAPPOConfig:
    kwargs: dict[str, Any] = {
        "scenario": cfg.scenario,
        "map_size": cfg.map_size,
        "agents": int(cfg.agents),
        "fov_preset": cfg.fov_preset,
        "max_steps": int(cfg.max_steps),
        "energy_preset": cfg.energy_preset,
        "energy_private_monitor": cfg.energy_private_monitor,
        "obs_exploration_memory": cfg.obs_exploration_memory,
        "obs_exploration_age": cfg.obs_exploration_age,
        "comm": cfg.comm,
        "comm_token_limit": cfg.comm_token_limit,
        "comm_vocab_size": cfg.comm_vocab_size,
        "comm_max_messages": cfg.comm_max_messages,
        "comm_cost": cfg.comm_cost,
        "comm_len_cost": cfg.comm_len_cost,
        "comm_send_target": cfg.comm_send_target,
        "comm_send_target_coeff": cfg.comm_send_target_coeff,
        "updates": cfg.rl_updates,
        "rollout_steps": cfg.rl_rollout_steps,
        "epochs": cfg.rl_epochs,
        "minibatch": cfg.rl_minibatch,
        "entropy": cfg.rl_entropy,
        "lr": cfg.rl_lr,
        "anneal_lr": cfg.rl_anneal_lr,
        "critic_mode": cfg.critic_mode,
        "shared_actor": cfg.shared_actor,
        "hidden_dim": cfg.hidden_dim,
        "seed": cfg.seed,
        "bc_init": str(bc_checkpoint),
        "bc_kl_coeff": cfg.bc_kl_coeff,
        "bc_freeze_encoder": cfg.bc_freeze_encoder,
        "device": cfg.device,
        "wandb": cfg.wandb,
        "wandb_project": cfg.wandb_project,
        "wandb_run": _stage_wandb_name(cfg, "rl"),
        "save": str(save),
        "save_best": str(Path(save).with_name("mappo_bc_rl_best.pt")),
        "save_every": max(1, cfg.rl_updates),
        "eval_every": cfg.train_eval_every,
        "eval_episodes": cfg.train_eval_episodes,
        "eval_action_mode": cfg.eval_action_mode,
        "eval_action_temperature": cfg.eval_action_temperature,
        "eval_send_mode": cfg.eval_send_mode,
        "eval_send_threshold": cfg.eval_send_threshold,
        "eval_token_mode": cfg.eval_token_mode,
        "eval_token_temperature": cfg.eval_token_temperature,
        "eval_length_mode": cfg.eval_length_mode,
        "eval_length_temperature": cfg.eval_length_temperature,
    }
    kwargs.update(_scenario_shaping_kwargs(cfg.scenario))
    return MAPPOConfig(**kwargs)


def _scenario_shaping_kwargs(scenario: str) -> dict[str, Any]:
    if scenario == "signal_hunt":
        return {
            "signal_shaping": True,
            "signal_shaping_scale": 0.1,
            "signal_scan_bonus": 0.2,
            "signal_joint_scan_bonus": 3.0,
            "signal_colocation_bonus": 0.5,
            "signal_comm_utility": 0.1,
        }
    if scenario == "energy_grid":
        return {
            "energy_shaping": True,
            "energy_shaping_scale": 0.1,
        }
    if scenario == "pipeline_assembly":
        return {
            "pipeline_shaping": True,
            "pipeline_shaping_scale": 0.1,
        }
    return {}


def _eval_env_config(cfg: BCRLCurriculumConfig) -> SyncOrSinkConfig:
    kwargs = {
        "scenario": cfg.scenario,
        "map_size": cfg.map_size,
        "num_agents": int(cfg.agents),
        "fov_preset": cfg.fov_preset,
        "max_steps": int(cfg.max_steps),
        "energy_preset": cfg.energy_preset,
        "energy_private_monitor": cfg.energy_private_monitor,
        "obs_exploration_memory": cfg.obs_exploration_memory,
        "obs_exploration_age": cfg.obs_exploration_age,
        "comm_token_limit": cfg.comm_token_limit,
        "token_vocab_size": cfg.comm_vocab_size,
        "max_messages": cfg.comm_max_messages,
        "comm_cost": cfg.comm_cost,
        "comm_len_cost": cfg.comm_len_cost,
    }
    kwargs.update(_scenario_shaping_kwargs(cfg.scenario))
    return SyncOrSinkConfig(**kwargs)


def _eval_bc_checkpoint(
    cfg: BCRLCurriculumConfig,
    checkpoint_path: Path,
    *,
    deterministic: bool,
) -> dict[str, Any]:
    env = SyncOrSinkEnv(_eval_env_config(cfg))
    policy = _load_bc_policy(checkpoint_path, deterministic=deterministic, device=cfg.device)
    summary, episodes = run_episodes(env, policy, episodes=cfg.eval_episodes, seed=cfg.eval_seed)
    return {
        "summary": asdict(summary),
        "episodes": [asdict(ep) for ep in episodes],
        "deterministic": deterministic,
    }


def _eval_mappo_checkpoint(
    cfg: BCRLCurriculumConfig,
    checkpoint_path: Path,
    *,
    train_cfg: MAPPOConfig,
    deterministic: bool,
    decode_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    env = SyncOrSinkEnv(_eval_env_config(cfg))
    decode_kwargs = decode_kwargs or {}
    policy = load_mappo_checkpoint_policy(
        checkpoint_path,
        env,
        cfg=train_cfg,
        deterministic=deterministic,
        device=cfg.device,
        sample_seed=cfg.eval_seed,
        **decode_kwargs,
    )
    summary, episodes = run_episodes(env, policy, episodes=cfg.eval_episodes, seed=cfg.eval_seed)
    return {
        "summary": asdict(summary),
        "episodes": [asdict(ep) for ep in episodes],
        "deterministic": deterministic,
        "decode": policy.metadata(),
    }


def _eval_decode_kwargs(cfg: BCRLCurriculumConfig) -> dict[str, Any]:
    return {
        "action_mode": cfg.eval_action_mode,
        "action_temperature": cfg.eval_action_temperature,
        "send_mode": cfg.eval_send_mode,
        "send_threshold": cfg.eval_send_threshold,
        "token_mode": cfg.eval_token_mode,
        "token_temperature": cfg.eval_token_temperature,
        "length_mode": cfg.eval_length_mode,
        "length_temperature": cfg.eval_length_temperature,
    }


def _load_bc_policy(
    checkpoint_path: Path,
    *,
    deterministic: bool,
    device: str,
):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    device_obj = torch.device(device if device != "auto" else "cpu")
    model = MAPPOActor(
        obs_dim=int(ckpt["obs_dim"]),
        action_dim=8,
        hidden_dim=int(ckpt["hidden_dim"]),
        backbone="mlp",
        comm_enabled=bool(ckpt.get("comm", False)),
        comm_token_limit=int(ckpt.get("comm_token_limit", 0)),
        comm_vocab_size=int(ckpt.get("comm_vocab_size", 0)),
    ).to(device_obj)
    model.load_state_dict(ckpt["model"])
    model.eval()

    def _policy(obs: dict, info: dict, state: dict) -> dict[int, dict]:
        del info, state
        actions: dict[int, dict] = {}
        for aid in sorted(obs.keys()):
            flat_arr = _fit_obs_dim(
                flatten_obs(
                    obs[aid],
                    include_exploration_memory=bool(ckpt.get("obs_exploration_memory", False)),
                    include_exploration_age=bool(ckpt.get("obs_exploration_age", False)),
                ),
                int(ckpt["obs_dim"]),
            )
            flat = torch.tensor(flat_arr, dtype=torch.float32, device=device_obj).unsqueeze(0)
            with torch.no_grad():
                out = model(flat)
            if bool(ckpt.get("comm", False)):
                logits, send_logits, token_logits, len_logits = out
                logits = mask_action_logits(logits, action_mask_from_flat_obs(flat))
                if deterministic:
                    act = int(torch.argmax(logits, dim=-1).item())
                    send = int(torch.sigmoid(send_logits.squeeze(-1)).item() > 0.5)
                    msg_len = int(torch.argmax(len_logits, dim=-1).item()) if send else 0
                    msg_tokens = torch.argmax(token_logits, dim=-1)[0, :msg_len].tolist()
                else:
                    act = int(torch.distributions.Categorical(logits=logits).sample().item())
                    send = int(torch.distributions.Bernoulli(logits=send_logits.squeeze(-1)).sample().item())
                    msg_len = int(torch.distributions.Categorical(logits=len_logits).sample().item()) if send else 0
                    token_dist = torch.distributions.Categorical(logits=token_logits)
                    msg_tokens = token_dist.sample()[0, :msg_len].tolist()
                actions[aid] = {
                    "action": act,
                    "message_tokens": [int(token) for token in msg_tokens],
                }
            else:
                logits = mask_action_logits(out, action_mask_from_flat_obs(flat))
                if deterministic:
                    act = int(torch.argmax(logits, dim=-1).item())
                else:
                    act = int(torch.distributions.Categorical(logits=logits).sample().item())
                actions[aid] = {"action": act, "message_tokens": []}
        return actions

    return _policy


def _fit_obs_dim(obs: np.ndarray, target_dim: int) -> np.ndarray:
    if obs.shape[0] > target_dim:
        return obs[:target_dim]
    if obs.shape[0] < target_dim:
        return np.pad(obs, (0, target_dim - obs.shape[0]))
    return obs


def _demo_summary(path: Path) -> dict[str, Any]:
    data = np.load(path)
    actions = data["actions"]
    msg_lens = data["msg_lens"] if "msg_lens" in data else np.zeros(len(actions), dtype=np.int64)
    return {
        "path": str(path),
        "transitions": int(actions.shape[0]),
        "obs_dim": int(data["obs"].shape[1]),
        "action_hist": np.bincount(actions, minlength=8).astype(int).tolist(),
        "mean_message_len": float(msg_lens.mean()) if len(msg_lens) else 0.0,
        "send_rate": float((msg_lens > 0).mean()) if len(msg_lens) else 0.0,
    }


def _bc_checkpoint_diagnostics(checkpoint_path: Path, demo_path: Path) -> dict[str, Any]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    data = np.load(demo_path)
    obs = torch.tensor(data["obs"], dtype=torch.float32)
    actions = torch.tensor(data["actions"], dtype=torch.long)
    msg_lens = torch.tensor(data["msg_lens"], dtype=torch.long)
    model = MAPPOActor(
        obs_dim=int(ckpt["obs_dim"]),
        action_dim=8,
        hidden_dim=int(ckpt["hidden_dim"]),
        backbone="mlp",
        comm_enabled=bool(ckpt.get("comm", False)),
        comm_token_limit=int(ckpt.get("comm_token_limit", 0)),
        comm_vocab_size=int(ckpt.get("comm_vocab_size", 0)),
    )
    model.load_state_dict(ckpt["model"])
    model.eval()
    with torch.no_grad():
        out = model(obs)
    if bool(ckpt.get("comm", False)):
        logits, send_logits, _token_logits, len_logits = out
        send_prob = torch.sigmoid(send_logits.squeeze(-1))
        positive = msg_lens > 0
        negative = ~positive
        diagnostics = {
            "action_accuracy": float((logits.argmax(dim=-1) == actions).float().mean().item()),
            "demo_send_rate": float(positive.float().mean().item()),
            "send_prob_mean": float(send_prob.mean().item()),
            "send_prob_positive_mean": float(send_prob[positive].mean().item()) if positive.any() else 0.0,
            "send_prob_negative_mean": float(send_prob[negative].mean().item()) if negative.any() else 0.0,
            "pred_send_rate_threshold_0_50": float((send_prob > 0.5).float().mean().item()),
            "pred_send_rate_threshold_0_25": float((send_prob > 0.25).float().mean().item()),
            "length_accuracy": float((len_logits.argmax(dim=-1) == msg_lens).float().mean().item()),
        }
    else:
        diagnostics = {
            "action_accuracy": float((out.argmax(dim=-1) == actions).float().mean().item()),
            "demo_send_rate": 0.0,
        }
    return diagnostics


def _checkpoint_eval_metadata(checkpoint_path: Path) -> dict[str, Any]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    payload = {"path": str(checkpoint_path)}
    if isinstance(ckpt.get("best_eval"), dict):
        payload.update(ckpt["best_eval"])
    if "step" in ckpt:
        payload["step"] = int(ckpt["step"])
    return payload


def _stage_wandb_name(cfg: BCRLCurriculumConfig, stage: str) -> str | None:
    if not cfg.wandb:
        return None
    base = cfg.wandb_run or cfg.run_name or f"{cfg.scenario}_{cfg.map_size}x{cfg.map_size}_seed{cfg.seed}"
    return f"{base}_{stage}"


def _apply_wandb_mode(cfg: BCRLCurriculumConfig, run_dir: Path) -> None:
    if cfg.wandb:
        os.environ["WANDB_MODE"] = cfg.wandb_mode
        wandb_dir = run_dir / "wandb"
        data_dir = wandb_dir / "data"
        artifact_dir = wandb_dir / "artifacts"
        cache_dir = wandb_dir / "cache"
        config_dir = wandb_dir / "config"
        for directory in (wandb_dir, data_dir, artifact_dir, cache_dir, config_dir):
            directory.mkdir(parents=True, exist_ok=True)
        os.environ["WANDB_DIR"] = str(wandb_dir)
        os.environ["WANDB_DATA_DIR"] = str(data_dir)
        os.environ["WANDB_ARTIFACT_DIR"] = str(artifact_dir)
        os.environ["WANDB_CACHE_DIR"] = str(cache_dir)
        os.environ["WANDB_CONFIG_DIR"] = str(config_dir)


def _log_curriculum_summary(
    cfg: BCRLCurriculumConfig,
    run_dir: Path,
    summary_path: Path,
    result: dict[str, Any],
) -> dict[str, Any]:
    if not cfg.wandb:
        return {"enabled": False}
    try:
        import wandb
    except Exception as exc:
        return {"enabled": False, "error": f"wandb unavailable: {exc}"}

    run_name = _stage_wandb_name(cfg, "summary")
    try:
        run = wandb.init(
            project=cfg.wandb_project,
            name=run_name,
            mode=cfg.wandb_mode,
            config=asdict(cfg),
            dir=str(run_dir / "wandb"),
        )
    except Exception as exc:
        return {"enabled": False, "error": str(exc), "mode": cfg.wandb_mode}

    try:
        payload = _curriculum_wandb_payload(result)
        _log_curriculum_stage_series(run, result)
        run.log(payload)
        _write_curriculum_decode_summary(run, result)
        run.summary["status"] = result.get("status")
        run.summary["summary_path"] = str(summary_path)
        if cfg.wandb_mode != "disabled":
            artifact = wandb.Artifact(run_dir.name, type="bc_rl_curriculum_summary")
            artifact.add_file(str(summary_path))
            run.log_artifact(artifact)
        return {
            "enabled": True,
            "mode": cfg.wandb_mode,
            "run_name": run_name,
            "url": getattr(run, "url", None),
            "logged_keys": sorted(payload.keys()),
        }
    except Exception as exc:
        return {
            "enabled": True,
            "mode": cfg.wandb_mode,
            "run_name": run_name,
            "error": str(exc),
        }
    finally:
        run.finish()


def _curriculum_wandb_payload(result: dict[str, Any]) -> dict[str, float | int]:
    payload: dict[str, float | int] = {}
    demo = result.get("demo") or {}
    if "transitions" in demo:
        payload["demo/transitions"] = int(demo["transitions"])
    if "mean_message_len" in demo:
        payload["demo/mean_message_len"] = float(demo["mean_message_len"])
    if "send_rate" in demo:
        payload["demo/send_rate"] = float(demo["send_rate"])
    for key, value in (result.get("bc_diagnostics") or {}).items():
        if isinstance(value, (int, float)):
            payload[f"bc/{key}"] = float(value)
    for stage_name, prefix in [
        ("eval_bc", "eval_bc"),
        ("eval_rl_deterministic", "eval_rl_deterministic"),
        ("eval_rl_stochastic", "eval_rl_stochastic"),
        ("eval_rl_best_deterministic", "eval_rl_best_deterministic"),
        ("eval_rl_best_stochastic", "eval_rl_best_stochastic"),
    ]:
        summary = (result.get(stage_name) or {}).get("summary") or {}
        for key in ("episodes", "success_rate", "avg_return", "avg_steps", "avg_comm_tokens"):
            if key in summary:
                value = summary[key]
                payload[f"{prefix}/{key}"] = int(value) if key == "episodes" else float(value)
    return payload


def _log_curriculum_stage_series(run, result: dict[str, Any]) -> None:
    stages = [
        ("bc", "eval_bc"),
        ("rl_deterministic", "eval_rl_deterministic"),
        ("rl_stochastic", "eval_rl_stochastic"),
        ("rl_best_deterministic", "eval_rl_best_deterministic"),
        ("rl_best_stochastic", "eval_rl_best_stochastic"),
    ]
    try:
        run.define_metric("stage_idx")
        run.define_metric("stage_eval/*", step_metric="stage_idx")
    except Exception:
        pass

    table = None
    try:
        import wandb
        table = wandb.Table(columns=[
            "stage",
            "success_rate",
            "avg_return",
            "avg_steps",
            "avg_comm_tokens",
        ])
    except Exception:
        table = None

    for idx, (stage_label, result_key) in enumerate(stages):
        summary = (result.get(result_key) or {}).get("summary") or {}
        if not summary:
            continue
        row = {
            "stage_idx": idx,
            "stage_name": stage_label,
            "stage_eval/success_rate": float(summary.get("success_rate", 0.0)),
            "stage_eval/avg_return": float(summary.get("avg_return", 0.0)),
            "stage_eval/avg_steps": float(summary.get("avg_steps", 0.0)),
            "stage_eval/avg_comm_tokens": float(summary.get("avg_comm_tokens", 0.0)),
        }
        run.log(row)
        if table is not None:
            table.add_data(
                stage_label,
                row["stage_eval/success_rate"],
                row["stage_eval/avg_return"],
                row["stage_eval/avg_steps"],
                row["stage_eval/avg_comm_tokens"],
            )
    if table is not None:
        run.log({"curriculum/eval_table": table})


def _write_curriculum_decode_summary(run, result: dict[str, Any]) -> None:
    for result_key in ("eval_rl_stochastic", "eval_rl_best_stochastic"):
        decode = (result.get(result_key) or {}).get("decode") or {}
        for key in (
            "action_mode",
            "action_temperature",
            "send_mode",
            "send_threshold",
            "token_mode",
            "token_temperature",
            "length_mode",
            "length_temperature",
        ):
            if key in decode:
                run.summary[f"{result_key}/decode/{key}"] = decode[key]


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
