"""Evaluate MAPPO checkpoints under explicit inference-time decoding settings."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
import csv
from itertools import product
import json
import os
from pathlib import Path
from typing import Any

import torch

from syncorsink.envs import SyncOrSinkConfig, SyncOrSinkEnv
from syncorsink.eval.runner import run_episodes
from syncorsink.train.mappo import MAPPOConfig, load_mappo_checkpoint_policy
from syncorsink.train.seed import set_global_seeds


@dataclass
class MAPPODecodingSweepConfig:
    checkpoints: list[str] | tuple[str, ...]
    checkpoint_labels: list[str] | tuple[str, ...] | None = None

    scenario: str = "signal_hunt"
    map_size: int = 8
    agents: int = 2
    fov_preset: str = "easy"
    max_steps: int = 60
    energy_preset: str = "easy"
    energy_private_monitor: bool = True
    obs_exploration_memory: bool = False
    obs_exploration_age: bool = False
    comm_token_limit: int = 8
    comm_vocab_size: int = 32
    comm_max_messages: int = 8
    comm_cost: float = 0.001
    comm_len_cost: float = 0.0

    pipeline_shaping: bool = False
    pipeline_shaping_scale: float = 0.01
    energy_shaping: bool = False
    energy_shaping_scale: float = 0.01
    signal_shaping: bool = True
    signal_shaping_scale: float = 0.1
    signal_scan_bonus: float = 0.2
    signal_joint_scan_bonus: float = 3.0
    signal_colocation_bonus: float = 0.5
    signal_colocation_radius: int = 2
    signal_comm_utility: float = 0.1
    signal_target_visit_bonus: float = 0.0
    signal_decoy_visit_penalty: float = 0.0
    signal_unique_target_scan_bonus: float = 0.0

    episodes: int = 10
    seed: int = 1000
    device: str = "cpu"

    action_modes: tuple[str, ...] = ("argmax", "sample")
    action_temperatures: tuple[float, ...] = (1.0,)
    send_modes: tuple[str, ...] = ("threshold",)
    send_thresholds: tuple[float, ...] = (0.25, 0.5)
    token_modes: tuple[str, ...] = ("argmax", "sample")
    token_temperatures: tuple[float, ...] = (1.0,)
    length_modes: tuple[str, ...] = ("argmax", "sample")
    length_temperatures: tuple[float, ...] = (1.0,)

    output_dir: str = "logs/decoding_sweep"
    run_name: str | None = None

    wandb: bool = False
    wandb_project: str = "syncorsink-curriculum"
    wandb_run: str | None = None
    wandb_mode: str = "offline"


def run_mappo_decoding_sweep(cfg: MAPPODecodingSweepConfig) -> dict[str, Any]:
    """Run a checkpoint decoding sweep and persist ranked JSON/CSV artifacts."""
    _validate_config(cfg)
    run_dir = _make_run_dir(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    json_path = run_dir / "summary.json"
    csv_path = run_dir / "results.csv"

    env_config = _env_config(cfg)
    checkpoint_specs = _checkpoint_specs(cfg)
    combo_specs = _combo_specs(cfg)
    result: dict[str, Any] = {
        "status": "running",
        "run_dir": str(run_dir),
        "summary_path": str(json_path),
        "csv_path": str(csv_path),
        "config": asdict(cfg),
        "env_config": asdict(env_config),
        "checkpoint_count": len(checkpoint_specs),
        "combo_count": len(combo_specs),
        "rows": [],
        "wandb": {"enabled": False},
    }
    _write_json(json_path, result)

    rows: list[dict[str, Any]] = []
    for checkpoint_label, checkpoint_path in checkpoint_specs:
        checkpoint_cfg = _read_mappo_config(checkpoint_path)
        checkpoint_meta = _checkpoint_meta(checkpoint_path)
        env = SyncOrSinkEnv(env_config)
        for combo_index, combo in enumerate(combo_specs):
            set_global_seeds(cfg.seed + combo_index)
            deterministic = (
                combo["action_mode"] == "argmax"
                and combo["send_mode"] == "threshold"
                and combo["token_mode"] == "argmax"
                and combo["length_mode"] == "argmax"
            )
            policy = load_mappo_checkpoint_policy(
                checkpoint_path,
                env,
                cfg=checkpoint_cfg,
                deterministic=deterministic,
                device=cfg.device,
                sample_seed=cfg.seed,
                send_threshold=combo["send_threshold"],
                action_mode=combo["action_mode"],
                action_temperature=combo["action_temperature"],
                send_mode=combo["send_mode"],
                token_mode=combo["token_mode"],
                token_temperature=combo["token_temperature"],
                length_mode=combo["length_mode"],
                length_temperature=combo["length_temperature"],
            )
            summary, episodes = run_episodes(
                env,
                policy,
                episodes=cfg.episodes,
                seed=cfg.seed,
            )
            rows.append({
                "checkpoint_label": checkpoint_label,
                "checkpoint_path": str(checkpoint_path),
                **checkpoint_meta,
                **combo,
                "deterministic": deterministic,
                "policy_metadata": policy.metadata(),
                "summary": asdict(summary),
                "episodes": [asdict(ep) for ep in episodes],
            })

    rows = _rank_rows(rows)
    result["status"] = "complete"
    result["rows"] = rows
    result["best_row"] = rows[0] if rows else None
    _write_json(json_path, result)
    _write_csv(csv_path, rows)
    result["wandb"] = _log_wandb(cfg, run_dir, json_path, csv_path, rows)
    _write_json(json_path, result)
    return result


def _validate_config(cfg: MAPPODecodingSweepConfig) -> None:
    if not cfg.checkpoints:
        raise ValueError("at least one checkpoint is required")
    if cfg.checkpoint_labels is not None and len(cfg.checkpoint_labels) != len(cfg.checkpoints):
        raise ValueError("checkpoint_labels must match checkpoints length")
    valid_action = {"argmax", "sample"}
    valid_send = {"threshold", "sample"}
    if any(mode not in valid_action for mode in cfg.action_modes):
        raise ValueError(f"action_modes must be in {sorted(valid_action)}")
    if any(mode not in valid_send for mode in cfg.send_modes):
        raise ValueError(f"send_modes must be in {sorted(valid_send)}")
    if any(mode not in valid_action for mode in cfg.token_modes):
        raise ValueError(f"token_modes must be in {sorted(valid_action)}")
    if any(mode not in valid_action for mode in cfg.length_modes):
        raise ValueError(f"length_modes must be in {sorted(valid_action)}")
    if cfg.episodes < 1:
        raise ValueError("episodes must be >= 1")


def _make_run_dir(cfg: MAPPODecodingSweepConfig) -> Path:
    if cfg.run_name:
        name = cfg.run_name
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        name = f"{cfg.scenario}_{cfg.map_size}x{cfg.map_size}_decode_seed{cfg.seed}_{stamp}"
    return Path(cfg.output_dir) / name


def _env_config(cfg: MAPPODecodingSweepConfig) -> SyncOrSinkConfig:
    return SyncOrSinkConfig(
        scenario=cfg.scenario,
        map_size=cfg.map_size,
        num_agents=cfg.agents,
        fov_preset=cfg.fov_preset,
        max_steps=cfg.max_steps,
        energy_preset=cfg.energy_preset,
        energy_private_monitor=cfg.energy_private_monitor,
        obs_exploration_memory=cfg.obs_exploration_memory,
        obs_exploration_age=cfg.obs_exploration_age,
        comm_token_limit=cfg.comm_token_limit,
        token_vocab_size=cfg.comm_vocab_size,
        max_messages=cfg.comm_max_messages,
        comm_cost=cfg.comm_cost,
        comm_len_cost=cfg.comm_len_cost,
        pipeline_shaping=cfg.pipeline_shaping,
        pipeline_shaping_scale=cfg.pipeline_shaping_scale,
        energy_shaping=cfg.energy_shaping,
        energy_shaping_scale=cfg.energy_shaping_scale,
        signal_shaping=cfg.signal_shaping,
        signal_shaping_scale=cfg.signal_shaping_scale,
        signal_scan_bonus=cfg.signal_scan_bonus,
        signal_joint_scan_bonus=cfg.signal_joint_scan_bonus,
        signal_colocation_bonus=cfg.signal_colocation_bonus,
        signal_colocation_radius=cfg.signal_colocation_radius,
        signal_comm_utility=cfg.signal_comm_utility,
        signal_target_visit_bonus=cfg.signal_target_visit_bonus,
        signal_decoy_visit_penalty=cfg.signal_decoy_visit_penalty,
        signal_unique_target_scan_bonus=cfg.signal_unique_target_scan_bonus,
    )


def _checkpoint_specs(cfg: MAPPODecodingSweepConfig) -> list[tuple[str, Path]]:
    labels = cfg.checkpoint_labels or tuple(Path(path).stem for path in cfg.checkpoints)
    return [
        (str(label), Path(path))
        for label, path in zip(labels, cfg.checkpoints)
    ]


def _combo_specs(cfg: MAPPODecodingSweepConfig) -> list[dict[str, Any]]:
    combos = []
    for (
        action_mode,
        action_temperature,
        send_mode,
        send_threshold,
        token_mode,
        token_temperature,
        length_mode,
        length_temperature,
    ) in product(
        cfg.action_modes,
        cfg.action_temperatures,
        cfg.send_modes,
        cfg.send_thresholds,
        cfg.token_modes,
        cfg.token_temperatures,
        cfg.length_modes,
        cfg.length_temperatures,
    ):
        combos.append({
            "action_mode": action_mode,
            "action_temperature": float(action_temperature),
            "send_mode": send_mode,
            "send_threshold": float(send_threshold),
            "token_mode": token_mode,
            "token_temperature": float(token_temperature),
            "length_mode": length_mode,
            "length_temperature": float(length_temperature),
        })
    return combos


def _read_mappo_config(checkpoint_path: Path) -> MAPPOConfig | None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    raw = checkpoint.get("config")
    if not isinstance(raw, dict):
        return None
    allowed = {field.name for field in fields(MAPPOConfig)}
    return MAPPOConfig(**{key: value for key, value in raw.items() if key in allowed})


def _checkpoint_meta(checkpoint_path: Path) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    meta: dict[str, Any] = {}
    if "step" in checkpoint:
        meta["checkpoint_step"] = int(checkpoint["step"])
    if "checkpoint_role" in checkpoint:
        meta["checkpoint_role"] = checkpoint["checkpoint_role"]
    best_eval = checkpoint.get("best_eval")
    if isinstance(best_eval, dict):
        for key in ("update", "success_rate", "mean_return", "mean_steps"):
            if key in best_eval:
                meta[f"checkpoint_best_{key}"] = best_eval[key]
    return meta


def _rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(
        rows,
        key=lambda row: (
            row["summary"]["success_rate"],
            row["summary"]["avg_return"],
            -row["summary"]["avg_steps"],
            -row["summary"]["avg_comm_tokens"],
        ),
        reverse=True,
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    return ranked


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "checkpoint_label",
        "checkpoint_path",
        "checkpoint_step",
        "checkpoint_role",
        "checkpoint_best_update",
        "checkpoint_best_success_rate",
        "checkpoint_best_mean_return",
        "checkpoint_best_mean_steps",
        "action_mode",
        "action_temperature",
        "send_mode",
        "send_threshold",
        "token_mode",
        "token_temperature",
        "length_mode",
        "length_temperature",
        "deterministic",
        "episodes",
        "success_rate",
        "avg_return",
        "avg_steps",
        "avg_comm_tokens",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            summary = row["summary"]
            writer.writerow({
                "rank": row["rank"],
                "checkpoint_label": row["checkpoint_label"],
                "checkpoint_path": row["checkpoint_path"],
                "checkpoint_step": row.get("checkpoint_step", ""),
                "checkpoint_role": row.get("checkpoint_role", ""),
                "checkpoint_best_update": row.get("checkpoint_best_update", ""),
                "checkpoint_best_success_rate": row.get("checkpoint_best_success_rate", ""),
                "checkpoint_best_mean_return": row.get("checkpoint_best_mean_return", ""),
                "checkpoint_best_mean_steps": row.get("checkpoint_best_mean_steps", ""),
                "action_mode": row["action_mode"],
                "action_temperature": row["action_temperature"],
                "send_mode": row["send_mode"],
                "send_threshold": row["send_threshold"],
                "token_mode": row["token_mode"],
                "token_temperature": row["token_temperature"],
                "length_mode": row["length_mode"],
                "length_temperature": row["length_temperature"],
                "deterministic": row["deterministic"],
                "episodes": summary["episodes"],
                "success_rate": summary["success_rate"],
                "avg_return": summary["avg_return"],
                "avg_steps": summary["avg_steps"],
                "avg_comm_tokens": summary["avg_comm_tokens"],
            })


def _log_wandb(
    cfg: MAPPODecodingSweepConfig,
    run_dir: Path,
    json_path: Path,
    csv_path: Path,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not cfg.wandb:
        return {"enabled": False}
    _prepare_wandb_dirs(run_dir, cfg.wandb_mode)
    try:
        import wandb
    except Exception as exc:
        return {"enabled": False, "error": f"wandb unavailable: {exc}"}

    run = None
    try:
        run = wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_run or run_dir.name,
            mode=cfg.wandb_mode,
            config=asdict(cfg),
            dir=str(run_dir / "wandb"),
        )
        _define_wandb_metrics(run)
        table = wandb.Table(columns=[
            "rank",
            "checkpoint_label",
            "success_rate",
            "avg_return",
            "avg_steps",
            "avg_comm_tokens",
            "action_mode",
            "send_mode",
            "send_threshold",
            "token_mode",
            "length_mode",
        ])
        for row in rows:
            summary = row["summary"]
            table.add_data(
                row["rank"],
                row["checkpoint_label"],
                summary["success_rate"],
                summary["avg_return"],
                summary["avg_steps"],
                summary["avg_comm_tokens"],
                row["action_mode"],
                row["send_mode"],
                row["send_threshold"],
                row["token_mode"],
                row["length_mode"],
            )
            run.log({
                "decode/rank": int(row["rank"]),
                "decode/success_rate": float(summary["success_rate"]),
                "decode/avg_return": float(summary["avg_return"]),
                "decode/avg_steps": float(summary["avg_steps"]),
                "decode/avg_comm_tokens": float(summary["avg_comm_tokens"]),
                "decode/send_threshold": float(row["send_threshold"]),
            })
        payload: dict[str, Any] = {"decoding/table": table}
        if rows:
            best = rows[0]
            best_summary = best["summary"]
            payload.update({
                "best/success_rate": float(best_summary["success_rate"]),
                "best/avg_return": float(best_summary["avg_return"]),
                "best/avg_steps": float(best_summary["avg_steps"]),
                "best/avg_comm_tokens": float(best_summary["avg_comm_tokens"]),
                "best/send_threshold": float(best["send_threshold"]),
            })
            run.summary["best_checkpoint_label"] = best["checkpoint_label"]
            run.summary["best_action_mode"] = best["action_mode"]
            run.summary["best_send_mode"] = best["send_mode"]
            run.summary["best_token_mode"] = best["token_mode"]
            run.summary["best_length_mode"] = best["length_mode"]
        run.log(payload)
        run.summary["summary_path"] = str(json_path)
        run.summary["csv_path"] = str(csv_path)
        if cfg.wandb_mode != "disabled":
            artifact = wandb.Artifact(run_dir.name, type="mappo_decoding_sweep")
            artifact.add_file(str(json_path))
            artifact.add_file(str(csv_path))
            run.log_artifact(artifact)
        return {
            "enabled": True,
            "mode": cfg.wandb_mode,
            "run_name": cfg.wandb_run or run_dir.name,
            "url": getattr(run, "url", None),
        }
    except Exception as exc:
        return {"enabled": True, "mode": cfg.wandb_mode, "error": str(exc)}
    finally:
        if run is not None:
            run.finish()


def _define_wandb_metrics(run) -> None:
    try:
        run.define_metric("decode/rank")
        run.define_metric("decode/*", step_metric="decode/rank")
    except Exception:
        pass


def _prepare_wandb_dirs(run_dir: Path, mode: str) -> None:
    os.environ["WANDB_MODE"] = mode
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


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
