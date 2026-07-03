from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

from syncorsink.envs import SyncOrSinkConfig, SyncOrSinkEnv
from syncorsink.eval.runner import run_episodes
from syncorsink.train.mappo import MAPPOConfig, load_mappo_checkpoint_policy, train_mappo


@dataclass
class TrainEvalWorkbenchConfig:
    algorithm: str = "mappo"
    scenario: str = "signal_hunt"
    map_size: int = 8
    agents: int = 2
    fov_preset: str = "easy"
    max_steps: int = 40
    energy_preset: str = "easy"
    comm: bool = True
    comm_token_limit: int = 4
    comm_vocab_size: int = 8
    comm_max_messages: int = 4
    critic_mode: str = "central"
    shared_actor: bool = False
    hidden_dim: int = 32
    updates: int = 2
    rollout_steps: int = 16
    epochs: int = 1
    minibatch: int = 16
    lr: float = 3e-4
    device: str = "cpu"
    seed: int = 0
    eval_episodes: int = 2
    eval_seed: int = 1000
    output_dir: str = "logs/workbench"
    run_name: str | None = None
    wandb: bool = False
    wandb_project: str = "syncorsink-workbench"
    wandb_run: str | None = None
    wandb_mode: str = "offline"


def run_train_eval_workbench(cfg: TrainEvalWorkbenchConfig) -> dict[str, Any]:
    if cfg.algorithm != "mappo":
        raise ValueError(f"Unsupported workbench algorithm: {cfg.algorithm}")

    run_dir = _make_run_dir(cfg)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "mappo.pt"
    summary_path = run_dir / "summary.json"

    train_cfg = MAPPOConfig(
        scenario=cfg.scenario,
        map_size=cfg.map_size,
        agents=cfg.agents,
        fov_preset=cfg.fov_preset,
        max_steps=cfg.max_steps,
        energy_preset=cfg.energy_preset,
        comm=cfg.comm,
        comm_token_limit=cfg.comm_token_limit,
        comm_vocab_size=cfg.comm_vocab_size,
        comm_max_messages=cfg.comm_max_messages,
        critic_mode=cfg.critic_mode,
        shared_actor=cfg.shared_actor,
        hidden_dim=cfg.hidden_dim,
        updates=cfg.updates,
        rollout_steps=cfg.rollout_steps,
        epochs=cfg.epochs,
        minibatch=cfg.minibatch,
        lr=cfg.lr,
        device=cfg.device,
        seed=cfg.seed,
        eval_every=0,
        eval_episodes=0,
        save=str(checkpoint_path),
        save_every=max(1, cfg.updates),
    )
    train_mappo(train_cfg)

    env_config = SyncOrSinkConfig(
        scenario=cfg.scenario,
        map_size=cfg.map_size,
        num_agents=cfg.agents,
        fov_preset=cfg.fov_preset,
        max_steps=cfg.max_steps,
        energy_preset=cfg.energy_preset,
        comm_token_limit=cfg.comm_token_limit,
        token_vocab_size=cfg.comm_vocab_size,
        max_messages=cfg.comm_max_messages,
    )
    env = SyncOrSinkEnv(env_config)
    policy = load_mappo_checkpoint_policy(
        checkpoint_path,
        env,
        cfg=train_cfg,
        deterministic=True,
        device=cfg.device,
        sample_seed=cfg.eval_seed,
    )
    summary, episodes = run_episodes(env, policy, episodes=cfg.eval_episodes, seed=cfg.eval_seed)

    result: dict[str, Any] = {
        "status": "complete",
        "run_dir": str(run_dir),
        "checkpoint_path": str(checkpoint_path),
        "summary_path": str(summary_path),
        "workbench_config": asdict(cfg),
        "train_config": asdict(train_cfg),
        "eval": asdict(summary),
        "episodes": [asdict(ep) for ep in episodes],
    }
    _write_json(summary_path, result)
    result["wandb"] = _log_wandb(cfg, result, files=[checkpoint_path, summary_path])
    _write_json(summary_path, result)
    return result


def _make_run_dir(cfg: TrainEvalWorkbenchConfig) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    name = cfg.run_name or f"{cfg.algorithm}_{cfg.scenario}_{cfg.map_size}x{cfg.map_size}_seed{cfg.seed}_{stamp}"
    run_dir = Path(cfg.output_dir) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _log_wandb(cfg: TrainEvalWorkbenchConfig, result: dict[str, Any], files: list[Path]) -> dict[str, Any]:
    if not cfg.wandb:
        return {"enabled": False}
    wandb_dir = Path(result["run_dir"]) / "wandb"
    data_dir = wandb_dir / "data"
    artifact_dir = wandb_dir / "artifacts"
    cache_dir = wandb_dir / "cache"
    config_dir = wandb_dir / "config"
    for directory in (data_dir, artifact_dir, cache_dir, config_dir):
        directory.mkdir(parents=True, exist_ok=True)
    os.environ["WANDB_DIR"] = str(wandb_dir)
    os.environ["WANDB_DATA_DIR"] = str(data_dir)
    os.environ["WANDB_ARTIFACT_DIR"] = str(artifact_dir)
    os.environ["WANDB_CACHE_DIR"] = str(cache_dir)
    os.environ["WANDB_CONFIG_DIR"] = str(config_dir)
    try:
        import wandb
    except Exception as exc:
        return {"enabled": False, "error": f"wandb unavailable: {exc}"}

    run = None
    try:
        run = wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_run or Path(result["run_dir"]).name,
            mode=cfg.wandb_mode,
            config=result["workbench_config"],
            dir=str(wandb_dir),
        )
        run.log(
            {
                "eval/success_rate": float(result["eval"]["success_rate"]),
                "eval/avg_return": float(result["eval"]["avg_return"]),
                "eval/avg_steps": float(result["eval"]["avg_steps"]),
                "eval/avg_comm_tokens": float(result["eval"]["avg_comm_tokens"]),
                "train/updates": int(cfg.updates),
                "train/rollout_steps": int(cfg.rollout_steps),
            }
        )
        if cfg.wandb_mode != "disabled":
            artifact = wandb.Artifact(Path(result["run_dir"]).name, type="train_eval_workbench")
            for file_path in files:
                artifact.add_file(str(file_path))
            run.log_artifact(artifact)
        info = {
            "enabled": True,
            "mode": cfg.wandb_mode,
            "run_id": getattr(run, "id", None),
            "run_name": getattr(run, "name", None),
            "run_path": getattr(run, "path", None),
            "local_dir": str(wandb_dir),
        }
        return info
    except Exception as exc:
        return {"enabled": False, "error": str(exc)}
    finally:
        if run is not None:
            run.finish()
