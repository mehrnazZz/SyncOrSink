from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from json import JSONDecodeError
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ScenarioCase:
    scenario: str
    map_size: int
    agents: int
    fov_preset: str
    max_steps: int
    energy_preset: str | None = None


@dataclass
class RunRecord:
    algorithm: str
    scenario: str
    seed: int
    run_dir: str
    checkpoint_path: str
    command: list[str]
    status: str
    returncode: int | None
    elapsed_sec: float
    checkpoint_exists: bool
    stdout_path: str
    stderr_path: str
    stdout_tail: list[str]
    stderr_tail: list[str]
    eval_metrics: dict[str, float] | None
    final_eval_metrics: dict[str, float] | None
    best_eval_metrics: dict[str, float] | None
    best_checkpoint_path: str | None
    best_checkpoint_exists: bool
    wandb: dict


DEFAULT_CASES: dict[str, ScenarioCase] = {
    "signal_hunt": ScenarioCase(
        scenario="signal_hunt",
        map_size=8,
        agents=2,
        fov_preset="easy",
        max_steps=60,
    ),
    "energy_grid": ScenarioCase(
        scenario="energy_grid",
        map_size=8,
        agents=3,
        fov_preset="easy",
        max_steps=80,
        energy_preset="easy",
    ),
    "pipeline_assembly": ScenarioCase(
        scenario="pipeline_assembly",
        map_size=8,
        agents=3,
        fov_preset="easy",
        max_steps=80,
    ),
}

TRAIN_SCRIPTS = {
    "mappo": "examples/mappo_train.py",
    "comm_mat": "examples/comm_mat_train.py",
    "tarmac": "examples/tarmac_train.py",
    "recurrent_bc_rl": "examples/recurrent_train.py",
}

SCENARIO_SHAPING_ARGS = {
    "signal_hunt": [
        "--signal-shaping",
        "--signal-shaping-scale",
        "0.05",
        "--signal-scan-bonus",
        "0.05",
        "--signal-joint-scan-bonus",
        "1.0",
        "--signal-colocation-bonus",
        "0.25",
        "--signal-comm-utility",
        "0.05",
    ],
    "energy_grid": [
        "--energy-shaping",
        "--energy-shaping-scale",
        "0.05",
    ],
    "pipeline_assembly": [
        "--pipeline-shaping",
        "--pipeline-shaping-scale",
        "0.05",
    ],
}


RECURRENT_AUTO_ORACLES = {
    "signal_hunt": "signal_hint_comm",
    "energy_grid": "oracle_strong_comm",
    "pipeline_assembly": "oracle_strong_comm",
}


RECURRENT_PPO_PROFILES = {
    "standard": {
        "recurrent_rl_lr": 3e-5,
        "recurrent_clip": 0.2,
        "recurrent_entropy_coeff": 0.01,
        "recurrent_max_grad_norm": 0.5,
        "recurrent_bc_kl_coeff": 0.5,
        "recurrent_bc_comm_kl_coeff": 0.5,
        "recurrent_rl_balanced_rollouts": False,
        "recurrent_rl_rollout_eval_decoding": False,
    },
    "guarded": {
        "recurrent_rl_lr": 1e-5,
        "recurrent_clip": 0.1,
        "recurrent_entropy_coeff": 0.0,
        "recurrent_max_grad_norm": 0.25,
        "recurrent_bc_kl_coeff": 2.0,
        "recurrent_bc_comm_kl_coeff": 2.0,
        "recurrent_rl_balanced_rollouts": True,
        "recurrent_rl_rollout_eval_decoding": True,
    },
}


def build_command(
    *,
    algorithm: str,
    case: ScenarioCase,
    checkpoint_path: Path,
    args,
    run_name: str,
    seed: int,
) -> list[str]:
    if algorithm not in TRAIN_SCRIPTS:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    if algorithm == "recurrent_bc_rl":
        return _build_recurrent_command(
            case=case,
            checkpoint_path=checkpoint_path,
            args=args,
            run_name=run_name,
            seed=seed,
        )

    cmd = [
        sys.executable,
        "-u",
        str(ROOT / TRAIN_SCRIPTS[algorithm]),
        "--scenario",
        case.scenario,
        "--map-size",
        str(case.map_size),
        "--agents",
        str(case.agents),
        "--fov-preset",
        case.fov_preset,
        "--max-steps",
        str(case.max_steps),
        "--updates",
        str(args.updates),
        "--rollout-steps",
        str(args.rollout_steps),
        "--epochs",
        str(args.epochs),
        "--minibatch",
        str(args.minibatch),
        "--device",
        args.device,
        "--seed",
        str(seed),
        "--eval-every",
        str(args.eval_every),
        "--eval-episodes",
        str(args.eval_episodes),
        "--save",
        str(checkpoint_path),
        "--save-every",
        str(max(1, args.updates)),
        "--wandb-project",
        args.wandb_project,
        "--wandb-run",
        run_name,
    ]
    if case.energy_preset is not None:
        cmd.extend(["--energy-preset", case.energy_preset])
    if algorithm == "mappo":
        cmd.extend([
            "--comm",
            "--critic-mode",
            args.mappo_critic_mode,
            "--backbone",
            args.mappo_backbone,
            "--eval-action-mode",
            args.mappo_eval_action_mode,
            "--eval-action-temperature",
            str(args.mappo_eval_action_temperature),
            "--eval-send-mode",
            args.mappo_eval_send_mode,
            "--eval-send-threshold",
            str(args.mappo_eval_send_threshold),
            "--eval-token-mode",
            args.mappo_eval_token_mode,
            "--eval-token-temperature",
            str(args.mappo_eval_token_temperature),
            "--eval-length-mode",
            args.mappo_eval_length_mode,
            "--eval-length-temperature",
            str(args.mappo_eval_length_temperature),
        ])
        if args.mappo_shared_actor:
            cmd.append("--shared-actor")
        if args.mappo_obs_exploration_memory:
            cmd.append("--obs-exploration-memory")
        if args.mappo_obs_exploration_age:
            cmd.append("--obs-exploration-age")
    cmd.extend(_learning_profile_args(args.learning_profile, algorithm, case))
    if args.wandb:
        cmd.append("--wandb")
    return cmd


def _build_recurrent_command(
    *,
    case: ScenarioCase,
    checkpoint_path: Path,
    args,
    run_name: str,
    seed: int,
) -> list[str]:
    rl_updates = args.recurrent_rl_updates
    if rl_updates is None:
        rl_updates = args.updates

    cmd = [
        sys.executable,
        "-u",
        str(ROOT / TRAIN_SCRIPTS["recurrent_bc_rl"]),
        "--scenario",
        case.scenario,
        "--map-size",
        str(case.map_size),
        "--agents",
        str(case.agents),
        "--fov-preset",
        case.fov_preset,
        "--max-steps",
        str(case.max_steps),
        "--oracle",
        _resolve_recurrent_oracle(args, case),
        "--demo-episodes",
        str(args.recurrent_demo_episodes),
        "--bc-epochs",
        str(args.recurrent_bc_epochs),
        "--bc-lr",
        str(args.recurrent_bc_lr),
        "--bc-seq-len",
        str(args.recurrent_bc_seq_len),
        "--dagger-rounds",
        str(args.recurrent_dagger_rounds),
        "--dagger-episodes",
        str(args.recurrent_dagger_episodes),
        "--rl-updates",
        str(rl_updates),
        "--rollout-steps",
        str(args.rollout_steps),
        "--rl-epochs",
        str(args.recurrent_rl_epochs),
        "--minibatch-seqs",
        str(args.recurrent_minibatch_seqs),
        "--rl-lr",
        str(args.recurrent_rl_lr),
        "--clip",
        str(args.recurrent_clip),
        "--entropy-coeff",
        str(args.recurrent_entropy_coeff),
        "--max-grad-norm",
        str(args.recurrent_max_grad_norm),
        "--bc-kl-coeff",
        str(args.recurrent_bc_kl_coeff),
        "--bc-comm-kl-coeff",
        str(args.recurrent_bc_comm_kl_coeff),
        "--rl-eval-every",
        str(args.eval_every),
        "--rl-eval-episodes",
        str(args.eval_episodes),
        "--eval-episodes",
        str(args.eval_episodes),
        "--eval-seed-count",
        str(args.recurrent_eval_seed_count),
        "--eval-send-threshold",
        str(args.recurrent_eval_send_threshold),
        "--hidden-dim",
        str(args.recurrent_hidden_dim),
        "--comm-token-limit",
        str(args.recurrent_comm_token_limit),
        "--comm-vocab-size",
        str(args.recurrent_comm_vocab_size),
        "--device",
        args.device,
        "--seed",
        str(seed),
        "--save",
        str(checkpoint_path),
        "--wandb-project",
        args.wandb_project,
        "--wandb-run",
        run_name,
    ]
    if case.energy_preset is not None:
        cmd.extend(["--energy-preset", case.energy_preset])
    if args.recurrent_train_map_sizes:
        cmd.extend(["--train-map-sizes", args.recurrent_train_map_sizes])
    if args.recurrent_train_map_sampling_weights:
        cmd.extend(["--train-map-sampling-weights", args.recurrent_train_map_sampling_weights])
    if args.recurrent_map_max_steps:
        cmd.extend(["--map-max-steps", args.recurrent_map_max_steps])
    if args.recurrent_eval_map_sizes:
        cmd.extend(["--eval-map-sizes", args.recurrent_eval_map_sizes])
    if args.recurrent_eval_seed_list:
        cmd.extend(["--eval-seed-list", args.recurrent_eval_seed_list])
    if args.recurrent_dagger_seed_list:
        cmd.extend(["--dagger-seed-list", args.recurrent_dagger_seed_list])
    recurrent_init = _resolve_recurrent_init(
        args,
        case=case,
        seed=seed,
        run_name=run_name,
    )
    if recurrent_init:
        cmd.extend(["--recurrent-init", recurrent_init])
    if args.recurrent_init_for_dagger:
        cmd.append("--recurrent-init-for-dagger")
    if args.recurrent_init_allow_obs_dim_mismatch:
        cmd.append("--recurrent-init-allow-obs-dim-mismatch")
    if args.recurrent_rl_balanced_rollouts:
        cmd.append("--rl-balanced-rollouts")
    if args.recurrent_rl_rollout_eval_decoding:
        cmd.append("--rl-rollout-eval-decoding")
    if not args.recurrent_rl_restore_best:
        cmd.append("--no-rl-restore-best")
    if not args.recurrent_rl_save_best:
        cmd.append("--no-rl-save-best")
    if args.recurrent_comm:
        cmd.append("--comm")
    if args.recurrent_calibrate_send_threshold:
        cmd.append("--bc-calibrate-send-threshold")
    if args.recurrent_signal_preset == "specialist" and case.scenario == "signal_hunt":
        cmd.extend([
            "--obs-exploration-memory",
            "--obs-feedback",
            "--obs-normalize-tokens",
            "--obs-memory-mode",
            args.recurrent_obs_memory_mode,
            "--obs-memory-radius",
            str(args.recurrent_obs_memory_radius),
            "--obs-navigation-features",
            "--obs-signal-features",
            "--obs-signal-negative-memory",
            "--obs-signal-inferred-target-features",
            "--obs-signal-target-match-features",
            "--obs-signal-sync-feedback",
            "--obs-signal-scan-state",
            "--eval-signal-scan-sync-assist",
            "--eval-signal-scan-broadcast-assist",
            "--eval-signal-exact-target-message-guard",
            "--eval-signal-exact-target-navigation-assist",
            "--eval-signal-exact-target-memory-steps",
            str(args.recurrent_eval_signal_exact_target_memory_steps),
        ])
    cmd.extend(_learning_profile_args(args.learning_profile, "recurrent_bc_rl", case))
    if args.wandb:
        cmd.append("--wandb")
    return cmd


def _learning_profile_args(profile: str, algorithm: str, case: ScenarioCase) -> list[str]:
    if profile == "bare":
        return []

    cmd = list(SCENARIO_SHAPING_ARGS.get(case.scenario, []))
    if profile == "shaped":
        return cmd
    if profile != "comm_curriculum":
        raise ValueError(f"Unknown learning profile: {profile}")

    if algorithm in {"mappo", "comm_mat"}:
        cmd.extend([
            "--comm-cost",
            "0.0",
            "--comm-send-target",
            "0.25",
            "--comm-send-target-coeff",
            "0.05",
        ])
    elif algorithm == "tarmac":
        cmd.extend(["--attn-entropy-coeff", "0.01"])
    return cmd


def _resolve_recurrent_init(args, *, case: ScenarioCase, seed: int, run_name: str) -> str:
    if args.recurrent_init_template:
        return args.recurrent_init_template.format(
            seed=seed,
            scenario=case.scenario,
            map_size=case.map_size,
            agents=case.agents,
            algorithm="recurrent_bc_rl",
            run_name=run_name,
        )
    return args.recurrent_init or ""


def _resolve_recurrent_oracle(args, case: ScenarioCase) -> str:
    if args.recurrent_oracle == "auto":
        return RECURRENT_AUTO_ORACLES[case.scenario]
    return args.recurrent_oracle


def _apply_recurrent_ppo_profile(args):
    profile = RECURRENT_PPO_PROFILES[args.recurrent_ppo_profile]
    for key, value in profile.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    return args


def run_suite(args) -> dict:
    suite_dir = _suite_dir(args)
    suite_dir.mkdir(parents=True, exist_ok=True)
    cases = [DEFAULT_CASES[name] for name in args.scenarios]
    records: list[RunRecord] = []

    for algorithm in args.algorithms:
        for case in cases:
            for seed in args.seeds:
                run_name = f"{algorithm}_{case.scenario}_{case.map_size}x{case.map_size}_seed{seed}"
                run_dir = suite_dir / run_name
                checkpoint_dir = run_dir / "checkpoints"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_path = checkpoint_dir / f"{algorithm}.pt"
                stdout_path = run_dir / "stdout.log"
                stderr_path = run_dir / "stderr.log"
                cmd = build_command(
                    algorithm=algorithm,
                    case=case,
                    checkpoint_path=checkpoint_path,
                    args=args,
                    run_name=run_name,
                    seed=seed,
                )
                record = _run_one(
                    cmd=cmd,
                    algorithm=algorithm,
                    case=case,
                    seed=seed,
                    run_dir=run_dir,
                    checkpoint_path=checkpoint_path,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                    dry_run=args.dry_run,
                    wandb_mode=args.wandb_mode,
                    strict_wandb=args.strict_wandb,
                )
                records.append(record)
                _write_json(run_dir / "run_summary.json", asdict(record))
                print(_format_record(record), flush=True)
                if args.fail_fast and record.status == "failed":
                    break
            if args.fail_fast and records and records[-1].status == "failed":
                break
        if args.fail_fast and records and records[-1].status == "failed":
            break

    payload = {
        "suite": "core_training_sweep",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "suite_dir": str(suite_dir),
        "dry_run": bool(args.dry_run),
        "config": {
            "algorithms": args.algorithms,
            "scenarios": args.scenarios,
            "updates": args.updates,
            "rollout_steps": args.rollout_steps,
            "epochs": args.epochs,
            "minibatch": args.minibatch,
            "eval_every": args.eval_every,
            "eval_episodes": args.eval_episodes,
            "device": args.device,
            "seeds": args.seeds,
            "learning_profile": args.learning_profile,
            "mappo_backbone": args.mappo_backbone,
            "wandb": args.wandb,
            "wandb_mode": args.wandb_mode,
            "wandb_project": args.wandb_project,
            "strict_wandb": args.strict_wandb,
            "mappo_critic_mode": args.mappo_critic_mode,
            "mappo_shared_actor": args.mappo_shared_actor,
            "mappo_obs_exploration_memory": args.mappo_obs_exploration_memory,
            "mappo_obs_exploration_age": args.mappo_obs_exploration_age,
            "mappo_eval_action_mode": args.mappo_eval_action_mode,
            "mappo_eval_action_temperature": args.mappo_eval_action_temperature,
            "mappo_eval_send_mode": args.mappo_eval_send_mode,
            "mappo_eval_send_threshold": args.mappo_eval_send_threshold,
            "mappo_eval_token_mode": args.mappo_eval_token_mode,
            "mappo_eval_token_temperature": args.mappo_eval_token_temperature,
            "mappo_eval_length_mode": args.mappo_eval_length_mode,
            "mappo_eval_length_temperature": args.mappo_eval_length_temperature,
            "recurrent_oracle": args.recurrent_oracle,
            "recurrent_resolved_oracles": {
                case.scenario: _resolve_recurrent_oracle(args, case)
                for case in cases
            },
            "recurrent_signal_preset": args.recurrent_signal_preset,
            "recurrent_demo_episodes": args.recurrent_demo_episodes,
            "recurrent_bc_epochs": args.recurrent_bc_epochs,
            "recurrent_bc_lr": args.recurrent_bc_lr,
            "recurrent_bc_seq_len": args.recurrent_bc_seq_len,
            "recurrent_dagger_rounds": args.recurrent_dagger_rounds,
            "recurrent_dagger_episodes": args.recurrent_dagger_episodes,
            "recurrent_rl_updates": args.recurrent_rl_updates,
            "recurrent_ppo_profile": args.recurrent_ppo_profile,
            "recurrent_rl_epochs": args.recurrent_rl_epochs,
            "recurrent_minibatch_seqs": args.recurrent_minibatch_seqs,
            "recurrent_rl_lr": args.recurrent_rl_lr,
            "recurrent_clip": args.recurrent_clip,
            "recurrent_entropy_coeff": args.recurrent_entropy_coeff,
            "recurrent_max_grad_norm": args.recurrent_max_grad_norm,
            "recurrent_bc_kl_coeff": args.recurrent_bc_kl_coeff,
            "recurrent_bc_comm_kl_coeff": args.recurrent_bc_comm_kl_coeff,
            "recurrent_rl_balanced_rollouts": args.recurrent_rl_balanced_rollouts,
            "recurrent_rl_rollout_eval_decoding": args.recurrent_rl_rollout_eval_decoding,
            "recurrent_rl_restore_best": args.recurrent_rl_restore_best,
            "recurrent_rl_save_best": args.recurrent_rl_save_best,
            "recurrent_train_map_sizes": args.recurrent_train_map_sizes,
            "recurrent_train_map_sampling_weights": args.recurrent_train_map_sampling_weights,
            "recurrent_map_max_steps": args.recurrent_map_max_steps,
            "recurrent_eval_map_sizes": args.recurrent_eval_map_sizes,
            "recurrent_eval_seed_count": args.recurrent_eval_seed_count,
            "recurrent_eval_seed_list": args.recurrent_eval_seed_list,
            "recurrent_dagger_seed_list": args.recurrent_dagger_seed_list,
            "recurrent_init": args.recurrent_init,
            "recurrent_init_template": args.recurrent_init_template,
            "recurrent_init_for_dagger": args.recurrent_init_for_dagger,
            "recurrent_init_allow_obs_dim_mismatch": args.recurrent_init_allow_obs_dim_mismatch,
            "recurrent_comm": args.recurrent_comm,
            "recurrent_comm_token_limit": args.recurrent_comm_token_limit,
            "recurrent_comm_vocab_size": args.recurrent_comm_vocab_size,
            "recurrent_hidden_dim": args.recurrent_hidden_dim,
            "recurrent_eval_send_threshold": args.recurrent_eval_send_threshold,
            "recurrent_calibrate_send_threshold": args.recurrent_calibrate_send_threshold,
            "recurrent_obs_memory_mode": args.recurrent_obs_memory_mode,
            "recurrent_obs_memory_radius": args.recurrent_obs_memory_radius,
            "recurrent_eval_signal_exact_target_memory_steps": (
                args.recurrent_eval_signal_exact_target_memory_steps
            ),
        },
        "cases": [asdict(case) for case in cases],
        "runs": [asdict(record) for record in records],
        "aggregate": _aggregate_records(records),
        "overall": {
            "total": len(records),
            "complete": sum(record.status == "complete" for record in records),
            "failed": sum(record.status == "failed" for record in records),
            "dry_run": sum(record.status == "dry_run" for record in records),
        },
    }
    _write_json(suite_dir / "suite_summary.json", payload)
    if args.output_json:
        _write_json(Path(args.output_json), payload)
    if payload["overall"]["failed"] > 0:
        return payload
    return payload


def _run_one(
    *,
    cmd: list[str],
    algorithm: str,
    case: ScenarioCase,
    seed: int,
    run_dir: Path,
    checkpoint_path: Path,
    stdout_path: Path,
    stderr_path: Path,
    dry_run: bool,
    wandb_mode: str,
    strict_wandb: bool,
) -> RunRecord:
    start = time.time()
    wandb_requested = "--wandb" in cmd
    best_checkpoint_path = _best_checkpoint_path(algorithm, checkpoint_path)
    if dry_run:
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return RunRecord(
            algorithm=algorithm,
            scenario=case.scenario,
            seed=seed,
            run_dir=str(run_dir),
            checkpoint_path=str(checkpoint_path),
            command=cmd,
            status="dry_run",
            returncode=None,
            elapsed_sec=0.0,
            checkpoint_exists=checkpoint_path.exists(),
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            stdout_tail=[],
            stderr_tail=[],
            eval_metrics=None,
            final_eval_metrics=None,
            best_eval_metrics=None,
            best_checkpoint_path=str(best_checkpoint_path) if best_checkpoint_path is not None else None,
            best_checkpoint_exists=best_checkpoint_path.exists() if best_checkpoint_path is not None else False,
            wandb=_wandb_record(requested=wandb_requested, mode=wandb_mode, status="dry_run", error_lines=[]),
        )

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["WANDB_MODE"] = wandb_mode
    env.setdefault("WANDB_SILENT", "true")
    wandb_dir = _prepare_wandb_dirs(run_dir)
    env["WANDB_DIR"] = str(wandb_dir)
    env["WANDB_DATA_DIR"] = str(wandb_dir / "data")
    env["WANDB_ARTIFACT_DIR"] = str(wandb_dir / "artifacts")
    env["WANDB_CACHE_DIR"] = str(wandb_dir / "cache")
    env["WANDB_CONFIG_DIR"] = str(wandb_dir / "config")
    env["TMPDIR"] = str(wandb_dir / "tmp")

    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=stdout, stderr=stderr, check=False)

    stdout_tail = _tail_lines(stdout_path)
    stderr_tail = _tail_lines(stderr_path)
    wandb = _parse_wandb_record(
        stdout_path,
        stderr_path,
        requested=wandb_requested,
        mode=wandb_mode,
        run_dir=run_dir,
    )
    status = "complete" if proc.returncode == 0 and checkpoint_path.exists() else "failed"
    if strict_wandb and wandb.get("status") == "failed":
        status = "failed"
    eval_metrics = _parse_eval_metrics(
        algorithm,
        stdout_path,
        stdout_tail,
        checkpoint_path=checkpoint_path,
    )
    final_eval_metrics = None
    best_eval_metrics = None
    if algorithm == "recurrent_bc_rl":
        recurrent_evals = _parse_recurrent_checkpoint_evals(checkpoint_path)
        final_eval_metrics = recurrent_evals.get("final_eval")
        best_eval_metrics = recurrent_evals.get("best_eval")
    return RunRecord(
        algorithm=algorithm,
        scenario=case.scenario,
        seed=seed,
        run_dir=str(run_dir),
        checkpoint_path=str(checkpoint_path),
        command=cmd,
        status=status,
        returncode=proc.returncode,
        elapsed_sec=round(time.time() - start, 3),
        checkpoint_exists=checkpoint_path.exists(),
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
        eval_metrics=eval_metrics,
        final_eval_metrics=final_eval_metrics,
        best_eval_metrics=best_eval_metrics,
        best_checkpoint_path=str(best_checkpoint_path) if best_checkpoint_path is not None else None,
        best_checkpoint_exists=best_checkpoint_path.exists() if best_checkpoint_path is not None else False,
        wandb=wandb,
    )


def _parse_eval_metrics(
    algorithm: str,
    stdout_path: Path,
    stdout_tail: Iterable[str],
    *,
    checkpoint_path: Path | None = None,
) -> dict[str, float] | None:
    if algorithm == "recurrent_bc_rl":
        checkpoint_metrics = _parse_recurrent_checkpoint_eval(checkpoint_path)
        if checkpoint_metrics is not None:
            return checkpoint_metrics
        stdout = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.exists() else ""
        return _parse_recurrent_stdout_eval(stdout)
    return _parse_last_eval(stdout_tail)


def _parse_last_eval(lines: Iterable[str]) -> dict[str, float] | None:
    for line in reversed(list(lines)):
        if "eval |" not in line:
            continue
        metrics: dict[str, float] = {}
        for part in line.split("|"):
            part = part.strip()
            if part.startswith("ret "):
                metrics["return"] = float(part.split()[1])
            elif part.startswith("steps "):
                metrics["steps"] = float(part.split()[1])
            elif part.startswith("success "):
                metrics["success_rate"] = float(part.split()[1])
        return metrics or None
    return None


def _parse_recurrent_checkpoint_eval(checkpoint_path: Path | None) -> dict[str, float] | None:
    evals, restored_best = _parse_recurrent_checkpoint_eval_data(checkpoint_path)
    if restored_best:
        metrics = evals.get("best_eval")
        if metrics is not None:
            return metrics
    for key in ("eval_recurrent_policy", "final_eval", "best_eval"):
        metrics = evals.get(key)
        if metrics is not None:
            return metrics
    return None


def _parse_recurrent_checkpoint_evals(checkpoint_path: Path | None) -> dict[str, dict[str, float]]:
    evals, _ = _parse_recurrent_checkpoint_eval_data(checkpoint_path)
    return evals


def _parse_recurrent_checkpoint_eval_data(
    checkpoint_path: Path | None,
) -> tuple[dict[str, dict[str, float]], bool]:
    if checkpoint_path is None or not checkpoint_path.exists():
        return {}, False
    try:
        import torch

        ckpt = torch.load(checkpoint_path, map_location="cpu")
    except Exception:
        return {}, False
    if not isinstance(ckpt, dict):
        return {}, False
    evals: dict[str, dict[str, float]] = {}
    for key in ("eval_recurrent_policy", "final_eval", "best_eval"):
        metrics = _metrics_from_recurrent_eval(ckpt.get(key))
        if metrics is not None:
            evals[key] = metrics
    return evals, bool(ckpt.get("restored_best", False))


def _parse_recurrent_stdout_eval(stdout: str) -> dict[str, float] | None:
    latest: dict[str, float] | None = None
    for obj in _iter_json_objects(stdout):
        if not isinstance(obj, dict):
            continue
        for key in ("eval_recurrent_bc", "eval_recurrent_init"):
            metrics = _metrics_from_recurrent_eval(obj.get(key))
            if metrics is not None:
                latest = metrics
        for key in ("recurrent_dagger", "recurrent_dagger_initial"):
            row = obj.get(key)
            if isinstance(row, dict):
                metrics = _metrics_from_recurrent_eval(row.get("eval"))
                if metrics is not None:
                    latest = metrics
        metrics = _metrics_from_recurrent_eval(obj.get("recurrent_rl_eval"))
        if metrics is not None:
            latest = metrics
    return latest


def _iter_json_objects(text: str):
    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(text):
        start = text.find("{", idx)
        if start < 0:
            return
        try:
            obj, end = decoder.raw_decode(text[start:])
        except JSONDecodeError:
            idx = start + 1
            continue
        yield obj
        idx = start + end


def _metrics_from_recurrent_eval(eval_result) -> dict[str, float] | None:
    if not isinstance(eval_result, dict):
        return None
    try:
        return {
            "success_rate": float(eval_result["success_rate"]),
            "return": float(eval_result["avg_return"]),
            "steps": float(eval_result["avg_steps"]),
        }
    except (KeyError, TypeError, ValueError):
        return None


def _aggregate_records(records: list[RunRecord]) -> list[dict]:
    groups: dict[tuple[str, str], list[RunRecord]] = {}
    for record in records:
        groups.setdefault((record.algorithm, record.scenario), []).append(record)

    aggregate = []
    for (algorithm, scenario), group in sorted(groups.items()):
        eval_records = [record.eval_metrics for record in group if record.eval_metrics is not None]
        final_eval_records = [
            record.final_eval_metrics for record in group if record.final_eval_metrics is not None
        ]
        best_eval_records = [
            record.best_eval_metrics for record in group if record.best_eval_metrics is not None
        ]
        aggregate.append(
            {
                "algorithm": algorithm,
                "scenario": scenario,
                "seeds": sorted(record.seed for record in group),
                "runs": len(group),
                "complete": sum(record.status == "complete" for record in group),
                "failed": sum(record.status == "failed" for record in group),
                "dry_run": sum(record.status == "dry_run" for record in group),
                "checkpoint_count": sum(record.checkpoint_exists for record in group),
                "wandb_requested": sum(bool(record.wandb.get("requested")) for record in group),
                "wandb_failed": sum(record.wandb.get("status") == "failed" for record in group),
                "mean_eval_success_rate": _mean_metric(eval_records, "success_rate"),
                "mean_eval_return": _mean_metric(eval_records, "return"),
                "mean_eval_steps": _mean_metric(eval_records, "steps"),
                "mean_final_eval_success_rate": _mean_metric(final_eval_records, "success_rate"),
                "mean_final_eval_return": _mean_metric(final_eval_records, "return"),
                "mean_final_eval_steps": _mean_metric(final_eval_records, "steps"),
                "mean_best_eval_success_rate": _mean_metric(best_eval_records, "success_rate"),
                "mean_best_eval_return": _mean_metric(best_eval_records, "return"),
                "mean_best_eval_steps": _mean_metric(best_eval_records, "steps"),
            }
        )
    return aggregate


def _mean_metric(metrics: list[dict[str, float]], key: str) -> float | None:
    values = [metric[key] for metric in metrics if key in metric]
    if not values:
        return None
    return float(sum(values) / len(values))


def _prepare_wandb_dirs(run_dir: Path) -> Path:
    wandb_dir = run_dir / "wandb"
    for name in ("data", "artifacts", "cache", "config", "tmp"):
        (wandb_dir / name).mkdir(parents=True, exist_ok=True)
    return wandb_dir


def _best_checkpoint_path(algorithm: str, checkpoint_path: Path) -> Path | None:
    if algorithm != "recurrent_bc_rl":
        return None
    return checkpoint_path.with_name(f"{checkpoint_path.stem}_best{checkpoint_path.suffix}")


def _parse_wandb_record(
    stdout_path: Path,
    stderr_path: Path,
    *,
    requested: bool,
    mode: str,
    run_dir: Path | None = None,
) -> dict:
    if not requested:
        return _wandb_record(requested=False, mode=mode, status="not_requested", error_lines=[])
    stdout = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.exists() else ""
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.exists() else ""
    lines = [line for line in [*stdout.splitlines(), *stderr.splitlines()] if "wandb" in line.lower()]
    failed = [
        line
        for line in lines
        if "wandb init failed" in line.lower()
        or "wandb-core exited" in line.lower()
        or "serve() returned error" in line.lower()
        or "wandb log failed, disabling wandb" in line.lower()
        or "wandb scalar retry failed" in line.lower()
    ]
    if failed:
        return _wandb_record(requested=True, mode=mode, status="failed", error_lines=failed[-5:])
    if mode == "disabled":
        status = "disabled"
    else:
        status = "initialized"
    return _wandb_record(
        requested=True,
        mode=mode,
        status=status,
        error_lines=[],
        **_find_wandb_run_info(run_dir),
    )


def _wandb_record(
    *,
    requested: bool,
    mode: str,
    status: str,
    error_lines: list[str],
    **extra,
) -> dict:
    record = {
        "requested": requested,
        "mode": mode,
        "status": status,
        "error_lines": error_lines,
    }
    record.update({key: value for key, value in extra.items() if value is not None})
    return record


def _find_wandb_run_info(run_dir: Path | None) -> dict[str, str]:
    if run_dir is None:
        return {}
    wandb_root = Path(run_dir) / "wandb" / "wandb"
    if not wandb_root.exists():
        return {}
    run_dirs = sorted(path for path in wandb_root.glob("run-*-*") if path.is_dir())
    if not run_dirs:
        return {}
    run_dir_path = run_dirs[-1]
    run_id = run_dir_path.name.rsplit("-", 1)[-1]
    info = {
        "run_id": run_id,
        "local_run_dir": str(run_dir_path),
    }
    debug_log = run_dir_path / "logs" / "debug.log"
    if debug_log.exists():
        text = debug_log.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"finishing run ([^\s/]+)/([^\s/]+)/([^\s]+)", text)
        if match:
            entity, project, parsed_run_id = match.groups()
            info["run_id"] = parsed_run_id
            info["run_path"] = f"{entity}/{project}/{parsed_run_id}"
            info["url"] = f"https://wandb.ai/{entity}/{project}/runs/{parsed_run_id}"
    return info


def _tail_lines(path: Path, limit: int = 20) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]


def _suite_dir(args) -> Path:
    if args.run_name:
        name = args.run_name
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        name = f"core_training_{stamp}"
    return Path(args.output_dir) / name


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _format_record(record: RunRecord) -> str:
    metrics = ""
    if record.eval_metrics:
        metrics = (
            f" eval_success={record.eval_metrics.get('success_rate', 0.0):.2f}"
            f" eval_return={record.eval_metrics.get('return', 0.0):.2f}"
        )
    if record.final_eval_metrics and record.final_eval_metrics != record.eval_metrics:
        metrics += f" final_success={record.final_eval_metrics.get('success_rate', 0.0):.2f}"
    if record.best_eval_metrics and record.best_eval_metrics != record.eval_metrics:
        metrics += f" best_success={record.best_eval_metrics.get('success_rate', 0.0):.2f}"
    return (
        f"{record.status:8s} {record.algorithm:8s} {record.scenario:18s} seed={record.seed:<3d} "
        f"elapsed={record.elapsed_sec:.1f}s ckpt={int(record.checkpoint_exists)}{metrics}"
    )


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Run core learned-policy training sweeps.")
    parser.add_argument("--algorithms", nargs="+", default=["mappo", "comm_mat", "tarmac"], choices=sorted(TRAIN_SCRIPTS))
    parser.add_argument("--scenarios", nargs="+", default=list(DEFAULT_CASES), choices=sorted(DEFAULT_CASES))
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--minibatch", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=3)
    parser.add_argument("--eval-episodes", type=int, default=2)
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=None, help="Single-seed alias for --seeds")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument(
        "--learning-profile",
        default="bare",
        choices=["bare", "shaped", "comm_curriculum"],
        help="Training aids to apply before benchmark evaluation; bare leaves trainers unchanged",
    )
    parser.add_argument("--mappo-critic-mode", default="central", choices=["local", "central"])
    parser.add_argument("--mappo-shared-actor", action="store_true")
    parser.add_argument("--mappo-backbone", default="mlp", choices=["mlp", "transformer"])
    parser.add_argument("--mappo-obs-exploration-memory", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mappo-obs-exploration-age", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mappo-eval-action-mode", default="argmax", choices=["argmax", "sample"])
    parser.add_argument("--mappo-eval-action-temperature", type=float, default=1.0)
    parser.add_argument("--mappo-eval-send-mode", default="threshold", choices=["threshold", "sample"])
    parser.add_argument("--mappo-eval-send-threshold", type=float, default=0.5)
    parser.add_argument("--mappo-eval-token-mode", default="argmax", choices=["argmax", "sample"])
    parser.add_argument("--mappo-eval-token-temperature", type=float, default=1.0)
    parser.add_argument("--mappo-eval-length-mode", default="argmax", choices=["argmax", "sample"])
    parser.add_argument("--mappo-eval-length-temperature", type=float, default=1.0)
    parser.add_argument(
        "--recurrent-oracle",
        default="auto",
        choices=["auto", "oracle_strong", "oracle_strong_comm", "signal_hint_comm"],
        help=(
            "Oracle used for recurrent demos. auto uses signal_hint_comm for Signal Hunt "
            "and oracle_strong_comm for Energy Grid/Pipeline."
        ),
    )
    parser.add_argument("--recurrent-signal-preset", default="specialist", choices=["minimal", "specialist"])
    parser.add_argument("--recurrent-demo-episodes", type=int, default=20)
    parser.add_argument("--recurrent-bc-epochs", type=int, default=1)
    parser.add_argument("--recurrent-bc-lr", type=float, default=1e-3)
    parser.add_argument("--recurrent-bc-seq-len", type=int, default=32)
    parser.add_argument("--recurrent-dagger-rounds", type=int, default=0)
    parser.add_argument("--recurrent-dagger-episodes", type=int, default=20)
    parser.add_argument(
        "--recurrent-rl-updates",
        type=int,
        default=None,
        help="Override recurrent PPO updates; defaults to the shared --updates value",
    )
    parser.add_argument(
        "--recurrent-ppo-profile",
        default="guarded",
        choices=sorted(RECURRENT_PPO_PROFILES),
        help="Named recurrent PPO tuning profile; explicit --recurrent-* PPO flags override it.",
    )
    parser.add_argument("--recurrent-rl-epochs", type=int, default=2)
    parser.add_argument("--recurrent-minibatch-seqs", type=int, default=8)
    parser.add_argument("--recurrent-rl-lr", type=float, default=None)
    parser.add_argument("--recurrent-clip", type=float, default=None)
    parser.add_argument("--recurrent-entropy-coeff", type=float, default=None)
    parser.add_argument("--recurrent-max-grad-norm", type=float, default=None)
    parser.add_argument("--recurrent-bc-kl-coeff", type=float, default=None)
    parser.add_argument("--recurrent-bc-comm-kl-coeff", type=float, default=None)
    parser.add_argument("--recurrent-rl-balanced-rollouts", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--recurrent-rl-rollout-eval-decoding", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--recurrent-rl-restore-best", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--recurrent-rl-save-best", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--recurrent-train-map-sizes", default="")
    parser.add_argument("--recurrent-train-map-sampling-weights", default="")
    parser.add_argument("--recurrent-map-max-steps", default="")
    parser.add_argument("--recurrent-eval-map-sizes", default="")
    parser.add_argument("--recurrent-eval-seed-count", type=int, default=1)
    parser.add_argument("--recurrent-eval-seed-list", default="")
    parser.add_argument("--recurrent-dagger-seed-list", default="")
    parser.add_argument("--recurrent-init", default=None)
    parser.add_argument(
        "--recurrent-init-template",
        default="",
        help=(
            "Seed-specific recurrent init path template. Supports {seed}, {scenario}, "
            "{map_size}, {agents}, {algorithm}, and {run_name}."
        ),
    )
    parser.add_argument("--recurrent-init-for-dagger", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--recurrent-init-allow-obs-dim-mismatch", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--recurrent-comm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--recurrent-comm-token-limit", type=int, default=8)
    parser.add_argument("--recurrent-comm-vocab-size", type=int, default=32)
    parser.add_argument("--recurrent-hidden-dim", type=int, default=128)
    parser.add_argument("--recurrent-eval-send-threshold", type=float, default=0.25)
    parser.add_argument("--recurrent-calibrate-send-threshold", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--recurrent-obs-memory-mode", default="egocentric", choices=["full", "egocentric"])
    parser.add_argument("--recurrent-obs-memory-radius", type=int, default=4)
    parser.add_argument("--recurrent-eval-signal-exact-target-memory-steps", type=int, default=32)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-mode", default="offline", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-project", default="syncorsink-core-training")
    parser.add_argument("--strict-wandb", action="store_true", help="Fail a run if W&B was requested but did not initialize")
    parser.add_argument("--output-dir", default="logs/core_training_sweep")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args(argv)
    if args.seeds is None:
        args.seeds = [0 if args.seed is None else args.seed]
    elif args.seed is not None and args.seed not in args.seeds:
        args.seeds = sorted([*args.seeds, args.seed])
    args.seeds = sorted(set(args.seeds))
    return _apply_recurrent_ppo_profile(args)


def main(argv: list[str] | None = None) -> int:
    payload = run_suite(parse_args(argv))
    return 1 if payload["overall"]["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
