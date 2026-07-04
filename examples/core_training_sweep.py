from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
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
}


def build_command(
    *,
    algorithm: str,
    case: ScenarioCase,
    checkpoint_path: Path,
    args,
    run_name: str,
) -> list[str]:
    if algorithm not in TRAIN_SCRIPTS:
        raise ValueError(f"Unknown algorithm: {algorithm}")

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
        str(args.seed),
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
        cmd.extend(["--comm", "--critic-mode", args.mappo_critic_mode])
        if args.mappo_shared_actor:
            cmd.append("--shared-actor")
    if args.wandb:
        cmd.append("--wandb")
    return cmd


def run_suite(args) -> dict:
    suite_dir = _suite_dir(args)
    suite_dir.mkdir(parents=True, exist_ok=True)
    cases = [DEFAULT_CASES[name] for name in args.scenarios]
    records: list[RunRecord] = []

    for algorithm in args.algorithms:
        for case in cases:
            run_name = f"{algorithm}_{case.scenario}_{case.map_size}x{case.map_size}_seed{args.seed}"
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
            )
            record = _run_one(
                cmd=cmd,
                algorithm=algorithm,
                case=case,
                run_dir=run_dir,
                checkpoint_path=checkpoint_path,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                dry_run=args.dry_run,
                wandb_mode=args.wandb_mode,
            )
            records.append(record)
            _write_json(run_dir / "run_summary.json", asdict(record))
            print(_format_record(record), flush=True)
            if args.fail_fast and record.status == "failed":
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
            "seed": args.seed,
            "wandb": args.wandb,
            "wandb_mode": args.wandb_mode,
            "wandb_project": args.wandb_project,
        },
        "cases": [asdict(case) for case in cases],
        "runs": [asdict(record) for record in records],
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
    run_dir: Path,
    checkpoint_path: Path,
    stdout_path: Path,
    stderr_path: Path,
    dry_run: bool,
    wandb_mode: str,
) -> RunRecord:
    start = time.time()
    if dry_run:
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return RunRecord(
            algorithm=algorithm,
            scenario=case.scenario,
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
        )

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["WANDB_MODE"] = wandb_mode
    env.setdefault("WANDB_SILENT", "true")
    wandb_dir = run_dir / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    env["WANDB_DIR"] = str(wandb_dir)

    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=stdout, stderr=stderr, check=False)

    stdout_tail = _tail_lines(stdout_path)
    stderr_tail = _tail_lines(stderr_path)
    status = "complete" if proc.returncode == 0 and checkpoint_path.exists() else "failed"
    return RunRecord(
        algorithm=algorithm,
        scenario=case.scenario,
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
        eval_metrics=_parse_last_eval(stdout_tail),
    )


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
    return (
        f"{record.status:8s} {record.algorithm:8s} {record.scenario:18s} "
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mappo-critic-mode", default="central", choices=["local", "central"])
    parser.add_argument("--mappo-shared-actor", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-mode", default="offline", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-project", default="syncorsink-core-training")
    parser.add_argument("--output-dir", default="logs/core_training_sweep")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    payload = run_suite(parse_args(argv))
    return 1 if payload["overall"]["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
