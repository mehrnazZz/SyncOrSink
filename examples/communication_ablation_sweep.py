from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Callable

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from syncorsink.envs import SyncOrSinkConfig, SyncOrSinkEnv
from syncorsink.eval.runner import run_episodes
from syncorsink.policies.local_oracle import local_oracle
from syncorsink.policies.oracle import energy_oracle_planner
from syncorsink.policies.planner_comm import (
    energy_planner_comm,
    pipeline_planner_comm,
    signal_hunt_planner_comm,
)


PolicyFactory = Callable[[SyncOrSinkEnv], Callable]


@dataclass(frozen=True)
class SweepCase:
    scenario: str
    map_size: int
    num_agents: int
    fov_preset: str
    max_steps: int
    energy_preset: str | None = None


@dataclass
class RunRow:
    scenario: str
    map_size: int
    num_agents: int
    fov_preset: str
    max_steps: int
    energy_preset: str | None
    condition: str
    policy: str
    episodes: int
    seed: int
    success_rate: float
    avg_return: float
    avg_steps: float
    avg_comm_tokens: float


@dataclass
class GapRow:
    scenario: str
    map_size: int
    comm_success_rate: float
    no_comm_success_rate: float
    success_gap: float
    comm_avg_tokens: float
    no_comm_avg_tokens: float
    passes_threshold: bool


COMM_POLICIES: dict[str, tuple[str, PolicyFactory]] = {
    "signal_hunt": ("signal_hunt_planner_comm", signal_hunt_planner_comm),
    "energy_grid": ("energy_planner_comm", energy_planner_comm),
    "pipeline_assembly": ("pipeline_planner_comm", pipeline_planner_comm),
}

UPPER_BOUND_POLICIES: dict[str, tuple[str, PolicyFactory]] = {
    "signal_hunt": ("signal_hunt_planner_comm", signal_hunt_planner_comm),
    "energy_grid": ("energy_oracle_planner", energy_oracle_planner),
    "pipeline_assembly": ("pipeline_planner_comm", pipeline_planner_comm),
}


def default_case(scenario: str, map_size: int) -> SweepCase:
    if scenario == "signal_hunt":
        return SweepCase(
            scenario=scenario,
            map_size=map_size,
            num_agents=2 if map_size == 8 else 4,
            fov_preset="easy" if map_size == 8 else "medium",
            max_steps=120 if map_size == 8 else (180 if map_size == 16 else 400),
        )
    if scenario == "energy_grid":
        return SweepCase(
            scenario=scenario,
            map_size=map_size,
            num_agents=3 if map_size == 8 else 4,
            fov_preset="easy" if map_size == 8 else "medium",
            max_steps=180 if map_size == 8 else (300 if map_size == 16 else 500),
            energy_preset="easy" if map_size == 8 else "hard",
        )
    if scenario == "pipeline_assembly":
        return SweepCase(
            scenario=scenario,
            map_size=map_size,
            num_agents=3 if map_size == 8 else 4,
            fov_preset="easy" if map_size == 8 else "medium",
            max_steps=180 if map_size == 8 else (300 if map_size == 16 else 600),
        )
    raise ValueError(f"Unknown scenario: {scenario}")


def make_env(case: SweepCase) -> SyncOrSinkEnv:
    kwargs = {
        "scenario": case.scenario,
        "map_size": case.map_size,
        "num_agents": case.num_agents,
        "fov_preset": case.fov_preset,
        "max_steps": case.max_steps,
        "track": "dtde",
    }
    if case.energy_preset is not None:
        kwargs["energy_preset"] = case.energy_preset
        kwargs["energy_private_monitor"] = True
    return SyncOrSinkEnv(SyncOrSinkConfig(**kwargs))


def run_condition(
    case: SweepCase,
    *,
    condition: str,
    policy_name: str,
    factory: PolicyFactory,
    episodes: int,
    seed: int,
) -> RunRow:
    env = make_env(case)
    policy = factory(env)
    summary, _ = run_episodes(env, policy, episodes=episodes, seed=seed)
    return RunRow(
        scenario=case.scenario,
        map_size=case.map_size,
        num_agents=case.num_agents,
        fov_preset=case.fov_preset,
        max_steps=case.max_steps,
        energy_preset=case.energy_preset,
        condition=condition,
        policy=policy_name,
        episodes=summary.episodes,
        seed=seed,
        success_rate=float(summary.success_rate),
        avg_return=float(summary.avg_return),
        avg_steps=float(summary.avg_steps),
        avg_comm_tokens=float(summary.avg_comm_tokens),
    )


def run_sweep(args) -> dict:
    cases = [default_case(scenario, map_size) for scenario in args.scenarios for map_size in args.map_sizes]
    rows: list[RunRow] = []
    gaps: list[GapRow] = []

    wandb_run = _start_wandb(args) if args.wandb else None
    try:
        for case in cases:
            comm_name, comm_factory = COMM_POLICIES[case.scenario]
            no_comm_name, no_comm_factory = "local_oracle", local_oracle
            condition_specs: list[tuple[str, str, PolicyFactory]] = [
                ("comm_expert", comm_name, comm_factory),
                ("no_comm_local", no_comm_name, no_comm_factory),
            ]
            if args.include_upper_bound:
                upper_name, upper_factory = UPPER_BOUND_POLICIES[case.scenario]
                condition_specs.append(("upper_bound", upper_name, upper_factory))

            case_rows: dict[str, RunRow] = {}
            for condition, policy_name, factory in condition_specs:
                row = run_condition(
                    case,
                    condition=condition,
                    policy_name=policy_name,
                    factory=factory,
                    episodes=args.episodes,
                    seed=args.seed,
                )
                rows.append(row)
                case_rows[condition] = row
                print(_format_row(row), flush=True)
                if wandb_run is not None:
                    wandb_run.log(_wandb_metrics(row))

            comm = case_rows["comm_expert"]
            no_comm = case_rows["no_comm_local"]
            gap = GapRow(
                scenario=case.scenario,
                map_size=case.map_size,
                comm_success_rate=comm.success_rate,
                no_comm_success_rate=no_comm.success_rate,
                success_gap=comm.success_rate - no_comm.success_rate,
                comm_avg_tokens=comm.avg_comm_tokens,
                no_comm_avg_tokens=no_comm.avg_comm_tokens,
                passes_threshold=(
                    comm.success_rate >= args.min_comm_success
                    and (comm.success_rate - no_comm.success_rate) >= args.min_success_gap
                    and comm.avg_comm_tokens > args.min_comm_tokens
                    and no_comm.avg_comm_tokens <= args.max_no_comm_tokens
                ),
            )
            gaps.append(gap)
            print(_format_gap(gap), flush=True)
            if wandb_run is not None:
                wandb_run.log(_wandb_gap(gap))
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    payload = {
        "suite": "communication_ablation_sweep",
        "episodes": int(args.episodes),
        "seed": int(args.seed),
        "thresholds": {
            "min_comm_success": float(args.min_comm_success),
            "min_success_gap": float(args.min_success_gap),
            "min_comm_tokens": float(args.min_comm_tokens),
            "max_no_comm_tokens": float(args.max_no_comm_tokens),
        },
        "rows": [asdict(row) for row in rows],
        "gaps": [asdict(gap) for gap in gaps],
        "overall": {
            "mean_comm_success_rate": mean(g.comm_success_rate for g in gaps) if gaps else 0.0,
            "mean_no_comm_success_rate": mean(g.no_comm_success_rate for g in gaps) if gaps else 0.0,
            "mean_success_gap": mean(g.success_gap for g in gaps) if gaps else 0.0,
            "all_pass_threshold": all(g.passes_threshold for g in gaps) if gaps else False,
        },
    }
    _write_outputs(payload, args)
    if args.fail_on_weak_gap and not payload["overall"]["all_pass_threshold"]:
        weak = [f"{g.scenario}:{g.map_size}" for g in gaps if not g.passes_threshold]
        raise SystemExit(f"communication gap threshold failed for {', '.join(weak)}")
    return payload


def _format_row(row: RunRow) -> str:
    return (
        f"{row.scenario:18s} {row.map_size:2d}x{row.map_size:<2d} "
        f"{row.condition:13s} success={row.success_rate:.3f} "
        f"return={row.avg_return:.2f} steps={row.avg_steps:.1f} comm={row.avg_comm_tokens:.1f}"
    )


def _format_gap(gap: GapRow) -> str:
    status = "PASS" if gap.passes_threshold else "WEAK"
    return (
        f"gap {gap.scenario:18s} {gap.map_size:2d}x{gap.map_size:<2d} "
        f"comm={gap.comm_success_rate:.3f} no_comm={gap.no_comm_success_rate:.3f} "
        f"gap={gap.success_gap:.3f} {status}"
    )


def _wandb_metrics(row: RunRow) -> dict:
    prefix = f"{row.scenario}/{row.map_size}x{row.map_size}/{row.condition}"
    return {
        f"{prefix}/success_rate": row.success_rate,
        f"{prefix}/avg_return": row.avg_return,
        f"{prefix}/avg_steps": row.avg_steps,
        f"{prefix}/avg_comm_tokens": row.avg_comm_tokens,
    }


def _wandb_gap(gap: GapRow) -> dict:
    prefix = f"{gap.scenario}/{gap.map_size}x{gap.map_size}/gap"
    return {
        f"{prefix}/success_gap": gap.success_gap,
        f"{prefix}/comm_success_rate": gap.comm_success_rate,
        f"{prefix}/no_comm_success_rate": gap.no_comm_success_rate,
        f"{prefix}/passes_threshold": int(gap.passes_threshold),
    }


def _start_wandb(args):
    import wandb

    return wandb.init(
        project=args.wandb_project,
        name=args.wandb_run,
        mode=args.wandb_mode,
        config={
            "suite": "communication_ablation_sweep",
            "scenarios": args.scenarios,
            "map_sizes": args.map_sizes,
            "episodes": args.episodes,
            "seed": args.seed,
        },
    )


def _write_outputs(payload: dict, args) -> None:
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {path}")
    if args.output_csv:
        path = Path(args.output_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(payload["rows"][0]) if payload["rows"] else [])
            if payload["rows"]:
                writer.writeheader()
                writer.writerows(payload["rows"])
        print(f"wrote {path}")


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Run communication-vs-no-communication ablation sweeps.")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["signal_hunt", "energy_grid", "pipeline_assembly"],
        choices=["signal_hunt", "energy_grid", "pipeline_assembly"],
    )
    parser.add_argument("--map-sizes", nargs="+", type=int, default=[8, 16], choices=[8, 16, 24, 32])
    parser.add_argument("--include-32", action="store_true", help="Append 32x32 to --map-sizes if absent")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-upper-bound", action="store_true")
    parser.add_argument("--min-comm-success", type=float, default=0.8)
    parser.add_argument("--min-success-gap", type=float, default=0.5)
    parser.add_argument("--min-comm-tokens", type=float, default=1.0)
    parser.add_argument("--max-no-comm-tokens", type=float, default=0.0)
    parser.add_argument("--fail-on-weak-gap", action="store_true")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="syncorsink")
    parser.add_argument("--wandb-run", default=None)
    parser.add_argument("--wandb-mode", default="offline", choices=["online", "offline", "disabled"])
    args = parser.parse_args(argv)
    if args.include_32 and 32 not in args.map_sizes:
        args.map_sizes = [*args.map_sizes, 32]
    args.map_sizes = sorted(set(args.map_sizes))
    return args


def main(argv: list[str] | None = None) -> int:
    run_sweep(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
