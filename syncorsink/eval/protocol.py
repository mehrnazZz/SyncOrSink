from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, stdev
from typing import Callable, Dict, List

from .metrics import EvalSummary
from .runner import run_episodes
from .splits import split_from_name, make_split_seeds


@dataclass
class SplitResult:
    split: str
    seeds: List[int]
    summaries: List[EvalSummary]
    mean_success: float
    std_success: float
    mean_return: float
    std_return: float
    mean_steps: float
    std_steps: float
    mean_comm: float
    std_comm: float


def evaluate_split(
    env_factory: Callable[[], object],
    policy_factory: Callable[[object], Callable],
    split: str,
    episodes_per_seed: int = 1,
) -> SplitResult:
    spec = split_from_name(split)
    seeds = make_split_seeds(spec)
    summaries: List[EvalSummary] = []

    for seed in seeds:
        env = env_factory()
        policy = policy_factory(env)
        summary, _ = run_episodes(env, policy, episodes=episodes_per_seed, seed=seed)
        summaries.append(summary)

    success_rates = [s.success_rate for s in summaries]
    returns = [s.avg_return for s in summaries]
    steps = [s.avg_steps for s in summaries]
    comms = [s.avg_comm_tokens for s in summaries]

    return SplitResult(
        split=split,
        seeds=seeds,
        summaries=summaries,
        mean_success=mean(success_rates),
        std_success=stdev(success_rates) if len(success_rates) > 1 else 0.0,
        mean_return=mean(returns),
        std_return=stdev(returns) if len(returns) > 1 else 0.0,
        mean_steps=mean(steps),
        std_steps=stdev(steps) if len(steps) > 1 else 0.0,
        mean_comm=mean(comms),
        std_comm=stdev(comms) if len(comms) > 1 else 0.0,
    )
