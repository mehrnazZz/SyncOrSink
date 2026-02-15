from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class EpisodeStats:
    total_reward: float
    steps: int
    success: bool
    comm_tokens: int
    per_agent_reward: Dict[int, float]
    per_agent_comm: Dict[int, int]


@dataclass
class EvalSummary:
    episodes: int
    success_rate: float
    avg_return: float
    avg_steps: float
    avg_comm_tokens: float
    avg_agent_reward: Dict[int, float]
    avg_agent_comm: Dict[int, float]


def summarize(episodes: List[EpisodeStats]) -> EvalSummary:
    if not episodes:
        return EvalSummary(0, 0.0, 0.0, 0.0, 0.0, {}, {})
    total = len(episodes)
    success_rate = sum(1 for e in episodes if e.success) / total
    avg_return = sum(e.total_reward for e in episodes) / total
    avg_steps = sum(e.steps for e in episodes) / total
    avg_comm = sum(e.comm_tokens for e in episodes) / total

    agent_ids = sorted({aid for e in episodes for aid in e.per_agent_reward.keys()})
    avg_agent_reward = {aid: 0.0 for aid in agent_ids}
    avg_agent_comm = {aid: 0.0 for aid in agent_ids}
    for aid in agent_ids:
        avg_agent_reward[aid] = sum(e.per_agent_reward.get(aid, 0.0) for e in episodes) / total
        avg_agent_comm[aid] = sum(e.per_agent_comm.get(aid, 0) for e in episodes) / total

    return EvalSummary(
        total,
        success_rate,
        avg_return,
        avg_steps,
        avg_comm,
        avg_agent_reward,
        avg_agent_comm,
    )
