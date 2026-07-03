from __future__ import annotations

from typing import Any, Mapping

from .result_schema import validate_result_artifact


def score_result_artifact(artifact: Mapping[str, Any]) -> dict[str, float | int]:
    """Score a leaderboard result artifact.

    SyncOrSink v0.1 ranks by weighted mean success rate. Return, steps, and
    communication are reported as secondary diagnostics rather than blended into
    the primary score, because the scenarios have different reward and horizon
    semantics.
    """
    validate_result_artifact(artifact)
    cases = artifact["cases"]
    total_weight = sum(float(case.get("weight", 1.0)) for case in cases)

    weighted_success = 0.0
    weighted_return = 0.0
    weighted_steps = 0.0
    weighted_comm = 0.0
    for case in cases:
        weight = float(case.get("weight", 1.0))
        metrics = case["metrics"]
        weighted_success += weight * float(metrics["success_rate"])
        weighted_return += weight * float(metrics["avg_return"])
        weighted_steps += weight * float(metrics["avg_steps"])
        weighted_comm += weight * float(metrics["avg_comm_tokens"])

    mean_success = weighted_success / total_weight
    return {
        "case_count": len(cases),
        "official_score": 100.0 * mean_success,
        "mean_success_rate": mean_success,
        "mean_avg_return": weighted_return / total_weight,
        "mean_avg_steps": weighted_steps / total_weight,
        "mean_avg_comm_tokens": weighted_comm / total_weight,
    }
