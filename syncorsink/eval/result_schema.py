from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from .metrics import EvalSummary


RESULT_SCHEMA_VERSION = "syncorsink.result.v0.1"

LEADERBOARD_TRACKS = {
    "symbolic_dtde",
    "symbolic_ctde",
    "rgb_vision",
    "low_comm",
    "no_comm",
    "llm_text",
    "vlm_rgb",
    "sample_efficiency",
    "ood_generalization",
    "human_playable",
}


@dataclass(frozen=True)
class SubmissionInfo:
    name: str
    method_name: str
    method_type: str
    authors: list[str]
    repository: str | None = None
    checkpoint_uri: str | None = None
    paper_uri: str | None = None
    notes: str | None = None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def summary_to_case_result(
    case_name: str,
    summary: EvalSummary,
    *,
    spec: Mapping[str, Any] | None = None,
    weight: float = 1.0,
    tags: Iterable[str] | None = None,
    seeds: Iterable[int] | None = None,
) -> Dict[str, Any]:
    return {
        "name": case_name,
        "weight": float(weight),
        "tags": list(tags or []),
        "spec": dict(spec or {}),
        "seeds": list(seeds or []),
        "metrics": {
            "episodes": int(summary.episodes),
            "success_rate": float(summary.success_rate),
            "avg_return": float(summary.avg_return),
            "avg_steps": float(summary.avg_steps),
            "avg_comm_tokens": float(summary.avg_comm_tokens),
            "avg_agent_reward": {str(k): float(v) for k, v in summary.avg_agent_reward.items()},
            "avg_agent_comm": {str(k): float(v) for k, v in summary.avg_agent_comm.items()},
        },
    }


def make_result_artifact(
    *,
    benchmark_name: str,
    benchmark_version: str,
    track: str,
    submission: SubmissionInfo | Mapping[str, Any],
    cases: Iterable[Mapping[str, Any]],
    generated_at: str | None = None,
) -> Dict[str, Any]:
    submission_data = asdict(submission) if isinstance(submission, SubmissionInfo) else dict(submission)
    artifact = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "benchmark_name": benchmark_name,
        "benchmark_version": benchmark_version,
        "track": track,
        "generated_at": generated_at or utc_now_iso(),
        "submission": submission_data,
        "cases": [dict(case) for case in cases],
    }
    validate_result_artifact(artifact)
    return artifact


def save_result_artifact(artifact: Mapping[str, Any], path: str | Path) -> None:
    validate_result_artifact(artifact)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_result_artifact(path: str | Path) -> Dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_result_artifact(data)
    return data


def validate_result_artifact(data: Mapping[str, Any]) -> None:
    if not isinstance(data, Mapping):
        raise ValueError("result artifact must be an object")
    _require_equal(data, "schema_version", RESULT_SCHEMA_VERSION)
    _require_nonempty_string(data, "benchmark_name")
    _require_nonempty_string(data, "benchmark_version")
    _require_nonempty_string(data, "generated_at")
    track = _require_nonempty_string(data, "track")
    if track not in LEADERBOARD_TRACKS:
        raise ValueError(f"result track must be one of {sorted(LEADERBOARD_TRACKS)}")

    submission = data.get("submission")
    if not isinstance(submission, Mapping):
        raise ValueError("result.submission must be an object")
    for key in ("name", "method_name", "method_type"):
        _require_nonempty_string(submission, key, prefix="result.submission")
    authors = submission.get("authors")
    if not isinstance(authors, list) or not all(isinstance(a, str) and a for a in authors):
        raise ValueError("result.submission.authors must be a non-empty list of strings")

    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("result.cases must be a non-empty list")
    names = set()
    for idx, case in enumerate(cases):
        _validate_case_result(case, idx)
        name = case["name"]
        if name in names:
            raise ValueError(f"result.cases contains duplicate case name: {name}")
        names.add(name)


def _validate_case_result(case: Any, idx: int) -> None:
    if not isinstance(case, Mapping):
        raise ValueError(f"result.cases[{idx}] must be an object")
    _require_nonempty_string(case, "name", prefix=f"result.cases[{idx}]")
    weight = case.get("weight", 1.0)
    if not _is_number(weight) or float(weight) <= 0:
        raise ValueError(f"result.cases[{idx}].weight must be a positive number")
    tags = case.get("tags", [])
    if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
        raise ValueError(f"result.cases[{idx}].tags must be a list of strings")
    seeds = case.get("seeds", [])
    if not isinstance(seeds, list) or not all(isinstance(seed, int) for seed in seeds):
        raise ValueError(f"result.cases[{idx}].seeds must be a list of integers")
    metrics = case.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError(f"result.cases[{idx}].metrics must be an object")
    episodes = metrics.get("episodes")
    if not isinstance(episodes, int) or episodes < 1:
        raise ValueError(f"result.cases[{idx}].metrics.episodes must be int >= 1")
    success = _require_number(metrics, "success_rate", prefix=f"result.cases[{idx}].metrics")
    if not 0.0 <= success <= 1.0:
        raise ValueError(f"result.cases[{idx}].metrics.success_rate must be in [0, 1]")
    _require_number(metrics, "avg_return", prefix=f"result.cases[{idx}].metrics")
    for key in ("avg_steps", "avg_comm_tokens"):
        value = _require_number(metrics, key, prefix=f"result.cases[{idx}].metrics")
        if value < 0:
            raise ValueError(f"result.cases[{idx}].metrics.{key} must be >= 0")


def _require_equal(data: Mapping[str, Any], key: str, expected: str) -> None:
    value = data.get(key)
    if value != expected:
        raise ValueError(f"result.{key} must be {expected!r}")


def _require_nonempty_string(data: Mapping[str, Any], key: str, prefix: str = "result") -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{prefix}.{key} must be a non-empty string")
    return value


def _require_number(data: Mapping[str, Any], key: str, prefix: str = "result") -> float:
    value = data.get(key)
    if not _is_number(value):
        raise ValueError(f"{prefix}.{key} must be a number")
    return float(value)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
