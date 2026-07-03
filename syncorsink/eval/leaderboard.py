from __future__ import annotations

import csv
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .benchmark_spec import BenchmarkSpec
from .result_schema import load_result_artifact, validate_result_artifact
from .scoring import score_result_artifact


POLICY_SPEC_KEYS = {
    "policy",
    "policy_checkpoint",
    "policy_entrypoint",
    "policy_kwargs",
    "comm_mat_deterministic",
    "comm_mat_send_threshold",
}


@dataclass(frozen=True)
class LeaderboardEntry:
    source_path: str
    benchmark_name: str
    benchmark_version: str
    track: str
    generated_at: str
    submission_name: str
    method_name: str
    method_type: str
    authors: str
    repository: str
    checkpoint_uri: str
    paper_uri: str
    notes: str
    case_count: int
    official_score: float
    mean_success_rate: float
    mean_avg_return: float
    mean_avg_steps: float
    mean_avg_comm_tokens: float


@dataclass(frozen=True)
class LeaderboardCollection:
    entries: list[LeaderboardEntry]
    warnings: list[str]


def discover_result_files(paths: Iterable[str | Path]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(p for p in path.rglob("*.json") if p.is_file())
        elif path.is_file() and path.suffix == ".json":
            files.append(path)
        elif path.exists():
            raise ValueError(f"result path is neither a JSON file nor directory: {path}")
        else:
            raise ValueError(f"result path does not exist: {path}")
    return sorted(set(files), key=lambda p: str(p))


def collect_leaderboard_entries(
    paths: Iterable[str | Path],
    *,
    benchmark: BenchmarkSpec | None = None,
    allow_partial: bool = False,
    skip_invalid: bool = False,
) -> LeaderboardCollection:
    entries: list[LeaderboardEntry] = []
    warnings: list[str] = []
    for path in discover_result_files(paths):
        try:
            artifact = load_result_artifact(path)
            if benchmark is not None:
                validate_artifact_against_benchmark(artifact, benchmark, allow_partial=allow_partial)
            entries.append(entry_from_artifact(artifact, path))
        except Exception as exc:
            if not skip_invalid:
                raise
            warnings.append(f"{path}: {exc}")
    return LeaderboardCollection(
        entries=sort_entries(entries),
        warnings=warnings,
    )


def validate_artifact_against_benchmark(
    artifact: Mapping[str, Any],
    benchmark: BenchmarkSpec,
    *,
    allow_partial: bool = False,
) -> None:
    validate_result_artifact(artifact)
    if artifact["benchmark_name"] != benchmark.name:
        raise ValueError(f"benchmark name mismatch: expected {benchmark.name}, got {artifact['benchmark_name']}")
    if artifact["benchmark_version"] != benchmark.version:
        raise ValueError(f"benchmark version mismatch: expected {benchmark.version}, got {artifact['benchmark_version']}")

    expected_cases = {case.name: case for case in benchmark.cases}
    actual_cases = {case["name"]: case for case in artifact["cases"]}
    missing = sorted(set(expected_cases) - set(actual_cases))
    extra = sorted(set(actual_cases) - set(expected_cases))
    if missing and not allow_partial:
        raise ValueError(f"result artifact is missing benchmark cases: {missing}")
    if extra:
        raise ValueError(f"result artifact contains unknown benchmark cases: {extra}")

    for name, actual in actual_cases.items():
        expected = expected_cases[name]
        if abs(float(actual.get("weight", 1.0)) - float(expected.weight)) > 1e-9:
            raise ValueError(f"case {name} weight mismatch")
        _validate_case_spec(name, actual.get("spec", {}), expected.spec)


def entry_from_artifact(artifact: Mapping[str, Any], path: str | Path) -> LeaderboardEntry:
    score = score_result_artifact(artifact)
    submission = artifact["submission"]
    return LeaderboardEntry(
        source_path=str(path),
        benchmark_name=str(artifact["benchmark_name"]),
        benchmark_version=str(artifact["benchmark_version"]),
        track=str(artifact["track"]),
        generated_at=str(artifact["generated_at"]),
        submission_name=str(submission["name"]),
        method_name=str(submission["method_name"]),
        method_type=str(submission["method_type"]),
        authors=", ".join(str(author) for author in submission.get("authors", [])),
        repository=str(submission.get("repository") or ""),
        checkpoint_uri=str(submission.get("checkpoint_uri") or ""),
        paper_uri=str(submission.get("paper_uri") or ""),
        notes=str(submission.get("notes") or ""),
        case_count=int(score["case_count"]),
        official_score=float(score["official_score"]),
        mean_success_rate=float(score["mean_success_rate"]),
        mean_avg_return=float(score["mean_avg_return"]),
        mean_avg_steps=float(score["mean_avg_steps"]),
        mean_avg_comm_tokens=float(score["mean_avg_comm_tokens"]),
    )


def sort_entries(entries: Iterable[LeaderboardEntry]) -> list[LeaderboardEntry]:
    return sorted(
        entries,
        key=lambda e: (
            e.track,
            -e.official_score,
            -e.mean_success_rate,
            e.mean_avg_steps,
            e.mean_avg_comm_tokens,
            e.submission_name.lower(),
        ),
    )


def render_markdown(
    entries: Iterable[LeaderboardEntry],
    *,
    title: str = "SyncOrSink Leaderboard Results",
    benchmark_name: str | None = None,
    benchmark_version: str | None = None,
    warnings: Iterable[str] = (),
) -> str:
    sorted_entries = sort_entries(entries)
    lines = [f"# {title}", ""]
    if benchmark_name is not None:
        suffix = f" v{benchmark_version}" if benchmark_version else ""
        lines.extend([f"Benchmark: `{benchmark_name}{suffix}`", ""])
    lines.extend([
        "Primary score: `100 * weighted_mean(success_rate)`.",
        "",
    ])

    warning_list = list(warnings)
    if warning_list:
        lines.extend(["## Warnings", ""])
        for warning in warning_list:
            lines.append(f"- {_escape_md(warning)}")
        lines.append("")

    if not sorted_entries:
        lines.extend([
            "No validated submissions have been added yet.",
            "",
            "Add result artifacts under `results/syncorsink_v0_1/` and rebuild this file with `examples/build_leaderboard.py`.",
            "",
        ])
        return "\n".join(lines)

    for track in sorted({entry.track for entry in sorted_entries}):
        track_entries = [entry for entry in sorted_entries if entry.track == track]
        lines.extend([
            f"## {track}",
            "",
            "| Rank | Submission | Method | Type | Score | Success | Steps | Comm | Cases | Authors | Links |",
            "|---:|---|---|---|---:|---:|---:|---:|---:|---|---|",
        ])
        for rank, entry in enumerate(track_entries, start=1):
            lines.append(
                "| "
                + " | ".join([
                    str(rank),
                    _escape_md(entry.submission_name),
                    _escape_md(entry.method_name),
                    _escape_md(entry.method_type),
                    _fmt(entry.official_score),
                    _fmt_pct(entry.mean_success_rate),
                    _fmt(entry.mean_avg_steps),
                    _fmt(entry.mean_avg_comm_tokens),
                    str(entry.case_count),
                    _escape_md(entry.authors),
                    _links(entry),
                ])
                + " |"
            )
        lines.append("")
    return "\n".join(lines)


def render_csv(entries: Iterable[LeaderboardEntry]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(asdict(_empty_entry()).keys()), lineterminator="\n")
    writer.writeheader()
    for entry in sort_entries(entries):
        writer.writerow(asdict(entry))
    return output.getvalue()


def render_json(entries: Iterable[LeaderboardEntry], warnings: Iterable[str] = ()) -> str:
    data = {
        "entries": [asdict(entry) for entry in sort_entries(entries)],
        "warnings": list(warnings),
    }
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def _validate_case_spec(name: str, actual: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    if not isinstance(actual, Mapping):
        raise ValueError(f"case {name} spec must be an object")
    for key, expected_value in expected.items():
        if key in POLICY_SPEC_KEYS:
            continue
        if actual.get(key) != expected_value:
            raise ValueError(f"case {name} spec mismatch for {key}: expected {expected_value!r}, got {actual.get(key)!r}")


def _links(entry: LeaderboardEntry) -> str:
    links = []
    if entry.repository:
        links.append(f"[repo]({entry.repository})")
    if entry.paper_uri:
        links.append(f"[paper]({entry.paper_uri})")
    if entry.checkpoint_uri:
        links.append(f"[ckpt]({entry.checkpoint_uri})")
    return ", ".join(links)


def _fmt(value: float) -> str:
    return f"{value:.2f}"


def _fmt_pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _escape_md(value: str) -> str:
    return str(value).replace("|", "\\|")


def _empty_entry() -> LeaderboardEntry:
    return LeaderboardEntry(
        source_path="",
        benchmark_name="",
        benchmark_version="",
        track="",
        generated_at="",
        submission_name="",
        method_name="",
        method_type="",
        authors="",
        repository="",
        checkpoint_uri="",
        paper_uri="",
        notes="",
        case_count=0,
        official_score=0.0,
        mean_success_rate=0.0,
        mean_avg_return=0.0,
        mean_avg_steps=0.0,
        mean_avg_comm_tokens=0.0,
    )
