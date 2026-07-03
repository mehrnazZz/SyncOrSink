import argparse
import os
import sys
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from syncorsink.eval.benchmark_spec import load_benchmark
from syncorsink.eval.leaderboard import (
    collect_leaderboard_entries,
    render_csv,
    render_json,
    render_markdown,
)


def _write(path: str | None, text: str) -> None:
    if path is None:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print("wrote", out)


def main():
    parser = argparse.ArgumentParser(description="Build SyncOrSink leaderboard tables from result artifacts.")
    parser.add_argument(
        "--results",
        nargs="+",
        default=["results/syncorsink_v0_1"],
        help="Result JSON file(s) or directories to scan recursively",
    )
    parser.add_argument(
        "--benchmark",
        default="benchmarks/syncorsink_v0_1.json",
        help="Benchmark manifest used to validate result case coverage",
    )
    parser.add_argument("--out-md", default="docs/leaderboard_results.md", help="Markdown output path")
    parser.add_argument("--out-csv", default=None, help="Optional CSV output path")
    parser.add_argument("--out-json", default=None, help="Optional machine-readable JSON output path")
    parser.add_argument("--allow-partial", action="store_true", help="Allow artifacts with a subset of benchmark cases")
    parser.add_argument("--skip-invalid", action="store_true", help="Skip invalid result files and report warnings")
    args = parser.parse_args()

    benchmark = load_benchmark(args.benchmark) if args.benchmark else None
    collection = collect_leaderboard_entries(
        args.results,
        benchmark=benchmark,
        allow_partial=args.allow_partial,
        skip_invalid=args.skip_invalid,
    )

    benchmark_name = benchmark.name if benchmark is not None else None
    benchmark_version = benchmark.version if benchmark is not None else None
    markdown = render_markdown(
        collection.entries,
        benchmark_name=benchmark_name,
        benchmark_version=benchmark_version,
        warnings=collection.warnings,
    )
    _write(args.out_md, markdown)
    _write(args.out_csv, render_csv(collection.entries))
    _write(args.out_json, render_json(collection.entries, collection.warnings))
    print("entries", len(collection.entries))
    if collection.warnings:
        print("warnings", len(collection.warnings))


if __name__ == "__main__":
    main()
