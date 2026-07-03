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


def _check_or_write(path: str, expected: str, *, fix: bool) -> bool:
    out = Path(path)
    if fix:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(expected, encoding="utf-8")
        print("wrote", out)
        return True

    if not out.exists():
        print(f"missing generated file: {out}")
        return False
    actual = out.read_text(encoding="utf-8")
    if actual != expected:
        print(f"stale generated file: {out}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Validate generated SyncOrSink leaderboard outputs.")
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
    parser.add_argument("--md", default="docs/leaderboard_results.md", help="Markdown table path")
    parser.add_argument("--csv", default="docs/leaderboard_results.csv", help="CSV table path")
    parser.add_argument("--json", default="docs/leaderboard_results.json", help="JSON table path")
    parser.add_argument("--allow-partial", action="store_true", help="Allow artifacts with a subset of benchmark cases")
    parser.add_argument("--skip-invalid", action="store_true", help="Skip invalid result files and report warnings")
    parser.add_argument("--fix", action="store_true", help="Rewrite generated outputs instead of checking them")
    args = parser.parse_args()

    benchmark = load_benchmark(args.benchmark)
    collection = collect_leaderboard_entries(
        args.results,
        benchmark=benchmark,
        allow_partial=args.allow_partial,
        skip_invalid=args.skip_invalid,
    )
    expected_md = render_markdown(
        collection.entries,
        benchmark_name=benchmark.name,
        benchmark_version=benchmark.version,
        warnings=collection.warnings,
    )
    expected_csv = render_csv(collection.entries)
    expected_json = render_json(collection.entries, collection.warnings)

    ok = True
    ok = _check_or_write(args.md, expected_md, fix=args.fix) and ok
    ok = _check_or_write(args.csv, expected_csv, fix=args.fix) and ok
    ok = _check_or_write(args.json, expected_json, fix=args.fix) and ok

    print("entries", len(collection.entries))
    if collection.warnings:
        print("warnings", len(collection.warnings))
    if not ok:
        print("leaderboard outputs are stale; rerun with --fix or examples/build_leaderboard.py")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
