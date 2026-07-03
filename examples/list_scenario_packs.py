import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from syncorsink.envs.procedural import (
    pack_benchmark_manifest,
    scenario_pack_table_rows,
    scenario_packs_as_dict,
    validate_all_scenario_packs,
)


def main():
    parser = argparse.ArgumentParser(description="List SyncOrSink scenario packs and procedural presets.")
    parser.add_argument("--tier", default=None, help="Optional pack tier: core, core_ood, advanced, procedural, stress")
    parser.add_argument("--json", action="store_true", help="Print full pack metadata as JSON")
    parser.add_argument(
        "--benchmark",
        nargs="+",
        default=None,
        help="Print a benchmark manifest generated from the named pack(s)",
    )
    parser.add_argument("--name", default="syncorsink_generated", help="Generated benchmark name for --benchmark")
    parser.add_argument("--version", default="generated", help="Generated benchmark version for --benchmark")
    parser.add_argument("--compatibility-note", default=None, help="Optional compatibility note for generated manifests")
    args = parser.parse_args()

    validate_all_scenario_packs()

    if args.benchmark:
        manifest = pack_benchmark_manifest(
            args.benchmark,
            name=args.name,
            version=args.version,
            description=f"Generated from scenario packs: {', '.join(args.benchmark)}",
            extra_metadata={"compatibility_note": args.compatibility_note} if args.compatibility_note else None,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return

    if args.json:
        print(json.dumps(scenario_packs_as_dict(tier=args.tier), indent=2, sort_keys=True))
        return

    rows = scenario_pack_table_rows(tier=args.tier)
    headers = ["name", "tier", "version", "presets", "scenarios", "axes"]
    widths = {
        header: max(len(header), *(len(row[header]) for row in rows))
        for header in headers
    }
    print("  ".join(header.ljust(widths[header]) for header in headers))
    print("  ".join("-" * widths[header] for header in headers))
    for row in rows:
        print("  ".join(row[header].ljust(widths[header]) for header in headers))


if __name__ == "__main__":
    main()
