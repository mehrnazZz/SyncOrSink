import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from syncorsink.envs.scenario_registry import scenario_registry_as_dict, scenario_table_rows


def main():
    parser = argparse.ArgumentParser(description="List registered SyncOrSink scenarios and metadata.")
    parser.add_argument("--tier", default=None, help="Optional scenario tier: core, advanced, procedural, stress")
    parser.add_argument("--json", action="store_true", help="Print full metadata as JSON")
    args = parser.parse_args()

    if args.json:
        print(json.dumps(scenario_registry_as_dict(tier=args.tier), indent=2, sort_keys=True))
        return

    rows = scenario_table_rows(tier=args.tier)
    headers = ["name", "tier", "domain", "communication_role", "private_information", "coordination_types"]
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
