import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from syncorsink.envs import SyncOrSinkEnv, SyncOrSinkConfig
from syncorsink.eval.splits import make_split_seeds, split_from_name
from syncorsink.eval.solvability import check_solvability


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="signal_hunt")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max", type=int, default=20)
    args = parser.parse_args()

    spec = split_from_name(args.split)
    seeds = make_split_seeds(spec)[: args.max]

    fails = 0
    for seed in seeds:
        config = SyncOrSinkConfig(scenario=args.scenario, split=args.split, map_variant=0, track="ctde")
        env = SyncOrSinkEnv(config)
        env.reset(seed=seed)
        ok, reason = check_solvability(env)
        if not ok:
            print("FAIL", seed, reason)
            fails += 1
    print("checked", len(seeds), "fails", fails)


if __name__ == "__main__":
    main()
