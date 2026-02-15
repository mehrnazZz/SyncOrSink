import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from syncorsink.envs import SyncOrSinkEnv, SyncOrSinkConfig
from syncorsink.eval.protocol import evaluate_split
from syncorsink.policies.scripted import pipeline_planner, energy_planner, signal_hunt_planner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="signal_hunt")
    parser.add_argument("--split", default="test")
    parser.add_argument("--episodes-per-seed", type=int, default=1)
    parser.add_argument("--policy", default="scripted", choices=["scripted"])
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="syncorsink")
    parser.add_argument("--wandb-run", default=None)
    args = parser.parse_args()

    def env_factory():
        config = SyncOrSinkConfig(scenario=args.scenario, split=args.split)
        return SyncOrSinkEnv(config)

    def policy_factory(env):
        if args.scenario == "pipeline_assembly":
            return pipeline_planner(env)
        if args.scenario == "energy_grid":
            return energy_planner(env)
        return signal_hunt_planner(env)

    if args.wandb:
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_run, config={
            "scenario": args.scenario,
            "split": args.split,
            "episodes_per_seed": args.episodes_per_seed,
            "policy": args.policy,
        })
    else:
        wandb_run = None

    result = evaluate_split(
        env_factory,
        policy_factory,
        split=args.split,
        episodes_per_seed=args.episodes_per_seed,
    )
    print("split", result.split)
    print("mean_success", result.mean_success, "std", result.std_success)
    print("mean_return", result.mean_return, "std", result.std_return)
    print("mean_steps", result.mean_steps, "std", result.std_steps)
    print("mean_comm", result.mean_comm, "std", result.std_comm)

    if wandb_run is not None:
        # log per-seed metrics
        for seed, summary in zip(result.seeds, result.summaries):
            wandb_run.log({
                "seed": seed,
                "seed_success": summary.success_rate,
                "seed_return": summary.avg_return,
                "seed_steps": summary.avg_steps,
                "seed_comm": summary.avg_comm_tokens,
            })
        wandb_run.log({
            "mean_success": result.mean_success,
            "std_success": result.std_success,
            "mean_return": result.mean_return,
            "std_return": result.std_return,
            "mean_steps": result.mean_steps,
            "std_steps": result.std_steps,
            "mean_comm": result.mean_comm,
            "std_comm": result.std_comm,
        })
        wandb_run.finish()


if __name__ == "__main__":
    main()
