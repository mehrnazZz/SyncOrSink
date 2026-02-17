import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from syncorsink.envs import SyncOrSinkEnv, SyncOrSinkConfig
from syncorsink.eval.benchmark_spec import load_benchmark
from syncorsink.eval.metrics import summarize
from syncorsink.eval.runner import run_episodes
from syncorsink.eval.llm_runner import run_llm_episodes
from syncorsink.policies.random_policy import random_policy
from syncorsink.policies.scripted import pipeline_planner, energy_planner, signal_hunt_planner
from syncorsink.policies.comm_mat_policy import CommMATPolicy, CommMATPolicyConfig
from syncorsink.llm.policy import LLMPolicy


def dummy_llm(prompt: str) -> str:
    return '{"action": 4, "message_text": ""}'


def build_policy(spec, env):
    policy = spec.get("policy", "random")
    mode = spec.get("mode", "marl")
    if mode == "llm":
        return LLMPolicy(dummy_llm)
    if policy == "random":
        return random_policy(env.action_space, env.num_agents)
    if policy == "scripted":
        if env.config.scenario == "pipeline_assembly":
            return pipeline_planner(env)
        if env.config.scenario == "energy_grid":
            return energy_planner(env)
        return signal_hunt_planner(env)
    if policy == "comm_mat":
        return CommMATPolicy(
            config=CommMATPolicyConfig(
                deterministic=bool(spec.get("comm_mat_deterministic", True)),
                send_threshold=float(spec.get("comm_mat_send_threshold", 0.5)),
            ),
            checkpoint=spec.get("policy_checkpoint"),
        )
    return random_policy(env.action_space, env.num_agents)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="syncorsink")
    parser.add_argument("--wandb-run", default=None)
    args = parser.parse_args()

    bench = load_benchmark(args.spec)

    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_run, config={
            "benchmark": bench.name,
        })

    for case in bench.cases:
        spec = case.spec
        config = SyncOrSinkConfig(
            scenario=spec["scenario"],
            split=spec.get("split"),
            map_variant=int(spec.get("map_variant", 0)),
            fov_preset=spec.get("fov_preset", "medium"),
            map_size=int(spec.get("map_size", 16)),
            comm_mode=spec.get("comm_mode", "tokens"),
            track=spec.get("track", "dtde"),
        )
        env = SyncOrSinkEnv(config)
        policy = build_policy(spec, env)
        episodes = int(spec.get("episodes", 1))

        if spec.get("mode", "marl") == "llm":
            ep_stats = run_llm_episodes(env, policy, episodes=episodes, seed=0)
            summary = summarize(ep_stats)
        else:
            summary, _ = run_episodes(env, policy, episodes=episodes, seed=0)

        print("case", case.name, "success", summary.success_rate, "return", summary.avg_return)

        if wandb_run is not None:
            wandb_run.log({
                f"{case.name}/success_rate": summary.success_rate,
                f"{case.name}/avg_return": summary.avg_return,
                f"{case.name}/avg_steps": summary.avg_steps,
                f"{case.name}/avg_comm_tokens": summary.avg_comm_tokens,
                f"{case.name}/track": spec.get("track", "dtde"),
            })

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
