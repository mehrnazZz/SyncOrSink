import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from syncorsink.envs import SyncOrSinkEnv, SyncOrSinkConfig
from syncorsink.eval.spec import load_spec
from syncorsink.eval.metrics import summarize
from syncorsink.eval.runner import run_episodes
from syncorsink.eval.llm_runner import run_llm_episodes
from syncorsink.policies.random_policy import random_policy
from syncorsink.policies.scripted import pipeline_planner, energy_planner, signal_hunt_planner
from syncorsink.llm.policy import LLMPolicy


def dummy_llm(prompt: str):
    return '{"action": 4, "message_text": ""}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    args = parser.parse_args()

    spec = load_spec(args.spec)
    config = SyncOrSinkConfig(
        scenario=spec.scenario,
        split=spec.split,
        map_variant=spec.map_variant,
        track=getattr(spec, "track", "dtde"),
    )
    env = SyncOrSinkEnv(config)

    if spec.mode == "llm":
        policy = LLMPolicy(dummy_llm)
        episodes = run_llm_episodes(env, policy, episodes=spec.episodes, seed=0)
        summary = summarize(episodes)
    else:
        if spec.policy == "random":
            policy = random_policy(env.action_space, env.num_agents)
        elif spec.policy == "scripted":
            if spec.scenario == "pipeline_assembly":
                policy = pipeline_planner(env)
            elif spec.scenario == "energy_grid":
                policy = energy_planner(env)
            else:
                policy = signal_hunt_planner(env)
        else:
            policy = random_policy(env.action_space, env.num_agents)
        summary, _ = run_episodes(env, policy, episodes=spec.episodes, seed=0)

    print("episodes", summary.episodes)
    print("success_rate", summary.success_rate)
    print("avg_return", summary.avg_return)
    print("avg_steps", summary.avg_steps)
    print("avg_comm_tokens", summary.avg_comm_tokens)


if __name__ == "__main__":
    main()
