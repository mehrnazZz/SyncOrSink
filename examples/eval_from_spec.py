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
from syncorsink.policies.oracle import (
    pipeline_oracle,
    pipeline_oracle_strong,
    energy_oracle,
    energy_oracle_strong,
    energy_oracle_planner,
    signal_hunt_oracle,
    signal_hunt_oracle_strong,
)
from syncorsink.policies.comm_wrapper import wrap_oracle_with_comm
from syncorsink.policies.planner_comm import (
    pipeline_planner_comm,
    pipeline_planner_follower,
    pipeline_planner_comm_followers,
    pipeline_planner_comm_followers_regions,
    pipeline_planner_dispatcher,
    pipeline_planner_semidec,
    energy_planner_comm,
    signal_hunt_planner_comm,
)
from syncorsink.policies.comm_mat_policy import CommMATPolicy, CommMATPolicyConfig
from syncorsink.policies.submission import load_policy_entrypoint
from syncorsink.llm.policy import LLMPolicy


def dummy_llm(prompt: str):
    return '{"action": 4, "message_text": ""}'


def build_marl_policy(spec, env):
    if spec.policy_entrypoint:
        return load_policy_entrypoint(
            spec.policy_entrypoint,
            env=env,
            spec=spec.__dict__,
            checkpoint=spec.policy_checkpoint,
            kwargs=spec.policy_kwargs or {},
            decentralized=True,
        ).policy

    policy = spec.policy
    if policy == "random":
        return random_policy(env.action_space, env.num_agents)
    if policy == "scripted":
        if spec.scenario == "pipeline_assembly":
            return pipeline_planner(env)
        if spec.scenario == "energy_grid":
            return energy_planner(env)
        return signal_hunt_planner(env)
    if policy == "oracle":
        if spec.scenario == "pipeline_assembly":
            return pipeline_oracle(env)
        if spec.scenario == "energy_grid":
            return energy_oracle(env)
        return signal_hunt_oracle(env)
    if policy == "oracle_strong":
        if spec.scenario == "pipeline_assembly":
            return pipeline_oracle_strong(env)
        if spec.scenario == "energy_grid":
            return energy_oracle_strong(env)
        return signal_hunt_oracle_strong(env)
    if policy == "oracle_planner":
        if spec.scenario == "energy_grid":
            return energy_oracle_planner(env)
        if spec.scenario == "pipeline_assembly":
            return pipeline_oracle_strong(env)
        return signal_hunt_oracle_strong(env)
    if policy == "oracle_comm":
        if spec.scenario == "pipeline_assembly":
            base = pipeline_oracle_strong(env)
        elif spec.scenario == "energy_grid":
            base = energy_oracle_strong(env)
        else:
            base = signal_hunt_oracle_strong(env)
        return wrap_oracle_with_comm(base, env)
    if policy == "pipeline_planner_comm":
        return pipeline_planner_comm(env)
    if policy == "pipeline_planner_follower":
        return pipeline_planner_follower(env)
    if policy == "pipeline_planner_comm_followers":
        return pipeline_planner_comm_followers(env)
    if policy == "pipeline_planner_comm_followers_regions":
        return pipeline_planner_comm_followers_regions(env)
    if policy == "pipeline_planner_dispatcher":
        return pipeline_planner_dispatcher(env)
    if policy == "pipeline_planner_semidec":
        return pipeline_planner_semidec(env)
    if policy == "energy_planner_comm":
        return energy_planner_comm(env)
    if policy == "signal_hunt_planner_comm":
        return signal_hunt_planner_comm(env)
    if policy == "comm_mat":
        checkpoint = spec.policy_checkpoint
        if checkpoint and not os.path.isabs(checkpoint):
            checkpoint = os.path.join(ROOT, checkpoint)
        return CommMATPolicy(
            config=CommMATPolicyConfig(
                deterministic=spec.comm_mat_deterministic,
                send_threshold=spec.comm_mat_send_threshold,
            ),
            checkpoint=checkpoint,
        )
    raise ValueError(f"Unsupported policy in spec: {policy}")


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
        map_size=spec.map_size,
        num_agents=spec.num_agents,
        fov_preset=spec.fov_preset,
        max_steps=spec.max_steps,
        comm_mode=spec.comm_mode,
        energy_preset=spec.energy_preset,
        energy_private_monitor=spec.energy_private_monitor,
    )
    env = SyncOrSinkEnv(config)

    if spec.mode == "llm":
        policy = LLMPolicy(dummy_llm)
        episodes = run_llm_episodes(env, policy, episodes=spec.episodes, seed=0)
        summary = summarize(episodes)
    else:
        policy = build_marl_policy(spec, env)
        summary, _ = run_episodes(env, policy, episodes=spec.episodes, seed=0)

    print("episodes", summary.episodes)
    print("success_rate", summary.success_rate)
    print("avg_return", summary.avg_return)
    print("avg_steps", summary.avg_steps)
    print("avg_comm_tokens", summary.avg_comm_tokens)


if __name__ == "__main__":
    main()
