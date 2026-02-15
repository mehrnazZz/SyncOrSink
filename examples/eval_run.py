import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from syncorsink.envs import SyncOrSinkEnv, SyncOrSinkConfig
from syncorsink.eval.runner import run_episodes
from syncorsink.policies.random_policy import random_policy
from syncorsink.policies.heuristic import heuristic_policy
from syncorsink.policies.scripted import pipeline_planner, energy_planner, signal_hunt_planner
from syncorsink.policies.oracle import (
    pipeline_oracle,
    pipeline_oracle_strong,
    energy_oracle,
    energy_oracle_strong,
    signal_hunt_oracle,
    signal_hunt_oracle_strong,
)
from syncorsink.policies.comm_wrapper import wrap_oracle_with_comm
from syncorsink.policies.local_oracle import (
    local_oracle,
    local_oracle_comm,
    local_oracle_plus,
    local_oracle_plus_comm,
    local_oracle_team_comm,
    local_pipeline_policy,
    local_energy_policy,
    local_signal_policy,
)
from syncorsink.policies.planner import (
    pipeline_central_planner,
    energy_central_planner,
    signal_hunt_central_planner,
)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="signal_hunt")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--split", default=None)
    parser.add_argument("--variant", type=int, default=0)
    parser.add_argument(
        "--policy",
        default="random",
        choices=[
            "random",
            "heuristic",
            "scripted",
            "oracle",
            "oracle_strong",
            "oracle_comm",
            "local_oracle",
            "local_oracle_comm",
            "local_oracle_plus",
            "local_oracle_plus_comm",
            "local_oracle_team_comm",
            "local_pipeline",
            "local_energy",
            "local_signal",
            "pipeline_planner",
            "pipeline_planner_comm",
            "pipeline_planner_follower",
            "pipeline_planner_comm_followers",
            "pipeline_planner_comm_followers_regions",
            "pipeline_planner_dispatcher",
            "pipeline_planner_semidec",
            "energy_planner",
            "signal_hunt_planner",
            "energy_planner_comm",
            "signal_hunt_planner_comm",
        ],
    )
    parser.add_argument("--energy-preset", default="hard", choices=["easy", "hard"])
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="syncorsink")
    parser.add_argument("--wandb-run", default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render-fps", type=float, default=10.0)
    args = parser.parse_args()

    config = SyncOrSinkConfig(
        scenario=args.scenario,
        split=args.split,
        map_variant=args.variant,
        track="ctde" if "oracle" in args.policy else "dtde",
        energy_preset=args.energy_preset,
    )
    env = SyncOrSinkEnv(config, render_mode="human" if args.render else None)

    if args.policy == "random":
        policy = random_policy(env.action_space, env.num_agents)
    elif args.policy == "heuristic":
        policy = heuristic_policy(env)
    elif args.policy == "scripted":
        if args.scenario == "pipeline_assembly":
            policy = pipeline_planner(env)
        elif args.scenario == "energy_grid":
            policy = energy_planner(env)
        else:
            policy = signal_hunt_planner(env)
    elif args.policy == "oracle_strong":
        if args.scenario == "pipeline_assembly":
            policy = pipeline_oracle_strong(env)
        elif args.scenario == "energy_grid":
            policy = energy_oracle_strong(env)
        else:
            policy = signal_hunt_oracle_strong(env)
    elif args.policy == "oracle_comm":
        if args.scenario == "pipeline_assembly":
            base = pipeline_oracle_strong(env)
        elif args.scenario == "energy_grid":
            base = energy_oracle_strong(env)
        else:
            base = signal_hunt_oracle_strong(env)
        policy = wrap_oracle_with_comm(base, env)
    elif args.policy == "local_oracle":
        policy = local_oracle(env)
    elif args.policy == "local_oracle_comm":
        policy = local_oracle_comm(env)
    elif args.policy == "local_oracle_plus":
        policy = local_oracle_plus(env)
    elif args.policy == "local_oracle_plus_comm":
        policy = local_oracle_plus_comm(env)
    elif args.policy == "local_oracle_team_comm":
        policy = local_oracle_team_comm(env)
    elif args.policy == "local_pipeline":
        policy = local_pipeline_policy(env)
    elif args.policy == "local_energy":
        policy = local_energy_policy(env)
    elif args.policy == "local_signal":
        policy = local_signal_policy(env)
    elif args.policy == "pipeline_planner":
        policy = pipeline_central_planner(env)
    elif args.policy == "pipeline_planner_comm":
        policy = pipeline_planner_comm(env)
    elif args.policy == "pipeline_planner_follower":
        policy = pipeline_planner_follower(env)
    elif args.policy == "pipeline_planner_comm_followers":
        policy = pipeline_planner_comm_followers(env)
    elif args.policy == "pipeline_planner_comm_followers_regions":
        policy = pipeline_planner_comm_followers_regions(env)
    elif args.policy == "pipeline_planner_dispatcher":
        policy = pipeline_planner_dispatcher(env)
    elif args.policy == "pipeline_planner_semidec":
        policy = pipeline_planner_semidec(env)
    elif args.policy == "energy_planner":
        policy = energy_central_planner(env)
    elif args.policy == "signal_hunt_planner":
        policy = signal_hunt_central_planner(env)
    elif args.policy == "energy_planner_comm":
        policy = energy_planner_comm(env)
    elif args.policy == "signal_hunt_planner_comm":
        policy = signal_hunt_planner_comm(env)
    else:
        if args.scenario == "pipeline_assembly":
            policy = pipeline_oracle(env)
        elif args.scenario == "energy_grid":
            policy = energy_oracle(env)
        else:
            policy = signal_hunt_oracle(env)

    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_run, config={
            "scenario": args.scenario,
            "episodes": args.episodes,
            "split": args.split,
            "variant": args.variant,
            "policy": args.policy,
        })

    def _log_episode(ep_idx, ep_stats):
        if wandb_run is None:
            return
        data = {
            "episode": ep_idx,
            "ep_return": ep_stats.total_reward,
            "ep_steps": ep_stats.steps,
            "ep_success": 1.0 if ep_stats.success else 0.0,
            "ep_comm_tokens": ep_stats.comm_tokens,
        }
        for aid, r in ep_stats.per_agent_reward.items():
            data[f"ep_agent_{aid}_return"] = r
        for aid, c in ep_stats.per_agent_comm.items():
            data[f"ep_agent_{aid}_comm"] = c
        wandb_run.log(data)

    summary, episodes = run_episodes(
        env,
        policy,
        episodes=args.episodes,
        seed=0,
        per_episode_cb=_log_episode,
        render=args.render,
        render_fps=args.render_fps,
    )
    print("episodes", summary.episodes)
    print("success_rate", summary.success_rate)
    print("avg_return", summary.avg_return)
    print("avg_steps", summary.avg_steps)
    print("avg_comm_tokens", summary.avg_comm_tokens)
    print("avg_agent_return", summary.avg_agent_reward)
    print("avg_agent_comm", summary.avg_agent_comm)

    if wandb_run is not None:
        data = {
            "success_rate": summary.success_rate,
            "avg_return": summary.avg_return,
            "avg_steps": summary.avg_steps,
            "avg_comm_tokens": summary.avg_comm_tokens,
        }
        for aid, r in summary.avg_agent_reward.items():
            data[f"avg_agent_{aid}_return"] = r
        for aid, c in summary.avg_agent_comm.items():
            data[f"avg_agent_{aid}_comm"] = c
        wandb_run.log(data)
        wandb_run.finish()


if __name__ == "__main__":
    main()
