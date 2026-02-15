import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from syncorsink.envs import SyncOrSinkEnv, SyncOrSinkConfig
from syncorsink.eval.metrics import summarize
from syncorsink.eval.llm_runner import run_llm_episodes
from syncorsink.llm.policy import LLMPolicy
from syncorsink.llm.tool_policy import ToolCallingPolicy
from syncorsink.llm.cache import PromptCache


def dummy_llm(prompt: str):
    return '{"action": 4, "message_text": ""}'


def dummy_tool_llm(prompt: str):
    return [
        {"name": "move", "arguments": {"direction": "stay"}},
        {"name": "message", "arguments": {"text": "holding"}},
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="signal_hunt")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--split", default=None)
    parser.add_argument("--variant", type=int, default=0)
    parser.add_argument("--tool", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="syncorsink")
    parser.add_argument("--wandb-run", default=None)
    parser.add_argument("--cache", default=None, help="Path to prompt cache JSON")
    args = parser.parse_args()

    config = SyncOrSinkConfig(scenario=args.scenario, split=args.split, map_variant=args.variant, comm_mode="text", track="dtde")
    env = SyncOrSinkEnv(config)

    cache = PromptCache(path=args.cache) if args.cache else None

    if args.tool:
        policy = ToolCallingPolicy(dummy_tool_llm)
    else:
        policy = LLMPolicy(dummy_llm, cache=cache)

    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_run, config={
            "scenario": args.scenario,
            "episodes": args.episodes,
            "split": args.split,
            "variant": args.variant,
            "tool": args.tool,
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

    episodes = run_llm_episodes(env, policy, episodes=args.episodes, seed=0, per_episode_cb=_log_episode)
    summary = summarize(episodes)
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
