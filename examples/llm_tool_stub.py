import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from syncorsink.envs import SyncOrSinkEnv, SyncOrSinkConfig
from syncorsink.llm.tool_policy import ToolCallingPolicy
from syncorsink.eval.runner import run_episodes


def dummy_tool_llm(prompt: str):
    # Example tool call output
    return [
        {"name": "move", "arguments": {"direction": "stay"}},
        {"name": "message", "arguments": {"text": "holding position"}},
    ]


def main():
    env = SyncOrSinkEnv(SyncOrSinkConfig(scenario="signal_hunt"))
    policy = ToolCallingPolicy(dummy_tool_llm)
    summary, _ = run_episodes(env, policy, episodes=2, seed=0)
    print("success_rate", summary.success_rate)


if __name__ == "__main__":
    main()
