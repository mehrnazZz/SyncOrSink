import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from syncorsink.envs import SyncOrSinkConfig, SyncOrSinkVector


def main():
    venv = SyncOrSinkVector(num_envs=2, config=SyncOrSinkConfig())
    obs, infos = venv.reset(seed=0)
    done = False
    truncated = False
    while not (done or truncated):
        actions = []
        for env_idx in range(2):
            env_actions = {}
            for agent_id in range(3):
                env_actions[agent_id] = {"action": 4, "message_tokens": []}
            actions.append(env_actions)
        obs, rewards, done, truncated, infos = venv.step(actions)
        print("step", done, truncated)


if __name__ == "__main__":
    main()
