import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from syncorsink.envs import SyncOrSinkConfig
from syncorsink.envs.pz_wrapper import SyncOrSinkParallel
from pettingzoo.test import parallel_api_test

def main():
    env = SyncOrSinkParallel(SyncOrSinkConfig(), render_mode="ansi")
    # parallel_api_test(env, num_cycles=1000)
    obs, info = env.reset(seed=0)
    done = False
    truncated = False
    while not (done or truncated):
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, rewards, terminations, truncations, infos = env.step(actions)
        done = all(terminations.values()) if terminations else True
        truncated = all(truncations.values()) if truncations else True
        print(env.render())
    env.close()


if __name__ == "__main__":
    main()
