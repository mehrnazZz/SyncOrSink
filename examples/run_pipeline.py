import os
import sys
import time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from syncorsink.envs import SyncOrSinkEnv, SyncOrSinkConfig


def main():
    config = SyncOrSinkConfig(
        scenario="pipeline_assembly",
        map_size=16,
        num_agents=3,
        fov_preset="medium",
        comm_mode="tokens",
        use_rooms=True,
        use_doors=True,
        enable_fog_of_war=True,
    )
    env = SyncOrSinkEnv(config, render_mode="human")
    obs, info = env.reset(seed=0)
    done = False
    truncated = False
    while not (done or truncated):
        actions = {}
        for agent_id in range(env.num_agents):
            action_id = np.random.randint(0, 8)
            actions[agent_id] = {"action": int(action_id), "message_tokens": []}
        obs, rewards, done, truncated, info = env.step(actions)
        env.render()
        time.sleep(0.05)
    env.close()


if __name__ == "__main__":
    main()
