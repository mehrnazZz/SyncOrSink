import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from syncorsink.envs import SyncOrSinkEnv, SyncOrSinkConfig
from syncorsink.human_control import HumanController


def main():
    config = SyncOrSinkConfig(
        scenario="signal_hunt",
        map_size=16,
        num_agents=3,
        fov_preset="medium",
        comm_mode="text",
        use_rooms=True,
        use_doors=True,
        enable_fog_of_war=True,
        signal_decoy_count=3,
        render_split_view=True,
    )
    env = SyncOrSinkEnv(config, render_mode="human")
    controller = HumanController(env, human_agent_id=0)
    obs, info = env.reset(seed=0)

    done = False
    truncated = False
    while not (done or truncated):
        actions = controller.collect_actions()
        obs, rewards, done, truncated, info = env.step(actions)
        env.render()
        time.sleep(0.02)

    env.close()


if __name__ == "__main__":
    main()
