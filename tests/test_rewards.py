import numpy as np

from syncorsink.envs import SyncOrSinkEnv, SyncOrSinkConfig


def test_comm_cost_applied():
    config = SyncOrSinkConfig(scenario="pipeline_assembly", comm_mode="tokens", comm_token_limit=4)
    env = SyncOrSinkEnv(config)
    env.reset(seed=0)
    actions = {i: {"action": env.ACTION_STAY, "message_tokens": [1, 2, 3]} for i in range(env.num_agents)}
    obs, rewards, done, truncated, info = env.step(actions)
    # each agent should be charged comm_cost * tokens
    for i in range(env.num_agents):
        assert rewards[i] <= 0
        assert info["comm_tokens"][i] == 3
