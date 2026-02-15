import numpy as np

from syncorsink.envs import SyncOrSinkEnv, SyncOrSinkConfig


def test_map_seed_determinism_signal_hunt():
    config = SyncOrSinkConfig(scenario="signal_hunt", map_seed=123, map_variant=0)
    env1 = SyncOrSinkEnv(config)
    env2 = SyncOrSinkEnv(config)
    obs1, _ = env1.reset(seed=1)
    obs2, _ = env2.reset(seed=999)
    assert np.array_equal(env1.grid, env2.grid)


def test_split_seed_determinism_pipeline():
    config = SyncOrSinkConfig(scenario="pipeline_assembly", split="test", map_variant=5)
    env1 = SyncOrSinkEnv(config)
    env2 = SyncOrSinkEnv(config)
    env1.reset(seed=0)
    env2.reset(seed=0)
    assert np.array_equal(env1.grid, env2.grid)
