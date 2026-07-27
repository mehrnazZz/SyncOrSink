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


def test_pipeline_curriculum_knobs_control_stage_shape():
    config = SyncOrSinkConfig(
        scenario="pipeline_assembly",
        map_size=8,
        num_agents=3,
        fov_preset="easy",
        pipeline_stage_count=2,
        pipeline_required_per_stage_min=1,
        pipeline_required_per_stage_max=1,
        pipeline_sync_probability=0.0,
        pipeline_dependency_probability=0.0,
    )
    env = SyncOrSinkEnv(config)
    env.reset(seed=0)

    stages = env.scenario_state.data["stages"]

    assert len(stages) == 2
    assert [len(stage["required"]) for stage in stages] == [1, 1]
    assert all(not stage["sync"] for stage in stages)
    assert all(stage["deps"] == [] for stage in stages)


def test_pipeline_curriculum_knobs_can_force_sync_and_dependencies():
    config = SyncOrSinkConfig(
        scenario="pipeline_assembly",
        map_size=8,
        num_agents=3,
        fov_preset="easy",
        pipeline_stage_count=3,
        pipeline_required_per_stage_min=1,
        pipeline_required_per_stage_max=1,
        pipeline_sync_probability=1.0,
        pipeline_dependency_probability=1.0,
    )
    env = SyncOrSinkEnv(config)
    env.reset(seed=1)

    stages = env.scenario_state.data["stages"]

    assert len(stages) == 3
    assert all(stage["sync"] for stage in stages)
    assert stages[0]["deps"] == []
    assert all(stage["deps"] for stage in stages[1:])
