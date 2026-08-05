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


def test_resource_pickup_emits_training_event():
    config = SyncOrSinkConfig(
        scenario="pipeline_assembly",
        map_size=8,
        num_agents=3,
        fov_preset="easy",
        pipeline_stage_count=1,
        pipeline_required_per_stage_min=1,
        pipeline_required_per_stage_max=1,
        pipeline_sync_probability=0.0,
        pipeline_dependency_probability=0.0,
    )
    env = SyncOrSinkEnv(config)
    env.reset(seed=0)
    resource_pos, resource_type = next(iter(env.scenario_state.data["resource_types"].items()))
    env.agent_positions[0] = resource_pos

    _, _, _, _, info = env.step({
        0: {"action": env.ACTION_PICKUP},
        1: {"action": env.ACTION_STAY},
        2: {"action": env.ACTION_STAY},
    })

    assert {"event": "picked_resource", "resource_type": int(resource_type)} in info["events"][0]


def test_pipeline_stage_completion_emits_training_events():
    config = SyncOrSinkConfig(
        scenario="pipeline_assembly",
        map_size=8,
        num_agents=3,
        fov_preset="easy",
        pipeline_stage_count=1,
        pipeline_required_per_stage_min=1,
        pipeline_required_per_stage_max=1,
        pipeline_sync_probability=0.0,
        pipeline_dependency_probability=0.0,
    )
    env = SyncOrSinkEnv(config)
    env.reset(seed=0)
    stage = env.scenario_state.data["stages"][0]
    resource_type = int(stage["required"][0])
    env.agent_positions[0] = stage["station"]
    env.inventories[0] = resource_type

    _, _, done, _, info = env.step({
        0: {"action": env.ACTION_INTERACT},
        1: {"action": env.ACTION_STAY},
        2: {"action": env.ACTION_STAY},
    })

    assert done is True
    assert {"event": "delivered", "stage": 0} in info["events"][0]
    assert {"event": "stage_completed", "stage": 0} in info["events"][0]
    assert all({"event": "pipeline_complete"} in events for events in info["events"].values())


def test_pipeline_wrong_delivery_and_sync_wait_emit_focus_events():
    wrong_config = SyncOrSinkConfig(
        scenario="pipeline_assembly",
        map_size=8,
        num_agents=3,
        fov_preset="easy",
        pipeline_stage_count=1,
        pipeline_required_per_stage_min=1,
        pipeline_required_per_stage_max=1,
        pipeline_sync_probability=0.0,
        pipeline_dependency_probability=0.0,
    )
    wrong_env = SyncOrSinkEnv(wrong_config)
    wrong_env.reset(seed=0)
    wrong_stage = wrong_env.scenario_state.data["stages"][0]
    wrong_env.agent_positions[0] = wrong_stage["station"]
    wrong_env.inventories[0] = 99

    _, _, _, _, wrong_info = wrong_env.step({
        0: {"action": wrong_env.ACTION_INTERACT},
        1: {"action": wrong_env.ACTION_STAY},
        2: {"action": wrong_env.ACTION_STAY},
    })

    assert {"event": "pipeline_wrong_delivery", "resource_type": 99} in wrong_info["events"][0]

    sync_config = SyncOrSinkConfig(
        scenario="pipeline_assembly",
        map_size=8,
        num_agents=3,
        fov_preset="easy",
        pipeline_stage_count=1,
        pipeline_required_per_stage_min=1,
        pipeline_required_per_stage_max=1,
        pipeline_sync_probability=1.0,
        pipeline_dependency_probability=0.0,
    )
    sync_env = SyncOrSinkEnv(sync_config)
    sync_env.reset(seed=0)
    sync_stage = sync_env.scenario_state.data["stages"][0]
    sync_stage["delivered"] = list(sync_stage["required"])
    sync_env.agent_positions[0] = sync_stage["station"]

    _, _, done, _, sync_info = sync_env.step({
        0: {"action": sync_env.ACTION_INTERACT},
        1: {"action": sync_env.ACTION_STAY},
        2: {"action": sync_env.ACTION_STAY},
    })

    assert done is False
    assert {"event": "pipeline_sync_wait", "stage": 0} in sync_info["events"][0]


def test_pipeline_dependency_blocked_emits_focus_event():
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
    stages[0]["done"] = False
    stages[1]["deps"] = [0]
    stages[1]["delivered"] = list(stages[1]["required"])
    env.agent_positions[0] = stages[1]["station"]

    _, _, done, _, info = env.step({
        0: {"action": env.ACTION_INTERACT},
        1: {"action": env.ACTION_STAY},
        2: {"action": env.ACTION_STAY},
    })

    assert done is False
    assert {"event": "pipeline_dependency_blocked", "stage": 1} in info["events"][0]
