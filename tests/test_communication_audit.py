import pytest


PRIVATE_SHARED_INFO_KEYS = {
    "agent_clues",
    "agent_hints",
    "agent_hint_specs",
    "agent_nodes",
    "clue_specs",
    "constraints",
    "full_plan",
    "goal_hint_texts",
    "hints",
    "node_assignments",
    "node_energy",
    "node_types",
    "stages",
}


def assert_no_private_shared_info(info: dict):
    leaked = PRIVATE_SHARED_INFO_KEYS & set(info)
    assert leaked == set()
    assert "central_obs" not in info


@pytest.mark.parametrize(
    "scenario,kwargs",
    [
        ("pipeline_assembly", {"num_agents": 3, "fov_preset": "easy"}),
        ("energy_grid", {"num_agents": 3, "fov_preset": "easy", "energy_preset": "easy"}),
        ("signal_hunt", {"num_agents": 3, "fov_preset": "easy"}),
    ],
)
def test_dtde_shared_info_does_not_expose_private_scenario_state(scenario, kwargs):
    from syncorsink.envs import SyncOrSinkConfig, SyncOrSinkEnv

    env = SyncOrSinkEnv(SyncOrSinkConfig(scenario=scenario, map_size=8, track="dtde", **kwargs))
    _, info = env.reset(seed=0)
    assert_no_private_shared_info(info)

    actions = {i: {"action": env.ACTION_STAY} for i in range(env.num_agents)}
    _, _, _, _, info = env.step(actions)
    assert_no_private_shared_info(info)


def test_signal_hunt_private_clue_is_encoded_per_agent_not_shared_info():
    from syncorsink.envs import SyncOrSinkConfig, SyncOrSinkEnv

    env = SyncOrSinkEnv(SyncOrSinkConfig(scenario="signal_hunt", map_size=8, num_agents=3, fov_preset="easy"))
    obs, info = env.reset(seed=0)

    assert_no_private_shared_info(info)
    assert all(int(agent_obs["goal_hint"][0]) in {21, 22, 23, 24, 25} for agent_obs in obs.values())
    assert len({tuple(agent_obs["goal_hint"]) for agent_obs in obs.values()}) > 1


def test_signal_hunt_clue_collection_does_not_emit_private_clue_text_in_info():
    from syncorsink.envs import SyncOrSinkConfig, SyncOrSinkEnv

    env = SyncOrSinkEnv(SyncOrSinkConfig(scenario="signal_hunt", map_size=8, num_agents=2, fov_preset="easy"))
    env.reset(seed=2)
    clue_pos = env.meta["clues"][0]
    env.agent_positions[0] = clue_pos

    actions = {i: {"action": env.ACTION_STAY} for i in range(env.num_agents)}
    actions[0] = {"action": env.ACTION_INTERACT}
    obs, _, _, _, info = env.step(actions)

    assert_no_private_shared_info(info)
    assert info["events"][0] == [{"event": "clue_found"}]
    assert all("clue" not in event for events in info["events"].values() for event in events)
    assert int(obs[0]["goal_hint"][0]) in {21, 22, 23, 24, 25}


def test_energy_grid_private_critical_event_is_not_broadcast():
    from syncorsink.envs import SyncOrSinkConfig, SyncOrSinkEnv

    env = SyncOrSinkEnv(
        SyncOrSinkConfig(scenario="energy_grid", map_size=8, num_agents=3, fov_preset="easy", energy_preset="easy")
    )
    env.reset(seed=0)
    node_pos, assigned_agent = next(iter(env.scenario_state.data["node_assignments"].items()))
    env.scenario_state.data["node_energy"][node_pos] = env.scenario_state.data["sync_threshold"]

    actions = {i: {"action": env.ACTION_STAY} for i in range(env.num_agents)}
    _, _, _, _, info = env.step(actions)

    assert_no_private_shared_info(info)
    for agent_id, events in info["events"].items():
        critical = [event for event in events if event.get("event") == "node_critical" and event.get("node") == node_pos]
        if agent_id == assigned_agent:
            assert len(critical) == 1
        else:
            assert critical == []


def test_ctde_central_obs_is_explicit_track_only():
    from syncorsink.envs import SyncOrSinkConfig, SyncOrSinkEnv

    dtde_env = SyncOrSinkEnv(SyncOrSinkConfig(scenario="pipeline_assembly", map_size=8, track="dtde"))
    _, dtde_info = dtde_env.reset(seed=0)
    assert "central_obs" not in dtde_info

    ctde_env = SyncOrSinkEnv(SyncOrSinkConfig(scenario="pipeline_assembly", map_size=8, track="ctde"))
    _, ctde_info = ctde_env.reset(seed=0)
    assert "central_obs" in ctde_info
