import pytest


def _run_expert(factory, *, scenario, episodes, seed=0, **config_kwargs):
    from syncorsink.envs import SyncOrSinkConfig, SyncOrSinkEnv
    from syncorsink.eval.runner import run_episodes

    config = SyncOrSinkConfig(scenario=scenario, track="ctde", **config_kwargs)
    env = SyncOrSinkEnv(config)
    policy = factory(env)
    return run_episodes(env, policy, episodes=episodes, seed=seed)


@pytest.mark.parametrize(
    "scenario,policy_path,episodes,config_kwargs",
    [
        (
            "signal_hunt",
            "syncorsink.policies.planner_comm:signal_hunt_planner_comm",
            8,
            {"map_size": 8, "num_agents": 2, "fov_preset": "easy", "max_steps": 120},
        ),
        (
            "signal_hunt",
            "syncorsink.policies.planner_comm:signal_hunt_planner_comm",
            8,
            {"map_size": 16, "num_agents": 4, "fov_preset": "medium", "max_steps": 180},
        ),
        (
            "energy_grid",
            "syncorsink.policies.oracle:energy_oracle_planner",
            8,
            {
                "map_size": 8,
                "num_agents": 3,
                "fov_preset": "easy",
                "energy_preset": "easy",
                "max_steps": 180,
            },
        ),
        (
            "energy_grid",
            "syncorsink.policies.oracle:energy_oracle_planner",
            8,
            {
                "map_size": 16,
                "num_agents": 4,
                "fov_preset": "medium",
                "energy_preset": "hard",
                "max_steps": 300,
            },
        ),
        (
            "pipeline_assembly",
            "syncorsink.policies.planner_comm:pipeline_planner_comm",
            8,
            {"map_size": 8, "num_agents": 3, "fov_preset": "easy", "max_steps": 180},
        ),
        (
            "pipeline_assembly",
            "syncorsink.policies.planner_comm:pipeline_planner_comm",
            32,
            {"map_size": 16, "num_agents": 4, "fov_preset": "medium", "max_steps": 300},
        ),
    ],
)
def test_core_scenarios_have_expert_acceptance(scenario, policy_path, episodes, config_kwargs):
    from syncorsink.policies.submission import import_entrypoint

    summary, _ = _run_expert(
        import_entrypoint(policy_path),
        scenario=scenario,
        episodes=episodes,
        **config_kwargs,
    )

    assert summary.success_rate == 1.0


@pytest.mark.parametrize("scenario", ["signal_hunt", "energy_grid", "pipeline_assembly"])
def test_generated_scenarios_pass_solvability_checks(scenario):
    from syncorsink.envs import SyncOrSinkConfig, SyncOrSinkEnv
    from syncorsink.eval.solvability import check_solvability

    for seed in range(8):
        config = SyncOrSinkConfig(scenario=scenario, map_size=16, num_agents=4, fov_preset="medium")
        env = SyncOrSinkEnv(config)
        env.reset(seed=seed)

        ok, reason = check_solvability(env)

        assert ok, f"{scenario} seed={seed}: {reason}"
