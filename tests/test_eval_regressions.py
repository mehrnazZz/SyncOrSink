import json

import pytest


def test_episode_success_distinguishes_energy_failure():
    from syncorsink.eval.success import episode_success

    assert episode_success("energy_grid", True, {"success": False}) is False
    assert episode_success("energy_grid", True, {"success": True}) is True
    assert episode_success("signal_hunt", True, {"success": False}) is True
    assert episode_success("pipeline_assembly", False, {"success": True}) is False


def test_run_episodes_uses_energy_success_flag():
    from syncorsink.eval.runner import run_episodes

    class Config:
        scenario = "energy_grid"

    class EnergyFailureEnv:
        num_agents = 1
        config = Config()

        def reset(self, seed=None):
            return {0: {}}, {}

        def step(self, actions):
            return {0: {}}, {0: 0.0}, True, False, {"success": False}

    def policy(obs, info, state):
        return {0: {"action": 4, "message_tokens": []}}

    summary, episodes = run_episodes(EnergyFailureEnv(), policy, episodes=1, seed=0)

    assert summary.success_rate == 0.0
    assert episodes[0].success is False


def test_eval_spec_loads_extended_benchmark_fields(tmp_path):
    from syncorsink.eval.spec import load_spec
    from syncorsink.eval.spec_validate import validate_spec

    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps({
        "scenario": "energy_grid",
        "mode": "marl",
        "episodes": 2,
        "agents": 4,
        "map_size": 12,
        "max_steps": 90,
        "fov_preset": "easy",
        "comm_mode": "tokens",
        "track": "ctde",
        "energy_preset": "easy",
        "policy": "comm_mat",
        "policy_checkpoint": "checkpoints/comm_mat_energy.pt",
        "comm_mat_deterministic": False,
        "comm_mat_send_threshold": 0.25,
    }))

    spec = load_spec(str(spec_path))

    assert spec.num_agents == 4
    assert spec.map_size == 12
    assert spec.max_steps == 90
    assert spec.track == "ctde"
    assert spec.energy_preset == "easy"
    assert spec.policy_checkpoint == "checkpoints/comm_mat_energy.pt"
    assert spec.comm_mat_deterministic is False
    assert spec.comm_mat_send_threshold == 0.25
    with pytest.raises(Exception):
        validate_spec({"scenario": "energy_grid", "mode": "marl", "max_steps": 0})


def test_benchmark_policy_dispatch_no_random_fallback_and_pipeline_follower_runs():
    from examples.benchmark_run import build_policy
    from syncorsink.envs import SyncOrSinkConfig, SyncOrSinkEnv
    from syncorsink.eval.runner import run_episodes

    env = SyncOrSinkEnv(SyncOrSinkConfig(
        scenario="pipeline_assembly",
        map_size=8,
        num_agents=3,
        fov_preset="easy",
        max_steps=5,
    ))

    with pytest.raises(ValueError):
        build_policy({"policy": "missing_policy"}, env)

    policy = build_policy({"policy": "pipeline_planner_follower"}, env)
    summary, _ = run_episodes(env, policy, episodes=1, seed=0)

    assert summary.episodes == 1


def test_eval_from_spec_policy_dispatch_rejects_unknown_policy():
    from examples.eval_from_spec import build_marl_policy
    from syncorsink.envs import SyncOrSinkConfig, SyncOrSinkEnv
    from syncorsink.eval.spec import EvalSpec

    spec = EvalSpec(
        scenario="signal_hunt",
        split=None,
        episodes=1,
        map_variant=0,
        policy="missing_policy",
        mode="marl",
    )
    env = SyncOrSinkEnv(SyncOrSinkConfig(scenario=spec.scenario, max_steps=5))

    with pytest.raises(ValueError):
        build_marl_policy(spec, env)


def test_comm_mat_checkpoint_load_is_deferred_until_model_build(monkeypatch):
    from syncorsink.envs import SyncOrSinkConfig, SyncOrSinkEnv
    from syncorsink.policies.comm_mat_policy import CommMATPolicy, CommMATPolicyConfig

    load_calls = []

    def fake_load_checkpoint(self, path):
        assert self.model is not None
        assert self._built is True
        load_calls.append(path)

    monkeypatch.setattr(CommMATPolicy, "_load_checkpoint", fake_load_checkpoint)

    env = SyncOrSinkEnv(SyncOrSinkConfig(
        scenario="signal_hunt",
        map_size=8,
        num_agents=2,
        fov_preset="easy",
        max_steps=5,
        comm_mode="tokens",
    ))
    obs, info = env.reset(seed=0)
    policy = CommMATPolicy(
        config=CommMATPolicyConfig(
            comm_vocab_size=16,
            comm_token_limit=4,
            max_messages=4,
            hidden_dim=16,
            n_heads=2,
            n_layers=1,
        ),
        checkpoint="relative_checkpoint.pt",
    )

    assert load_calls == []

    actions = policy(obs, info, {"step": 0})
    policy(obs, info, {"step": 1})

    assert load_calls == ["relative_checkpoint.pt"]
    assert set(actions) == set(obs)
