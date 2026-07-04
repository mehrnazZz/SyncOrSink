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
        "energy_private_monitor": True,
        "policy": "comm_mat",
        "policy_entrypoint": "my_package.agent:build_policy",
        "policy_kwargs": {"temperature": 0.0},
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
    assert spec.energy_private_monitor is True
    assert spec.policy_entrypoint == "my_package.agent:build_policy"
    assert spec.policy_kwargs == {"temperature": 0.0}
    assert spec.policy_checkpoint == "checkpoints/comm_mat_energy.pt"
    assert spec.comm_mat_deterministic is False
    assert spec.comm_mat_send_threshold == 0.25
    with pytest.raises(Exception):
        validate_spec({"scenario": "energy_grid", "mode": "marl", "max_steps": 0})
    with pytest.raises(Exception):
        validate_spec({"scenario": "energy_grid", "mode": "marl", "policy_kwargs": []})


def test_official_benchmark_v0_1_loads():
    from syncorsink.eval.benchmark_spec import load_benchmark

    bench = load_benchmark("benchmarks/syncorsink_v0_1.json")

    assert bench.name == "syncorsink_v0_1"
    assert bench.version == "0.1.0"
    assert len(bench.cases) == 6
    assert all(case.weight > 0 for case in bench.cases)
    assert any("communication_required" in case.tags for case in bench.cases)


def test_leaderboard_result_artifact_validates_saves_and_scores(tmp_path):
    from syncorsink.eval.metrics import EvalSummary
    from syncorsink.eval.result_schema import (
        SubmissionInfo,
        load_result_artifact,
        make_result_artifact,
        save_result_artifact,
        summary_to_case_result,
    )
    from syncorsink.eval.scoring import score_result_artifact

    summary = EvalSummary(
        episodes=2,
        success_rate=0.5,
        avg_return=3.0,
        avg_steps=20.0,
        avg_comm_tokens=4.0,
        avg_agent_reward={0: 1.5},
        avg_agent_comm={0: 2.0},
    )
    case = summary_to_case_result(
        "signal_hunt_8x8_private_clues",
        summary,
        spec={"scenario": "signal_hunt", "mode": "marl"},
        tags=["communication_required"],
        seeds=[3000, 3001],
    )
    artifact = make_result_artifact(
        benchmark_name="syncorsink_v0_1",
        benchmark_version="0.1.0",
        track="symbolic_dtde",
        submission=SubmissionInfo(
            name="smoke",
            method_name="test_policy",
            method_type="unit_test",
            authors=["SyncOrSink"],
        ),
        cases=[case],
        generated_at="2026-01-01T00:00:00+00:00",
    )
    artifact["score"] = score_result_artifact(artifact)
    out = tmp_path / "result.json"

    save_result_artifact(artifact, out)
    loaded = load_result_artifact(out)

    assert loaded["score"]["official_score"] == 50.0
    assert loaded["cases"][0]["metrics"]["avg_agent_reward"] == {"0": 1.5}
    with pytest.raises(ValueError):
        make_result_artifact(
            benchmark_name="syncorsink_v0_1",
            benchmark_version="0.1.0",
            track="missing_track",
            submission=artifact["submission"],
            cases=[case],
        )


def test_leaderboard_builder_collects_ranks_and_validates_manifest(tmp_path):
    from syncorsink.eval.benchmark_spec import load_benchmark
    from syncorsink.eval.leaderboard import (
        collect_leaderboard_entries,
        render_csv,
        render_json,
        render_markdown,
    )
    from syncorsink.eval.metrics import EvalSummary
    from syncorsink.eval.result_schema import (
        SubmissionInfo,
        make_result_artifact,
        save_result_artifact,
        summary_to_case_result,
    )

    bench = load_benchmark("benchmarks/syncorsink_v0_1.json")

    def write_artifact(name, success_rate):
        cases = []
        for case in bench.cases:
            summary = EvalSummary(
                episodes=32,
                success_rate=success_rate,
                avg_return=success_rate * 10.0,
                avg_steps=100.0,
                avg_comm_tokens=5.0,
                avg_agent_reward={0: 1.0},
                avg_agent_comm={0: 1.0},
            )
            cases.append(summary_to_case_result(case.name, summary, spec=case.spec, weight=case.weight, tags=case.tags))
        artifact = make_result_artifact(
            benchmark_name=bench.name,
            benchmark_version=bench.version,
            track="symbolic_dtde",
            submission=SubmissionInfo(
                name=name,
                method_name=name,
                method_type="unit_test",
                authors=["SyncOrSink"],
            ),
            cases=cases,
            generated_at="2026-01-01T00:00:00+00:00",
        )
        path = tmp_path / f"{name}.json"
        save_result_artifact(artifact, path)
        return path

    weaker = write_artifact("weaker", 0.25)
    stronger = write_artifact("stronger", 0.75)

    collection = collect_leaderboard_entries([tmp_path], benchmark=bench)
    markdown = render_markdown(collection.entries, benchmark_name=bench.name, benchmark_version=bench.version)
    csv_text = render_csv(collection.entries)
    json_text = render_json(collection.entries)

    assert [entry.submission_name for entry in collection.entries] == ["stronger", "weaker"]
    assert collection.entries[0].official_score == 75.0
    assert "| 1 | stronger | stronger |" in markdown
    assert "official_score" in csv_text
    assert "\r\n" not in csv_text
    assert '"submission_name": "stronger"' in json_text

    partial_dir = tmp_path / "partial"
    partial_dir.mkdir()
    partial_artifact = make_result_artifact(
        benchmark_name=bench.name,
        benchmark_version=bench.version,
        track="symbolic_dtde",
        submission=SubmissionInfo(
            name="partial",
            method_name="partial",
            method_type="unit_test",
            authors=["SyncOrSink"],
        ),
        cases=[
            summary_to_case_result(
                bench.cases[0].name,
                EvalSummary(1, 0.0, 0.0, 1.0, 0.0, {}, {}),
                spec=bench.cases[0].spec,
            )
        ],
        generated_at="2026-01-01T00:00:00+00:00",
    )
    partial_path = partial_dir / "partial.json"
    save_result_artifact(partial_artifact, partial_path)

    with pytest.raises(ValueError):
        collect_leaderboard_entries([partial_path], benchmark=bench)
    allowed = collect_leaderboard_entries([partial_path], benchmark=bench, allow_partial=True)
    assert allowed.entries[0].submission_name == "partial"

    weaker.unlink()
    stronger.unlink()


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


def test_external_policy_entrypoint_runs_and_resets(tmp_path, monkeypatch):
    import importlib

    module_path = tmp_path / "external_submission.py"
    module_path.write_text(
        """
RESET_CALLS = 0
CHECKPOINTS = []
BUILD_SPECS = []
TEMPERATURES = []
ENV_VIEW_HAS_AGENT_POSITIONS = None
OBS_KEYS = []
INFO_MESSAGE_KEYS = []


class SubmittedPolicy:
    def __init__(self, env, spec, temperature=1.0):
        global ENV_VIEW_HAS_AGENT_POSITIONS
        self.num_agents = env.num_agents
        ENV_VIEW_HAS_AGENT_POSITIONS = hasattr(env, "agent_positions")
        BUILD_SPECS.append(dict(spec))
        TEMPERATURES.append(temperature)

    def load_checkpoint(self, path):
        CHECKPOINTS.append(path)

    def reset(self, episode=None, seed=None):
        global RESET_CALLS
        RESET_CALLS += 1

    def metadata(self):
        return {"method_name": "SubmittedPolicy"}

    def act_agent(self, agent_id, obs, info, state):
        OBS_KEYS.append(sorted(obs.keys()))
        INFO_MESSAGE_KEYS.append(sorted(info.get("messages_text", {}).keys()))
        return {
            "action": 4,
            "message_tokens": [],
            "message_text": "",
        }


def build_policy(env, spec, temperature=1.0):
    return SubmittedPolicy(env, spec, temperature=temperature)
""",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    from examples.benchmark_run import build_policy
    from syncorsink.envs import SyncOrSinkConfig, SyncOrSinkEnv
    from syncorsink.eval.runner import run_episodes

    env = SyncOrSinkEnv(SyncOrSinkConfig(
        scenario="signal_hunt",
        map_size=8,
        num_agents=2,
        fov_preset="easy",
        max_steps=2,
    ))
    spec = {"scenario": "signal_hunt", "mode": "marl", "policy": "random"}
    policy = build_policy(
        spec,
        env,
        external_entrypoint="external_submission:build_policy",
        external_checkpoint="checkpoint.pt",
        external_kwargs={"temperature": 0.25},
    )
    summary, _ = run_episodes(env, policy, episodes=2, seed=0)
    external_submission = importlib.import_module("external_submission")

    assert summary.episodes == 2
    assert external_submission.RESET_CALLS == 2
    assert external_submission.CHECKPOINTS == ["checkpoint.pt"]
    assert external_submission.BUILD_SPECS[0]["scenario"] == "signal_hunt"
    assert external_submission.TEMPERATURES == [0.25]
    assert external_submission.ENV_VIEW_HAS_AGENT_POSITIONS is False
    assert external_submission.OBS_KEYS
    assert all("local_grid" in keys for keys in external_submission.OBS_KEYS)
    assert all(keys in ([0], [1]) for keys in external_submission.INFO_MESSAGE_KEYS)


def test_external_dtde_rejects_team_policy_by_default(tmp_path, monkeypatch):
    module_path = tmp_path / "centralized_submission.py"
    module_path.write_text(
        """
class TeamPolicy:
    def act(self, obs, info, state):
        return {int(agent_id): {"action": 4, "message_tokens": []} for agent_id in obs}


def build_policy(env, spec):
    return TeamPolicy()
""",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    from examples.benchmark_run import build_policy
    from syncorsink.envs import SyncOrSinkConfig, SyncOrSinkEnv
    from syncorsink.eval.runner import run_episodes

    env = SyncOrSinkEnv(SyncOrSinkConfig(scenario="signal_hunt", map_size=8, num_agents=2, fov_preset="easy", max_steps=1))
    policy = build_policy(
        {"scenario": "signal_hunt", "mode": "marl", "policy": "random"},
        env,
        external_entrypoint="centralized_submission:build_policy",
    )

    with pytest.raises(TypeError, match="decentralized external policies"):
        run_episodes(env, policy, episodes=1, seed=0)


def test_external_centralized_debug_escape_hatch(tmp_path, monkeypatch):
    module_path = tmp_path / "centralized_debug_submission.py"
    module_path.write_text(
        """
class TeamPolicy:
    def act(self, obs, info, state):
        return {int(agent_id): {"action": 4, "message_tokens": []} for agent_id in obs}


def build_policy(env, spec):
    return TeamPolicy()
""",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    from examples.benchmark_run import build_policy
    from syncorsink.envs import SyncOrSinkConfig, SyncOrSinkEnv
    from syncorsink.eval.runner import run_episodes

    env = SyncOrSinkEnv(SyncOrSinkConfig(scenario="signal_hunt", map_size=8, num_agents=2, fov_preset="easy", max_steps=1))
    policy = build_policy(
        {"scenario": "signal_hunt", "mode": "marl", "policy": "random"},
        env,
        external_entrypoint="centralized_debug_submission:build_policy",
        decentralized_external=False,
    )
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
