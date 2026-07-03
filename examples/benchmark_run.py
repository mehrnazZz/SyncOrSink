import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from syncorsink.envs import SyncOrSinkEnv, SyncOrSinkConfig
from syncorsink.eval.benchmark_spec import load_benchmark
from syncorsink.eval.metrics import summarize
from syncorsink.eval.result_schema import (
    SubmissionInfo,
    make_result_artifact,
    save_result_artifact,
    summary_to_case_result,
)
from syncorsink.eval.runner import run_episodes
from syncorsink.eval.scoring import score_result_artifact
from syncorsink.eval.llm_runner import run_llm_episodes
from syncorsink.policies.random_policy import random_policy
from syncorsink.policies.scripted import pipeline_planner, energy_planner, signal_hunt_planner
from syncorsink.policies.oracle import (
    pipeline_oracle,
    pipeline_oracle_strong,
    energy_oracle,
    energy_oracle_strong,
    energy_oracle_planner,
    signal_hunt_oracle,
    signal_hunt_oracle_strong,
)
from syncorsink.policies.comm_wrapper import wrap_oracle_with_comm
from syncorsink.policies.planner_comm import (
    pipeline_planner_comm,
    pipeline_planner_follower,
    pipeline_planner_comm_followers,
    pipeline_planner_comm_followers_regions,
    pipeline_planner_dispatcher,
    pipeline_planner_semidec,
    energy_planner_comm,
    signal_hunt_planner_comm,
)
from syncorsink.policies.comm_mat_policy import CommMATPolicy, CommMATPolicyConfig
from syncorsink.policies.submission import load_policy_entrypoint
from syncorsink.llm.policy import LLMPolicy


def dummy_llm(prompt: str) -> str:
    return '{"action": 4, "message_text": ""}'


def _parse_policy_kwargs(raw: str | None) -> dict:
    if not raw:
        return {}
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("--policy-kwargs must decode to a JSON object")
    return data


def build_policy(
    spec,
    env,
    external_entrypoint: str | None = None,
    external_checkpoint: str | None = None,
    external_kwargs: dict | None = None,
):
    entrypoint = external_entrypoint or spec.get("policy_entrypoint")
    if entrypoint:
        policy_kwargs = dict(spec.get("policy_kwargs", {}))
        policy_kwargs.update(external_kwargs or {})
        checkpoint = external_checkpoint if external_checkpoint is not None else spec.get("policy_checkpoint")
        return load_policy_entrypoint(
            entrypoint,
            env=env,
            spec=spec,
            checkpoint=checkpoint,
            kwargs=policy_kwargs,
        ).policy

    policy = spec.get("policy", "random")
    mode = spec.get("mode", "marl")
    if mode == "llm":
        return LLMPolicy(dummy_llm)
    if policy == "random":
        return random_policy(env.action_space, env.num_agents)
    if policy == "scripted":
        if env.config.scenario == "pipeline_assembly":
            return pipeline_planner(env)
        if env.config.scenario == "energy_grid":
            return energy_planner(env)
        return signal_hunt_planner(env)
    if policy == "oracle":
        if env.config.scenario == "pipeline_assembly":
            return pipeline_oracle(env)
        if env.config.scenario == "energy_grid":
            return energy_oracle(env)
        return signal_hunt_oracle(env)
    if policy == "oracle_strong":
        if env.config.scenario == "pipeline_assembly":
            return pipeline_oracle_strong(env)
        if env.config.scenario == "energy_grid":
            return energy_oracle_strong(env)
        return signal_hunt_oracle_strong(env)
    if policy == "oracle_planner":
        if env.config.scenario == "energy_grid":
            return energy_oracle_planner(env)
        if env.config.scenario == "pipeline_assembly":
            return pipeline_oracle_strong(env)
        return signal_hunt_oracle_strong(env)
    if policy == "oracle_comm":
        if env.config.scenario == "pipeline_assembly":
            base = pipeline_oracle_strong(env)
        elif env.config.scenario == "energy_grid":
            base = energy_oracle_strong(env)
        else:
            base = signal_hunt_oracle_strong(env)
        return wrap_oracle_with_comm(base, env)
    if policy == "pipeline_planner_comm":
        return pipeline_planner_comm(env)
    if policy == "pipeline_planner_follower":
        return pipeline_planner_follower(env)
    if policy == "pipeline_planner_comm_followers":
        return pipeline_planner_comm_followers(env)
    if policy == "pipeline_planner_comm_followers_regions":
        return pipeline_planner_comm_followers_regions(env)
    if policy == "pipeline_planner_dispatcher":
        return pipeline_planner_dispatcher(env)
    if policy == "pipeline_planner_semidec":
        return pipeline_planner_semidec(env)
    if policy == "energy_planner_comm":
        return energy_planner_comm(env)
    if policy == "signal_hunt_planner_comm":
        return signal_hunt_planner_comm(env)
    if policy == "comm_mat":
        checkpoint = spec.get("policy_checkpoint")
        if checkpoint and not os.path.isabs(checkpoint):
            checkpoint = os.path.join(ROOT, checkpoint)
        return CommMATPolicy(
            config=CommMATPolicyConfig(
                deterministic=bool(spec.get("comm_mat_deterministic", True)),
                send_threshold=float(spec.get("comm_mat_send_threshold", 0.5)),
            ),
            checkpoint=checkpoint,
        )
    raise ValueError(f"Unsupported benchmark policy: {policy}")


def _effective_result_spec(spec: dict, args, external_kwargs: dict) -> dict:
    result_spec = dict(spec)
    if args.policy_override:
        result_spec["policy"] = args.policy_override
    if args.policy_entrypoint:
        result_spec["policy"] = "external"
        result_spec["policy_entrypoint"] = args.policy_entrypoint
    if args.policy_checkpoint:
        result_spec["policy_checkpoint"] = args.policy_checkpoint
    if external_kwargs:
        result_spec["policy_kwargs"] = dict(external_kwargs)
    return result_spec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="syncorsink")
    parser.add_argument("--wandb-run", default=None)
    parser.add_argument("--results-json", default=None, help="Write a leaderboard result artifact JSON file")
    parser.add_argument("--track", default="symbolic_dtde", help="Leaderboard track for --results-json")
    parser.add_argument("--submission-name", default="local_run", help="Submission display name")
    parser.add_argument("--method-name", default="benchmark_run", help="Method/model name")
    parser.add_argument("--method-type", default="baseline", help="Method category, e.g. MAPPO, LLM, VLM")
    parser.add_argument("--authors", default="SyncOrSink Contributors", help="Comma-separated author names")
    parser.add_argument("--repository", default=None, help="Optional code repository URI")
    parser.add_argument("--checkpoint-uri", default=None, help="Optional checkpoint or artifact URI")
    parser.add_argument("--paper-uri", default=None, help="Optional paper/preprint URI")
    parser.add_argument("--notes", default=None, help="Optional submission notes")
    parser.add_argument(
        "--policy-entrypoint",
        default=None,
        help="External policy factory as 'module.submodule:object'; overrides built-in policy dispatch",
    )
    parser.add_argument(
        "--policy-override",
        default=None,
        help="Built-in policy name to use for every case, e.g. random, scripted, oracle_strong",
    )
    parser.add_argument(
        "--policy-checkpoint",
        default=None,
        help="Optional checkpoint path/URI passed to external policy loading",
    )
    parser.add_argument(
        "--policy-kwargs",
        default=None,
        help="Optional JSON object passed as extra keyword args to the external policy factory",
    )
    args = parser.parse_args()

    bench = load_benchmark(args.spec)
    case_results = []
    external_kwargs = _parse_policy_kwargs(args.policy_kwargs)

    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_run, config={
            "benchmark": bench.name,
        })

    for case in bench.cases:
        spec = dict(case.spec)
        if args.policy_override:
            spec["policy"] = args.policy_override
        config = SyncOrSinkConfig(
            scenario=spec["scenario"],
            split=spec.get("split"),
            map_variant=int(spec.get("map_variant", 0)),
            fov_preset=spec.get("fov_preset", "medium"),
            map_size=int(spec.get("map_size", 16)),
            num_agents=int(spec.get("agents", spec.get("num_agents", 3))),
            max_steps=int(spec.get("max_steps", 300)),
            comm_mode=spec.get("comm_mode", "tokens"),
            track=spec.get("track", "dtde"),
            energy_preset=spec.get("energy_preset", "hard"),
            energy_private_monitor=bool(spec.get("energy_private_monitor", True)),
        )
        env = SyncOrSinkEnv(config)
        policy = build_policy(
            spec,
            env,
            external_entrypoint=args.policy_entrypoint,
            external_checkpoint=args.policy_checkpoint,
            external_kwargs=external_kwargs,
        )
        episodes = int(spec.get("episodes", 1))

        if spec.get("mode", "marl") == "llm":
            ep_stats = run_llm_episodes(env, policy, episodes=episodes, seed=0)
            summary = summarize(ep_stats)
        else:
            summary, _ = run_episodes(env, policy, episodes=episodes, seed=0)

        print("case", case.name, "success", summary.success_rate, "return", summary.avg_return)
        case_results.append(
            summary_to_case_result(
                case.name,
                summary,
                spec=_effective_result_spec(spec, args, external_kwargs),
                weight=case.weight,
                tags=case.tags,
            )
        )

        if wandb_run is not None:
            wandb_run.log({
                f"{case.name}/success_rate": summary.success_rate,
                f"{case.name}/avg_return": summary.avg_return,
                f"{case.name}/avg_steps": summary.avg_steps,
                f"{case.name}/avg_comm_tokens": summary.avg_comm_tokens,
                f"{case.name}/track": spec.get("track", "dtde"),
            })

    if wandb_run is not None:
        wandb_run.finish()

    if args.results_json:
        artifact = make_result_artifact(
            benchmark_name=bench.name,
            benchmark_version=bench.version,
            track=args.track,
            submission=SubmissionInfo(
                name=args.submission_name,
                method_name=args.method_name,
                method_type=args.method_type,
                authors=[name.strip() for name in args.authors.split(",") if name.strip()],
                repository=args.repository,
                checkpoint_uri=args.checkpoint_uri,
                paper_uri=args.paper_uri,
                notes=args.notes,
            ),
            cases=case_results,
        )
        artifact["score"] = score_result_artifact(artifact)
        save_result_artifact(artifact, args.results_json)
        print("wrote", args.results_json)
        print("official_score", artifact["score"]["official_score"])


if __name__ == "__main__":
    main()
