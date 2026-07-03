import argparse
import os
import sys
import json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from syncorsink.envs import SyncOrSinkEnv, SyncOrSinkConfig
from syncorsink.eval.runner import run_episodes
from syncorsink.policies.random_policy import random_policy
from syncorsink.policies.heuristic import heuristic_policy
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
from syncorsink.policies.local_oracle import (
    local_oracle,
    local_oracle_comm,
    local_oracle_plus,
    local_oracle_plus_comm,
    local_oracle_team_comm,
    local_pipeline_policy,
    local_energy_policy,
    local_signal_policy,
)
from syncorsink.policies.planner import (
    pipeline_central_planner,
    energy_central_planner,
    signal_hunt_central_planner,
)
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


def _to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="signal_hunt")
    parser.add_argument("--map-size", type=int, default=None)
    parser.add_argument("--agents", type=int, default=None)
    parser.add_argument("--fov-preset", default=None, choices=["easy", "medium", "hard"])
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--split", default=None)
    parser.add_argument("--variant", type=int, default=0)
    parser.add_argument(
        "--policy",
        default="random",
        choices=[
            "random",
            "heuristic",
            "scripted",
            "oracle",
            "oracle_strong",
            "oracle_planner",
            "oracle_comm",
            "local_oracle",
            "local_oracle_comm",
            "local_oracle_plus",
            "local_oracle_plus_comm",
            "local_oracle_team_comm",
            "local_pipeline",
            "local_energy",
            "local_signal",
            "pipeline_planner",
            "pipeline_planner_comm",
            "pipeline_planner_follower",
            "pipeline_planner_comm_followers",
            "pipeline_planner_comm_followers_regions",
            "pipeline_planner_dispatcher",
            "pipeline_planner_semidec",
            "energy_planner",
            "signal_hunt_planner",
            "energy_planner_comm",
            "signal_hunt_planner_comm",
            "comm_mat",
            "bc",
        ],
    )
    parser.add_argument("--bc-ckpt", default=None, help="Path to BC model checkpoint")
    parser.add_argument("--energy-preset", default="hard", choices=["easy", "hard"])
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="syncorsink")
    parser.add_argument("--wandb-run", default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render-fps", type=float, default=10.0)
    parser.add_argument("--trace-jsonl", default=None, help="Write per-step trace to JSONL")
    parser.add_argument("--trace-local-obs", action="store_true", help="Include per-agent local observations in trace")
    parser.add_argument("--trace-render-ansi", action="store_true", help="Include env.render() ANSI map per step")
    parser.add_argument("--render-split-view", action="store_true", help="Render agent+god split view (rgb/video)")
    parser.add_argument("--render-god-view", action="store_true", help="Render god view (rgb/video)")
    parser.add_argument("--render-style", choices=["arcade_flat", "sprite"], default="arcade_flat")
    parser.add_argument("--record-video", action="store_true", help="Capture rgb frames and optionally log to W&B")
    parser.add_argument("--video-episodes", type=int, default=1, help="How many episodes to record")
    parser.add_argument("--video-fps", type=int, default=8, help="FPS metadata for W&B video")
    parser.add_argument("--wandb-log-trace-table", action="store_true", help="Log sampled step traces as W&B table")
    parser.add_argument("--wandb-trace-max-rows", type=int, default=2000, help="Max rows in W&B trace table")
    parser.add_argument("--wandb-log-trace-artifact", action="store_true", help="Upload trace JSONL as W&B artifact")
    parser.add_argument("--wandb-log-video", action="store_true", help="Log recorded videos to W&B")
    parser.add_argument("--comm-mat-ckpt", default=None, help="Optional Comm-MAT checkpoint path")
    parser.add_argument("--comm-mat-stochastic", action="store_true")
    parser.add_argument("--comm-mat-send-threshold", type=float, default=0.5)
    args = parser.parse_args()

    config_kwargs = dict(
        scenario=args.scenario,
        split=args.split,
        map_variant=args.variant,
        track="ctde" if "oracle" in args.policy else "dtde",
        energy_preset=args.energy_preset,
        render_split_view=args.render_split_view,
        render_god_view=args.render_god_view,
        render_style=args.render_style,
    )
    if args.map_size is not None:
        config_kwargs["map_size"] = args.map_size
    if args.agents is not None:
        config_kwargs["num_agents"] = args.agents
    if args.fov_preset is not None:
        config_kwargs["fov_preset"] = args.fov_preset
    if args.max_steps is not None:
        config_kwargs["max_steps"] = args.max_steps
    config = SyncOrSinkConfig(**config_kwargs)
    env = SyncOrSinkEnv(config, render_mode="rgb_array" if args.record_video else ("human" if args.render else None))
    trace_fh = open(args.trace_jsonl, "w", encoding="utf-8") if args.trace_jsonl else None
    trace_rows = []
    video_frames = {}
    render_failed = {"value": False}

    if args.policy == "random":
        policy = random_policy(env.action_space, env.num_agents)
    elif args.policy == "heuristic":
        policy = heuristic_policy(env)
    elif args.policy == "scripted":
        if args.scenario == "pipeline_assembly":
            policy = pipeline_planner(env)
        elif args.scenario == "energy_grid":
            policy = energy_planner(env)
        else:
            policy = signal_hunt_planner(env)
    elif args.policy == "oracle_strong":
        if args.scenario == "pipeline_assembly":
            policy = pipeline_oracle_strong(env)
        elif args.scenario == "energy_grid":
            policy = energy_oracle_strong(env)
        else:
            policy = signal_hunt_oracle_strong(env)
    elif args.policy == "oracle_planner":
        if args.scenario == "energy_grid":
            policy = energy_oracle_planner(env)
        elif args.scenario == "pipeline_assembly":
            policy = pipeline_oracle_strong(env)
        else:
            policy = signal_hunt_oracle_strong(env)
    elif args.policy == "oracle_comm":
        if args.scenario == "pipeline_assembly":
            base = pipeline_oracle_strong(env)
        elif args.scenario == "energy_grid":
            base = energy_oracle_strong(env)
        else:
            base = signal_hunt_oracle_strong(env)
        policy = wrap_oracle_with_comm(base, env)
    elif args.policy == "local_oracle":
        policy = local_oracle(env)
    elif args.policy == "local_oracle_comm":
        policy = local_oracle_comm(env)
    elif args.policy == "local_oracle_plus":
        policy = local_oracle_plus(env)
    elif args.policy == "local_oracle_plus_comm":
        policy = local_oracle_plus_comm(env)
    elif args.policy == "local_oracle_team_comm":
        policy = local_oracle_team_comm(env)
    elif args.policy == "local_pipeline":
        policy = local_pipeline_policy(env)
    elif args.policy == "local_energy":
        policy = local_energy_policy(env)
    elif args.policy == "local_signal":
        policy = local_signal_policy(env)
    elif args.policy == "pipeline_planner":
        policy = pipeline_central_planner(env)
    elif args.policy == "pipeline_planner_comm":
        policy = pipeline_planner_comm(env)
    elif args.policy == "pipeline_planner_follower":
        policy = pipeline_planner_follower(env)
    elif args.policy == "pipeline_planner_comm_followers":
        policy = pipeline_planner_comm_followers(env)
    elif args.policy == "pipeline_planner_comm_followers_regions":
        policy = pipeline_planner_comm_followers_regions(env)
    elif args.policy == "pipeline_planner_dispatcher":
        policy = pipeline_planner_dispatcher(env)
    elif args.policy == "pipeline_planner_semidec":
        policy = pipeline_planner_semidec(env)
    elif args.policy == "energy_planner":
        policy = energy_central_planner(env)
    elif args.policy == "signal_hunt_planner":
        policy = signal_hunt_central_planner(env)
    elif args.policy == "energy_planner_comm":
        policy = energy_planner_comm(env)
    elif args.policy == "signal_hunt_planner_comm":
        policy = signal_hunt_planner_comm(env)
    elif args.policy == "comm_mat":
        checkpoint = args.comm_mat_ckpt
        if checkpoint and not os.path.isabs(checkpoint):
            checkpoint = os.path.join(ROOT, checkpoint)
        policy = CommMATPolicy(
            config=CommMATPolicyConfig(
                deterministic=not args.comm_mat_stochastic,
                send_threshold=args.comm_mat_send_threshold,
            ),
            checkpoint=checkpoint,
        )
    elif args.policy == "bc":
        import torch
        from syncorsink.policies.mappo_models import MAPPOActor
        from syncorsink.train.mappo import flatten_obs, mask_action_logits
        if not args.bc_ckpt:
            raise RuntimeError("--bc-ckpt required for bc policy")
        ckpt = torch.load(args.bc_ckpt, map_location="cpu")
        bc_model = MAPPOActor(
            obs_dim=ckpt["obs_dim"],
            action_dim=8,
            hidden_dim=ckpt["hidden_dim"],
            backbone="mlp",
            comm_enabled=ckpt.get("comm", False),
            comm_token_limit=ckpt.get("comm_token_limit", 0),
            comm_vocab_size=ckpt.get("comm_vocab_size", 0),
        )
        bc_model.load_state_dict(ckpt["model"])
        bc_model.eval()
        def _bc_policy(obs, info, state):
            import numpy as np
            actions = {}
            for aid in sorted(obs.keys()):
                flat_arr = flatten_obs(obs[aid])
                model_dim = int(ckpt["obs_dim"])
                if flat_arr.shape[0] > model_dim:
                    model_arr = flat_arr[:model_dim]
                elif flat_arr.shape[0] < model_dim:
                    model_arr = np.pad(flat_arr, (0, model_dim - flat_arr.shape[0]))
                else:
                    model_arr = flat_arr
                flat = torch.tensor(model_arr, dtype=torch.float32).unsqueeze(0)
                mask = torch.tensor(
                    np.asarray(obs[aid].get("action_mask", np.ones((8,), dtype=np.float32)), dtype=np.float32),
                    dtype=torch.float32,
                ).reshape(1, -1)
                with torch.no_grad():
                    out = bc_model(flat)
                if ckpt.get("comm", False):
                    logits, send_logits, token_logits, len_logits = out
                    logits = mask_action_logits(logits, mask)
                    act = int(torch.argmax(logits, dim=-1).item())
                    send = int(torch.sigmoid(send_logits.squeeze(-1)).item() > 0.5)
                    if send:
                        msg_len = int(torch.argmax(len_logits, dim=-1).item())
                        msg_tokens = torch.argmax(token_logits, dim=-1)[0, :msg_len].tolist()
                    else:
                        msg_tokens = []
                    actions[aid] = {"action": act, "message_tokens": msg_tokens}
                else:
                    logits = mask_action_logits(out, mask)
                    act = int(torch.argmax(logits, dim=-1).item())
                    actions[aid] = {"action": act, "message_tokens": []}
            return actions
        policy = _bc_policy
    else:
        if args.scenario == "pipeline_assembly":
            policy = pipeline_oracle(env)
        elif args.scenario == "energy_grid":
            policy = energy_oracle(env)
        else:
            policy = signal_hunt_oracle(env)

    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_run, config={
            "scenario": args.scenario,
            "episodes": args.episodes,
            "split": args.split,
            "variant": args.variant,
            "policy": args.policy,
            "record_video": args.record_video,
            "render_split_view": args.render_split_view,
            "render_god_view": args.render_god_view,
            "render_style": args.render_style,
        })

    def _log_episode(ep_idx, ep_stats):
        if wandb_run is None:
            return
        data = {
            "episode": ep_idx,
            "ep_return": ep_stats.total_reward,
            "ep_steps": ep_stats.steps,
            "ep_success": 1.0 if ep_stats.success else 0.0,
            "ep_comm_tokens": ep_stats.comm_tokens,
        }
        for aid, r in ep_stats.per_agent_reward.items():
            data[f"ep_agent_{aid}_return"] = r
        for aid, c in ep_stats.per_agent_comm.items():
            data[f"ep_agent_{aid}_comm"] = c
        wandb_run.log(data)
        if args.record_video and args.wandb_log_video and wandb_run is not None:
            frames = video_frames.get(ep_idx, [])
            if frames:
                video = np.stack(frames, axis=0).transpose(0, 3, 1, 2)
                import wandb
                wandb_run.log({f"video/episode_{ep_idx}": wandb.Video(video, fps=args.video_fps, format="mp4")})

    def _log_step(ep_idx, step_idx, obs, info_before, actions, rewards, done, truncated, info_after):
        row = {
            "episode": ep_idx,
            "step": step_idx,
            "actions": _to_jsonable(actions),
            "rewards": _to_jsonable(rewards),
            "done": bool(done),
            "truncated": bool(truncated),
            "comm_tokens": _to_jsonable(info_after.get("comm_tokens", {})),
            "messages_text": _to_jsonable(info_before.get("messages_text", {})),
            "messages_with_sender": _to_jsonable(info_before.get("messages_with_sender", {})),
            "goal_hint_texts": _to_jsonable(info_before.get("goal_hint_texts", {})),
        }
        if args.trace_local_obs:
            row["obs"] = _to_jsonable(obs)
        if args.trace_render_ansi:
            row["ansi_map"] = env.render()
        if trace_fh is not None:
            trace_fh.write(json.dumps(row, ensure_ascii=True) + "\n")
            trace_fh.flush()
        if args.wandb and args.wandb_log_trace_table and len(trace_rows) < args.wandb_trace_max_rows:
            trace_rows.append(
                {
                    "episode": row["episode"],
                    "step": row["step"],
                    "done": row["done"],
                    "truncated": row["truncated"],
                    "comm_tokens_total": int(sum((row.get("comm_tokens") or {}).values())),
                    "actions": json.dumps(row.get("actions", {}), ensure_ascii=True),
                    "rewards": json.dumps(row.get("rewards", {}), ensure_ascii=True),
                }
            )
        if args.record_video and ep_idx < args.video_episodes:
            if not render_failed["value"]:
                try:
                    frame = env.render()
                except Exception as exc:
                    render_failed["value"] = True
                    print(f"video capture disabled: {exc}")
                    frame = None
                if frame is not None:
                    video_frames.setdefault(ep_idx, []).append(np.array(frame, copy=True))

    summary, episodes = run_episodes(
        env,
        policy,
        episodes=args.episodes,
        seed=0,
        per_episode_cb=_log_episode,
        per_step_cb=_log_step,
        render=args.render,
        render_fps=args.render_fps,
    )
    print("episodes", summary.episodes)
    print("success_rate", summary.success_rate)
    print("avg_return", summary.avg_return)
    print("avg_steps", summary.avg_steps)
    print("avg_comm_tokens", summary.avg_comm_tokens)
    print("avg_agent_return", summary.avg_agent_reward)
    print("avg_agent_comm", summary.avg_agent_comm)

    if wandb_run is not None:
        data = {
            "success_rate": summary.success_rate,
            "avg_return": summary.avg_return,
            "avg_steps": summary.avg_steps,
            "avg_comm_tokens": summary.avg_comm_tokens,
        }
        for aid, r in summary.avg_agent_reward.items():
            data[f"avg_agent_{aid}_return"] = r
        for aid, c in summary.avg_agent_comm.items():
            data[f"avg_agent_{aid}_comm"] = c
        wandb_run.log(data)
        if args.wandb_log_trace_table and trace_rows:
            import wandb
            cols = [
                "episode",
                "step",
                "done",
                "truncated",
                "comm_tokens_total",
                "actions",
                "rewards",
            ]
            table = wandb.Table(columns=cols)
            for r in trace_rows:
                table.add_data(*[r.get(c) for c in cols])
            wandb_run.log({"trace/steps_table": table})
        if args.wandb_log_trace_artifact and args.trace_jsonl and os.path.exists(args.trace_jsonl):
            import wandb
            art = wandb.Artifact("eval_trace", type="trace")
            art.add_file(args.trace_jsonl)
            wandb_run.log_artifact(art)
        wandb_run.finish()
    if trace_fh is not None:
        trace_fh.close()


if __name__ == "__main__":
    main()
