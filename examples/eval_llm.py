import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from syncorsink.envs import SyncOrSinkEnv, SyncOrSinkConfig
from syncorsink.eval.metrics import summarize
from syncorsink.eval.llm_runner import run_llm_episodes
from syncorsink.llm.policy import LLMPolicy, LLMExecutorPolicy
from syncorsink.llm.tool_policy import ToolCallingPolicy
from syncorsink.llm.cache import PromptCache
from syncorsink.llm.openai_client import OpenAIToolCaller
from syncorsink.llm.responses_client import OpenAIResponsesCaller


def dummy_llm(prompt: str):
    return '{"action": 4, "message_text": ""}'


def dummy_tool_llm(prompt: str):
    return [
        {"name": "move", "arguments": {"direction": "stay"}},
        {"name": "message", "arguments": {"text": "holding"}},
    ]


_MAX_PROMPT_CHARS = 100_000


def _sanitize_prompt(prompt: str) -> str:
    """Remove control characters that break JSON serialization and cap length."""
    # Strip characters outside printable ASCII + common whitespace that can
    # cause ``json.dumps`` (and therefore the OpenAI HTTP body) to produce
    # invalid JSON.  Keep newlines and tabs but remove NUL, BEL, etc.
    cleaned = "".join(
        ch for ch in prompt
        if ch in ("\n", "\r", "\t") or (ord(ch) >= 0x20 and ord(ch) != 0x7F)
    )
    if len(cleaned) > _MAX_PROMPT_CHARS:
        cleaned = cleaned[:_MAX_PROMPT_CHARS] + "\n[prompt truncated]"
    return cleaned


def build_text_llm_call(provider: str, client, model: str, planner_style: str = "action"):
    if provider == "openai-chat":
        system_text = (
            "Return only compact JSON with keys: action and message_text. "
            "action can be int or one of up/down/left/right/stay/interact/pickup/drop."
        )
        if planner_style == "executor":
            system_text = (
                "Return only compact JSON with keys: task, task_plan, message_text. "
                "task should be one of pickup_visible_resource/deliver_to_matching_node/sync_interact/explore_sector/hold_position/respond_to_teammate."
            )
        def _call(prompt: str):
            prompt = _sanitize_prompt(prompt)
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": system_text,
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                )
            except Exception as exc:
                print(f"[eval_llm] OpenAI API error (len={len(prompt)}): {exc}")
                return '{"action": "stay", "message_text": ""}'
            content = resp.choices[0].message.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                text_chunks = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_chunks.append(part.get("text", ""))
                return "\n".join(text_chunks)
            return ""
        return _call

    def _call(prompt: str):
        prompt = _sanitize_prompt(prompt)
        try:
            resp = client.responses.create(model=model, input=prompt)
        except Exception as exc:
            print(f"[eval_llm] OpenAI API error (len={len(prompt)}): {exc}")
            return '{"action": "stay", "message_text": ""}'
        if hasattr(resp, "output_text"):
            return str(resp.output_text)
        if hasattr(resp, "model_dump"):
            data = resp.model_dump()
        elif isinstance(resp, dict):
            data = resp
        else:
            return ""
        if "output_text" in data and data["output_text"] is not None:
            return str(data["output_text"])
        pieces = []
        for item in data.get("output", []):
            for content in item.get("content", []):
                if content.get("type") in {"output_text", "text"} and content.get("text"):
                    pieces.append(content["text"])
        return "\n".join(pieces)
    return _call


def build_litellm_call(model: str, planner_style: str = "action", num_retries: int = 2):
    """Build an LLM call function using litellm.

    litellm supports many backends via model prefixes:
      - "gpt-4o"               → OpenAI (needs OPENAI_API_KEY)
      - "anthropic/<model-name>"    → Anthropic (needs ANTHROPIC_API_KEY)
      - "ollama/llama3"        → local ollama
      - "ollama_chat/llama3"   → local ollama (chat format)
      - "together_ai/..."      → Together AI
      - etc.  See: https://docs.litellm.ai/docs/providers
    """
    try:
        import litellm
        litellm.num_retries = num_retries
    except ImportError as exc:
        raise RuntimeError("litellm package required. Install with: pip install litellm") from exc

    system_text = (
        "Return only compact JSON with keys: action and message_text. "
        "action can be int or one of up/down/left/right/stay/interact/pickup/drop."
    )
    if planner_style == "executor":
        system_text = (
            "Return only compact JSON with keys: task, task_plan, message_text. "
            "task should be one of pickup_visible_resource/deliver_to_matching_node/sync_interact/explore_sector/hold_position/respond_to_teammate."
        )

    def _call(prompt: str):
        prompt = _sanitize_prompt(prompt)
        try:
            resp = litellm.completion(
                model=model,
                messages=[
                    {"role": "system", "content": system_text},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:
            print(f"[eval_llm] litellm error ({model}, len={len(prompt)}): {exc}")
            return '{"action": "stay", "message_text": ""}'
    return _call


def parallel_batch(prompts, llm_call, max_workers: int = 8):
    results = [None for _ in prompts]
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(llm_call, p): i for i, p in enumerate(prompts)}
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except Exception as exc:
                print(f"[eval_llm] batch call {idx} failed: {exc}")
                results[idx] = '{"action": "stay", "message_text": ""}'
    return results


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
    parser.add_argument("--map-size", type=int, default=8)
    parser.add_argument("--agents", type=int, default=3)
    parser.add_argument("--fov-preset", choices=["easy", "medium", "hard"], default="easy")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--comm-cost", type=float, default=None)
    parser.add_argument("--comm-len-cost", type=float, default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--variant", type=int, default=0)
    parser.add_argument("--energy-preset", choices=["easy", "hard"], default="hard")
    parser.add_argument("--energy-private-monitor", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--provider", choices=["dummy", "openai-chat", "openai-responses", "litellm"], default="dummy")
    parser.add_argument("--mode", choices=["text", "tools"], default="tools")
    parser.add_argument("--planner", choices=["action", "executor"], default="action")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--trace-jsonl", default=None, help="Write per-step trace to JSONL")
    parser.add_argument("--trace-local-obs", action="store_true", help="Include per-agent local observations in trace")
    parser.add_argument("--trace-render-ansi", action="store_true", help="Include env.render() ANSI map per step")
    parser.add_argument("--render-split-view", action="store_true", help="Render agent+god split view (rgb/video)")
    parser.add_argument("--render-god-view", action="store_true", help="Render god view (rgb/video)")
    parser.add_argument("--render-style", choices=["arcade_flat", "sprite"], default="arcade_flat")
    parser.add_argument("--record-video", action="store_true", help="Capture rgb frames and optionally log to W&B")
    parser.add_argument("--video-episodes", type=int, default=1, help="How many episodes to record")
    parser.add_argument("--video-fps", type=int, default=8, help="FPS metadata for W&B video")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="syncorsink")
    parser.add_argument("--wandb-run", default=None)
    parser.add_argument("--wandb-log-trace-table", action="store_true", help="Log sampled step traces as W&B table")
    parser.add_argument("--wandb-trace-max-rows", type=int, default=2000, help="Max rows in W&B trace table")
    parser.add_argument("--wandb-log-trace-artifact", action="store_true", help="Upload trace JSONL as W&B artifact")
    parser.add_argument("--wandb-log-video", action="store_true", help="Log recorded videos to W&B")
    parser.add_argument("--cache", default=None, help="Path to prompt cache JSON")
    args = parser.parse_args()

    config = SyncOrSinkConfig(
        scenario=args.scenario,
        map_size=args.map_size,
        num_agents=args.agents,
        fov_preset=args.fov_preset,
        max_steps=args.max_steps,
        split=args.split,
        map_variant=args.variant,
        comm_mode="text",
        track="dtde",
        energy_preset=args.energy_preset,
        energy_private_monitor=args.energy_private_monitor,
        render_split_view=args.render_split_view,
        render_god_view=args.render_god_view,
        render_style=args.render_style,
    )
    if args.comm_cost is not None:
        config.comm_cost = float(args.comm_cost)
    if args.comm_len_cost is not None:
        config.comm_len_cost = float(args.comm_len_cost)
    env = SyncOrSinkEnv(config, render_mode="rgb_array" if args.record_video else None)
    trace_fh = open(args.trace_jsonl, "w", encoding="utf-8") if args.trace_jsonl else None
    trace_rows = []
    episode_task_rollup = {}
    video_frames = {}
    render_failed = {"value": False}

    cache = PromptCache(path=args.cache) if args.cache else None

    if args.provider == "dummy":
        if args.mode == "tools":
            policy = ToolCallingPolicy(dummy_tool_llm)
        else:
            if args.planner == "executor":
                policy = LLMExecutorPolicy(dummy_llm, cache=cache, batch_fn=parallel_batch)
            else:
                policy = LLMPolicy(dummy_llm, cache=cache, batch_fn=parallel_batch)
    elif args.provider == "litellm":
        llm_call = build_litellm_call(args.model, planner_style=args.planner)
        if args.planner == "executor":
            policy = LLMExecutorPolicy(llm_call, cache=cache, batch_fn=parallel_batch)
        else:
            policy = LLMPolicy(llm_call, cache=cache, batch_fn=parallel_batch)
    else:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai package is required for non-dummy provider. Install with: pip install openai") from exc
        api_key = os.getenv(args.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key env var: {args.api_key_env}")
        client = OpenAI(api_key=api_key)

        if args.mode == "tools":
            if args.provider == "openai-chat":
                caller = OpenAIToolCaller(client=client, model=args.model)
            else:
                caller = OpenAIResponsesCaller(client=client, model=args.model)
            policy = ToolCallingPolicy(caller.call)
        else:
            llm_call = build_text_llm_call(args.provider, client, args.model, planner_style=args.planner)
            if args.planner == "executor":
                policy = LLMExecutorPolicy(llm_call, cache=cache, batch_fn=parallel_batch)
            else:
                policy = LLMPolicy(llm_call, cache=cache, batch_fn=parallel_batch)

    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_run, config={
            "scenario": args.scenario,
            "map_size": args.map_size,
            "agents": args.agents,
            "fov_preset": args.fov_preset,
            "max_steps": args.max_steps,
            "episodes": args.episodes,
            "comm_cost": config.comm_cost,
            "comm_len_cost": config.comm_len_cost,
            "split": args.split,
            "variant": args.variant,
            "provider": args.provider,
            "mode": args.mode,
            "planner": args.planner,
            "model": args.model,
            "record_video": args.record_video,
            "render_split_view": args.render_split_view,
            "render_god_view": args.render_god_view,
            "render_style": args.render_style,
        })

    def _log_episode(ep_idx, ep_stats):
        print(f"episode {ep_idx + 1}/{args.episodes}: return={ep_stats.total_reward:.3f} steps={ep_stats.steps} success={int(ep_stats.success)} comm={ep_stats.comm_tokens}")
        task_rollup = episode_task_rollup.get(ep_idx, {})
        if task_rollup:
            print("task_rollup", task_rollup)
        if wandb_run is None:
            return
        data = {
            "episode": ep_idx,
            "ep_return": ep_stats.total_reward,
            "ep_steps": ep_stats.steps,
            "ep_success": 1.0 if ep_stats.success else 0.0,
            "ep_comm_tokens": ep_stats.comm_tokens,
        }
        for aid, counts in task_rollup.items():
            for name, v in counts.items():
                data[f"ep_agent_{aid}_task_{name}"] = int(v)
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

    def _log_step(ep_idx, step_idx, obs, info_before, actions, rewards, done, truncated, info_after, policy_state):
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
            "llm_calls": _to_jsonable(policy_state.get("llm_calls", [])),
            "task_metrics": _to_jsonable(policy_state.get("task_metrics", {})),
            "task_events": _to_jsonable(policy_state.get("task_events", [])),
        }
        tm = policy_state.get("task_metrics", {})
        if tm:
            ep_roll = episode_task_rollup.setdefault(ep_idx, {})
            for aid, m in tm.items():
                counts = (m or {}).get("completed_counts", {})
                dst = ep_roll.setdefault(str(aid), {"success": 0, "timeout": 0, "stuck": 0, "interrupted": 0})
                for key in ["success", "timeout", "stuck", "interrupted"]:
                    dst[key] = max(int(dst.get(key, 0)), int(counts.get(key, 0)))
        if args.trace_local_obs:
            row["obs"] = _to_jsonable(obs)
        if args.trace_render_ansi:
            row["ansi_map"] = env.render()
        if trace_fh is not None:
            trace_fh.write(json.dumps(row, ensure_ascii=True) + "\n")
            trace_fh.flush()
        if args.wandb and args.wandb_log_trace_table and len(trace_rows) < args.wandb_trace_max_rows:
            llm_calls = row.get("llm_calls", [])
            first_call = llm_calls[0] if llm_calls else {}
            task_events = row.get("task_events", [])
            first_task_event = task_events[0] if task_events else {}
            trace_rows.append(
                {
                    "episode": row["episode"],
                    "step": row["step"],
                    "done": row["done"],
                    "truncated": row["truncated"],
                    "comm_tokens_total": int(sum((row.get("comm_tokens") or {}).values())),
                    "actions": json.dumps(row.get("actions", {}), ensure_ascii=True),
                    "first_agent_id": first_call.get("agent_id"),
                    "first_mode": first_call.get("mode"),
                    "first_from_chunk": first_call.get("from_chunk"),
                    "first_model_action": json.dumps(first_call.get("model_action", {}), ensure_ascii=True),
                    "first_parsed_action": json.dumps(first_call.get("parsed_action", {}), ensure_ascii=True),
                    "first_prompt": str(first_call.get("prompt", ""))[:2000],
                    "first_raw_response": str(first_call.get("raw_response", ""))[:2000],
                    "first_task_event": json.dumps(first_task_event, ensure_ascii=True),
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
        policy_state["llm_calls"] = []

    print(
        f"running llm eval: provider={args.provider} mode={args.mode} model={args.model} "
        f"scenario={args.scenario} episodes={args.episodes} max_steps={args.max_steps} agents={args.agents}"
    )
    episodes = run_llm_episodes(
        env,
        policy,
        episodes=args.episodes,
        seed=0,
        per_episode_cb=_log_episode,
        per_step_cb=_log_step,
    )
    summary = summarize(episodes)
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
                "first_agent_id",
                "first_mode",
                "first_from_chunk",
                "first_model_action",
                "first_parsed_action",
                "first_prompt",
                "first_raw_response",
                "first_task_event",
            ]
            table = wandb.Table(columns=cols)
            for r in trace_rows:
                table.add_data(*[r.get(c) for c in cols])
            wandb_run.log({"trace/steps_table": table})
        if args.wandb_log_trace_artifact and args.trace_jsonl and os.path.exists(args.trace_jsonl):
            import wandb
            art = wandb.Artifact("llm_trace", type="trace")
            art.add_file(args.trace_jsonl)
            wandb_run.log_artifact(art)
        wandb_run.finish()
    if trace_fh is not None:
        trace_fh.close()


if __name__ == "__main__":
    main()
