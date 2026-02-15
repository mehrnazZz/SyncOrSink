# SyncOrSink

SyncOrSink is a communication‑focused, cooperative multi‑agent POMDP benchmark with long‑horizon coordination tasks. It provides multiple scenario families that stress different coordination problems while keeping a unified interface for MARL and LLM‑based agents.

Key goals:
- Partial observability with adjustable FOV presets (`hard`, `medium`, `easy`)
- Explicit communication with token budgets and efficiency penalties
- Long‑horizon coordination with semantic task structure
- Multiple map sizes (`8x8`, `16x16`, `32x32`)
- Multiple scenario families (task planning, resource sharing, cooperative search)

For detailed design, see:
- `docs/design.md`
- `docs/scenarios.md`

## Install

```bash
pip install -e .
```

Optional PettingZoo compatibility:

```bash
pip install -e ".[pettingzoo]"
```

Optional pygame renderer:

```bash
pip install -e ".[render]"
```

## Quick start

```bash
python examples/run_pipeline.py
```

Pygame visualizer:

```bash
python examples/run_pygame.py
```

Human playable demo:

```bash
python examples/run_human.py
```

God view / split view:
- `render_god_view=True` shows full map without fog
- `render_split_view=True` shows agent view + god view side-by-side
- `render_style="arcade_flat"` uses flat arcade visuals

RGB array rendering (for vision agents):

```python
env = SyncOrSinkEnv(SyncOrSinkConfig(), render_mode="rgb_array")
frame = env.render()  # HxWx3 uint8
```

Sprite-based rendering is enabled by default for Pygame. To disable, set `use_sprites=False` when creating the renderer (via env render mode initialization).

Scenario demos:

```bash
python examples/run_pipeline.py
python examples/run_energy.py
python examples/run_signal_hunt.py
```

Model library + MAPPO training loop:

```bash
pip install -e ".[train]"
python examples/mappo_train.py
```

Evaluation harness:

```bash
python examples/eval_run.py --scenario signal_hunt --episodes 10 --policy heuristic
```

Oracle baselines (full‑state planners):

```bash
python examples/eval_run.py --scenario pipeline_assembly --episodes 5 --policy oracle
python examples/eval_run.py --scenario energy_grid --episodes 5 --policy oracle
python examples/eval_run.py --scenario signal_hunt --episodes 5 --policy oracle
```

Stronger oracles (assignment + sync coordination):

```bash
python examples/eval_run.py --scenario pipeline_assembly --episodes 5 --policy oracle_strong
python examples/eval_run.py --scenario energy_grid --episodes 5 --policy oracle_strong
python examples/eval_run.py --scenario signal_hunt --episodes 5 --policy oracle_strong
```

W&B logging:

```bash
python examples/eval_run.py --scenario signal_hunt --episodes 10 --policy scripted --wandb --wandb-project syncorsink
```

Split evaluation (mean/std across seeds):

```bash
python examples/eval_split.py --scenario signal_hunt --split test --episodes-per-seed 3 --wandb --wandb-project syncorsink
```

LLM tool-calling stub:

```bash
python examples/llm_tool_stub.py
```

LLM evaluation:

```bash
python examples/eval_llm.py --scenario signal_hunt --episodes 5 --wandb --wandb-project syncorsink
```

LLM prompt cache:

```bash
python examples/eval_llm.py --scenario signal_hunt --episodes 5 --cache /tmp/syncorsink_cache.json
```

Unified eval spec:

```bash
python examples/eval_from_spec.py --spec /path/to/spec.json
```

Benchmark suite runner:

```bash
python examples/benchmark_run.py --spec examples/benchmark_spec.json --wandb --wandb-project syncorsink
```

Benchmark presets:

- `benchmarks/pipeline_presets.json`
  - `pipeline_easy_expert_comm`: centralized planner with comm broadcast (sanity/IL expert)
  - `pipeline_hard_coord`: region-only comm baseline (intentionally hard)
  - `energy_easy_expert_comm`: centralized energy planner with comm (easy preset)
  - `signal_hunt_expert_comm`: centralized signal hunt planner with comm

Expert planners:

- `docs/experts.md`

Solvability check (oracle feasibility):

```bash
python examples/solvability_check.py --scenario signal_hunt --split test --max 20
```
DTDE vs CTDE tracks:
- `dtde` = decentralized training & execution (no privileged state)
- `ctde` = centralized training (adds `info["central_obs"]`) with decentralized execution

Strict spec validation (optional):

```bash
pip install -e ".[eval]"
```

OpenAI tool schema (Chat Completions style):

```python
from syncorsink.llm.tools import openai_tools_schema

tools = openai_tools_schema()
# pass `tools` to your OpenAI client call
```

## Environments

Scenario families (all cooperative):
- **Pipeline Assembly** (task planning): assemble a multi-stage pipeline with dependencies using partial blueprints.
- **Energy Grid** (resource sharing): stabilize a shared grid with typed resources and synchronized low-energy recharges.
- **Signal Hunt** (cooperative search): collect distributed clues and jointly verify a hidden target within a scan window. Uses rooms, doors, occlusion, landmarks, and decoy targets.

Scenario success conditions and rewards are documented in `docs/scenarios.md`.

Design details (world mechanics, observations, communication, rewards) are documented in `docs/design.md`.
Configuration reference is documented in `docs/config.md`.

Each scenario supports:
- map sizes: `8`, `16`, `32`
- variable number of agents
- communication modes: `tokens` or `text`
- FOV presets: `hard`, `medium`, `easy`
- configurable map options: rooms, doors, fog-of-war (all scenarios), decoy targets (Signal Hunt)

## PettingZoo wrapper

```python
from syncorsink.envs import SyncOrSinkConfig
from syncorsink.envs.pz_wrapper import SyncOrSinkParallel

env = SyncOrSinkParallel(SyncOrSinkConfig(), render_mode="ansi")
obs, info = env.reset(seed=0)
```

## Vectorized env

```python
from syncorsink.envs import SyncOrSinkVector, SyncOrSinkConfig

venv = SyncOrSinkVector(num_envs=4, config=SyncOrSinkConfig())
obs, infos = venv.reset(seed=0)
```

Note: this wrapper returns Python lists (not Gym vector tensors) because observations are dict-of-agents.

## API sketch

```python
from syncorsink.envs import SyncOrSinkEnv, SyncOrSinkConfig

config = SyncOrSinkConfig(
    scenario="signal_hunt",
    map_size=16,
    num_agents=4,
    fov_preset="hard",
    comm_mode="text",
)
env = SyncOrSinkEnv(config)
obs, info = env.reset(seed=0)

actions = {
    0: {"action": 0, "message_text": "I saw clue at north room"},
    1: {"action": 4},
}
obs, rewards, done, truncated, info = env.step(actions)
```

Observations are dicts per agent with:
- `local_grid`: local tile view
- `inventory`: resource id held
- `messages_tokens`: tokenized message inbox (padded)
- `message_from`: sender ids
- `goal_hint`: integer hint tokens (scenario-dependent)

For free-text communication, the text payload is passed through `action["message_text"]` and echoed in `info["messages_text"]`.

## Action Space

The default discrete action set is:
- `0`: up
- `1`: down
- `2`: left
- `3`: right
- `4`: stay
- `5`: interact
- `6`: pickup
- `7`: drop

Actions are passed as a dict per agent, e.g.:

```python
actions = {0: {"action": 5, "message_tokens": []}, 1: {"action": 4}}
```

## Observation Space (Summary)

Each agent observation includes:
- `local_grid`: `(2*radius+1, 2*radius+1)` grid of tile ids
- `inventory`: `(1,)` integer item id
- `messages_tokens`: `(max_messages, comm_token_limit)` padded tokens
- `message_from`: `(max_messages,)` sender ids
- `goal_hint`: `(16,)` integer hint tokens

See `docs/design.md` for more detail.

## Observation Schema Example

For `fov_preset="medium"` (radius=3), `comm_token_limit=24`, `max_messages=8`:
- `local_grid`: `(7, 7)` because `2*3+1 = 7`
- `messages_tokens`: `(8, 24)` most recent 8 messages, each up to 24 tokens
- `message_from`: `(8,)` sender id per message

`messages_tokens` stores **received** messages; `message_from` aligns with it to identify the sender.
If `obs_onehot=True`, `local_grid` becomes one‑hot channels `(C,H,W)` instead of integer ids.

## Metrics

- Task score (success, time-to-success)
- Communication efficiency (success per token or reward minus comm cost)
- Generalization across unseen maps of the same scenario

## Training Policies

### MAPPO (RL)

Minimal MAPPO training loop:

```bash
pip install -e ".[train]"
python examples/mappo_train.py --scenario signal_hunt --updates 20 --rollout-steps 256 --epochs 4
```

Shared vs per‑agent actors:
- Add `--shared-actor` for a shared policy.
- Omit it for per‑agent policies.

Transformer backbone:
- Use `--backbone transformer`.

Checkpointing:
- `--save /path/to/ckpt.pt`
- `--load /path/to/ckpt.pt`

### LLM / Tool‑Calling Policies

LLM evaluation runner (stub):

```bash
python examples/eval_llm.py --scenario signal_hunt --episodes 5 --wandb --wandb-project syncorsink
```

Tool‑calling stub:

```bash
python examples/llm_tool_stub.py
```

For OpenAI tool calling schemas, see `syncorsink/llm/tools.py`.

### Policy Architecture Selection

- **MLP**: fast baseline for structured obs (grid + inventory).\n
- **Transformer**: better at long‑range dependencies (e.g., larger FOV, messaging).
- **VLM**: use `render_mode=\"rgb_array\"` and `VLMPolicy` adapter.

Model library lives in `syncorsink/models/` and provides encoders and heads.

## Tests

Run tests (determinism + reward sanity):

```bash
pytest /Users/mehrnaz/Documents/New\\ project/tests
```

## Deterministic map splits

You can pin map generation via `map_seed`, `map_variant`, or by using named splits:

```python
from syncorsink.envs import SyncOrSinkConfig

config = SyncOrSinkConfig(
    scenario="signal_hunt",
    split="test",
    map_variant=3,
)
```

Split seeds are defined in `syncorsink/eval/splits.py`.

## License

MIT
