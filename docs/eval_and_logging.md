# Evaluation And Logging Configuration

This document is the reference for evaluation/training CLI parameters and W&B logging features, including trace capture and environment video logging.

## Scripts covered

- Evaluation:
  - `examples/eval_run.py`
  - `examples/eval_llm.py`
  - `examples/benchmark_run.py`
  - `examples/eval_from_spec.py`
  - `examples/communication_ablation_sweep.py`
- Training:
  - `examples/mappo_train.py` (`syncorsink/train/mappo.py`)
  - `examples/comm_mat_train.py` (`syncorsink/train/comm_mat.py`)
  - `examples/tarmac_train.py` (`syncorsink/train/tarmac.py`)
  - `examples/core_training_sweep.py`

## `eval_run.py` parameters

| Flag | Type | Default | Description |
|---|---|---|---|
| `--scenario` | str | `signal_hunt` | Scenario name. |
| `--episodes` | int | `10` | Number of episodes. |
| `--split` | str? | `None` | Dataset split (`train/val/test`) if used. |
| `--variant` | int | `0` | Map variant index. |
| `--policy` | str | `random` | Policy selector (`random`, `heuristic`, scripted/oracle variants, `comm_mat`, etc.). |
| `--energy-preset` | str | `hard` | Energy Grid dynamics preset (`easy`, `hard`). |
| `--render` | bool | `False` | Enable live rendering. |
| `--render-fps` | float | `10.0` | Render speed. |
| `--trace-jsonl` | str? | `None` | Write per-step trace rows to JSONL. |
| `--trace-local-obs` | bool | `False` | Include local observations in trace rows. |
| `--trace-render-ansi` | bool | `False` | Include ANSI map snapshot in trace rows. |
| `--render-split-view` | bool | `False` | Split view render (agent+god). |
| `--render-god-view` | bool | `False` | God-view render mode. |
| `--render-style` | str | `arcade_flat` | Render style (`arcade_flat`, `sprite`). |
| `--record-video` | bool | `False` | Capture RGB frames for video logging. |
| `--video-episodes` | int | `1` | Number of episodes to record. |
| `--video-fps` | int | `8` | FPS metadata for video logs. |
| `--wandb` | bool | `False` | Enable W&B logging for episode + summary stats. |
| `--wandb-project` | str | `syncorsink` | W&B project name. |
| `--wandb-run` | str? | `None` | W&B run name. |
| `--wandb-log-trace-table` | bool | `False` | Log sampled per-step traces as W&B table. |
| `--wandb-trace-max-rows` | int | `2000` | Max rows in W&B trace table. |
| `--wandb-log-trace-artifact` | bool | `False` | Upload trace JSONL as W&B artifact. |
| `--wandb-log-video` | bool | `False` | Upload recorded videos to W&B. |
| `--comm-mat-ckpt` | str? | `None` | Optional Comm-MAT checkpoint for `--policy comm_mat`. |
| `--comm-mat-stochastic` | bool | `False` | Stochastic Comm-MAT decoding (default is deterministic). |
| `--comm-mat-send-threshold` | float | `0.5` | Comm-MAT send gate threshold. |

## `eval_llm.py` parameters

### Environment/eval parameters

| Flag | Type | Default | Description |
|---|---|---|---|
| `--scenario` | str | `signal_hunt` | Scenario name. |
| `--map-size` | int | `8` | Map size. |
| `--agents` | int | `3` | Number of agents. |
| `--fov-preset` | str | `easy` | FOV preset (`easy`, `medium`, `hard`). |
| `--max-steps` | int | `300` | Max steps per episode. |
| `--episodes` | int | `5` | Number of episodes. |
| `--split` | str? | `None` | Split name. |
| `--variant` | int | `0` | Map variant index. |
| `--comm-cost` | float? | `None` | Override env comm cost. |
| `--comm-len-cost` | float? | `None` | Override env comm length cost. |

### LLM/provider parameters

| Flag | Type | Default | Description |
|---|---|---|---|
| `--provider` | str | `dummy` | `dummy`, `openai-chat`, `openai-responses`. |
| `--mode` | str | `tools` | LLM interaction mode (`text`, `tools`). |
| `--planner` | str | `action` | Text planner style (`action`, `executor`). |
| `--model` | str | `gpt-4o-mini` | Provider model name. |
| `--api-key-env` | str | `OPENAI_API_KEY` | API key environment variable name. |
| `--cache` | str? | `None` | Prompt cache path. |

### Trace parameters

| Flag | Type | Default | Description |
|---|---|---|---|
| `--trace-jsonl` | str? | `None` | Write per-step trace rows to JSONL. |
| `--trace-local-obs` | bool | `False` | Include raw local observations in trace rows. |
| `--trace-render-ansi` | bool | `False` | Include ANSI map snapshot in trace rows. |

### Render/video parameters

| Flag | Type | Default | Description |
|---|---|---|---|
| `--render-split-view` | bool | `False` | Agent+god split view rendering mode. |
| `--render-god-view` | bool | `False` | God-view rendering mode. |
| `--render-style` | str | `arcade_flat` | Visual style (`arcade_flat`, `sprite`). |
| `--record-video` | bool | `False` | Capture RGB frames for episode video. |
| `--video-episodes` | int | `1` | Number of episodes to record. |
| `--video-fps` | int | `8` | FPS metadata for W&B video export. |

### W&B-specific logging parameters (`eval_llm.py`)

| Flag | Type | Default | Description |
|---|---|---|---|
| `--wandb` | bool | `False` | Enable W&B run. |
| `--wandb-project` | str | `syncorsink` | W&B project name. |
| `--wandb-run` | str? | `None` | W&B run name. |
| `--wandb-log-trace-table` | bool | `False` | Log sampled per-step traces as W&B Table. |
| `--wandb-trace-max-rows` | int | `2000` | Max trace rows in W&B table. |
| `--wandb-log-trace-artifact` | bool | `False` | Upload `--trace-jsonl` as W&B artifact. |
| `--wandb-log-video` | bool | `False` | Upload recorded videos to W&B. |

## Trace schema (LLM eval)

Each JSONL row from `--trace-jsonl` includes:

- `episode`, `step`
- `actions`, `rewards`, `done`, `truncated`
- `comm_tokens`, `messages_text`, `messages_with_sender`
- `llm_calls` (prompt/response and parsed actions)
- `task_metrics`, `task_events`
- optional: `obs` (if `--trace-local-obs`)
- optional: `ansi_map` (if `--trace-render-ansi`)

This supports prompt/response analysis, communication timeline inspection, and post-hoc debugging.
Private scenario hints are encoded in each agent's `goal_hint` observation, not
in shared `info`, so traces should use `--trace-local-obs` when inspecting them.

## W&B outputs by script

### `eval_run.py`

- Summary metrics:
  - `success_rate`, `avg_return`, `avg_steps`, `avg_comm_tokens`
  - per-agent averages
- Episode metrics:
  - `ep_return`, `ep_steps`, `ep_success`, `ep_comm_tokens`
  - per-agent return/comm
- Optional:
  - per-step trace table (`trace/steps_table`)
  - trace artifact (`eval_trace`)
  - per-episode MP4 videos (`video/episode_*`)

### `eval_llm.py`

- All eval summary + per-episode metrics above
- Optional:
  - per-step trace table (`trace/steps_table`)
  - trace artifact (`llm_trace`)
  - per-episode MP4 videos (`video/episode_*`)

### `communication_ablation_sweep.py`

- Per scenario/size/condition:
  - success rate, average return, average steps, average communication tokens
- Per scenario/size gap metrics:
  - communication expert success
  - local no-communication success
  - success gap
  - communication-token checks
- Optional:
  - JSON artifact via `--output-json`
  - CSV rows via `--output-csv`
  - W&B scalar logging via `--wandb`

### Training scripts (`mappo`, `comm_mat`, and `tarmac`)

All three support:

- `--wandb`, `--wandb-project`, `--wandb-run`
- periodic training logs (`loss`, `policy_loss`, `value_loss`, `entropy`, rollout stats)
- periodic eval logs (`eval/mean_return`, `eval/mean_steps`, `eval/success_rate`)

### Recurrent BC/DAgger/PPO trainer

`python -m syncorsink.train.recurrent_bc_rl` supports W&B logging for oracle
demos, recurrent BC, DAgger collection/eval, and PPO fine-tuning. Useful
scalars include:

- `bc/*` losses, accuracies, auxiliary Signal Hunt action heads, send rates,
  and learning rate
- `bc/signal_clue_interact_*` counts, loss, match rate, and mean action
  probability for the opt-in center-clue pickup auxiliary
- `bc/signal_evidence_sweep_*` counts, loss, match rate, and mean action
  probability for the opt-in large-map sector-search auxiliary
- `bc/signal_frontier_exploration_*` labels and action-match metrics for the
  opt-in large-map search fallback auxiliary, active only when no true-target
  candidate is decoded from the observation
- `dagger/collect_*` model-visited rollout stats, focus event counts, replay
  snippets, oracle-message roll-in, and per-map dataset diagnostics
- `dagger/eval/*`, `init/eval/*`, and `rl/eval/*` success/return/step metrics,
  including per-map metrics when `--eval-map-sizes` is set
- `rollout/*` PPO reward, communication, KL, action histogram, learning rate,
  and optional guided-decoding indicators

Pipeline Assembly diagnostics:

- Positive training events include `picked_resource`, `delivered`,
  `stage_completed`, `sync_complete`, and `pipeline_complete`.
- Pipeline focus events include `pipeline_wrong_delivery`,
  `pipeline_dependency_blocked`, `pipeline_sync_wait`,
  `pipeline_pickup_miss`, `pipeline_delivery_miss`,
  `pipeline_station_stall_miss`, and `pipeline_drop_miss`.
- Pipeline BC action-supervision knobs include
  `--bc-pipeline-pickup-action-loss-weight`,
  `--bc-pipeline-delivery-action-loss-weight`,
  `--bc-pipeline-delivery-progress-action-loss-weight`,
  `--bc-pipeline-navigation-action-loss-weight`,
  `--bc-pipeline-frontier-exploration-action-loss-weight`,
  `--bc-pipeline-sync-action-loss-weight`,
  `--bc-pipeline-station-guard-action-loss-weight`,
  `--bc-pipeline-pickup-gate-loss-weight`,
  `--bc-pipeline-plan-action-loss-weight`,
  `--bc-pipeline-plan-head-loss-weight`,
  `--bc-pipeline-option-loss-weight`,
  `--bc-pipeline-bad-pickup-action-loss-weight`,
  `--bc-pipeline-bad-drop-action-loss-weight`, and
  `--bc-pipeline-bad-interact-action-loss-weight`. Structured planner-message
  content supervision can be upweighted with `--bc-pipeline-message-loss-weight`.
  Use `--bc-pipeline-send-gate-loss-weight` with
  `--bc-pipeline-send-gate-pos-weight` and
  `--bc-pipeline-send-gate-neg-weight` to teach when those planner messages
  should be sent without turning every step into a broadcast. W&B logs matching
  `bc/pipeline_*`, `bc/pipeline_plan_*`, `bc/pipeline_option_*`,
  `bc/pipeline_message_*`,
  `bc/pipeline_send_gate_*`, and `dagger/collect_pipeline_*` metrics. The
  delivery-progress auxiliary labels agents already carrying a needed resource
  to move toward the matching active station, then interact there, which is
  useful when audits show pickup without completion or a collapse to no
  delivery. The navigation auxiliary distills the existing trusted-plan Pipeline
  assist into movement-only labels for walking toward needed visible resources
  or the active station without relying on eval-time navigation assist. The
  frontier-exploration auxiliary is opt-in and labels exploration-memory
  frontier moves only when the trusted active stage still needs a resource, the
  agent is empty-handed, and no required resource is locally visible. It requires
  `--obs-exploration-memory`; for recurrent multi-map/full-memory runs use
  `--obs-memory-mode egocentric` to keep the observation contract stable. The
  pickup-gate auxiliary labels visible pickup opportunities as
  positive only when the resource belongs to the trusted current active stage,
  which helps separate active-stage resources from blocked or future-stage
  resources in Pipeline audits. The station-guard auxiliary labels a concrete
  non-idle recovery or navigation action whenever station `INTERACT` would be
  useless or wrong, reducing the chance that the policy treats every station
  tile as an interaction target.
- `--dagger-pipeline-wrong-delivery-provenance-labels` is an experimental
  Pipeline DAgger option that traces an actual `pipeline_wrong_delivery` event
  back to the earlier model-only pickup of the same carried resource. When this
  fires it labels the source pickup as a `pipeline_bad_pickup` example and adds
  a `pipeline_wrong_delivery_root_pickup` root-cause replay trigger. Tune it
  with `--dagger-pipeline-wrong-delivery-provenance-weight`; negative values
  reuse `--dagger-focus-error-weight`. The root event can also be bounded with
  `--dagger-replay-event-weights pipeline_wrong_delivery_root_pickup:0.5` and
  `--dagger-replay-event-caps pipeline_wrong_delivery_root_pickup:1`. Keep it
  opt-in until the replay balance is tuned.
- The guarded Pipeline sweep profile prioritizes `pipeline_sync_wait` replay,
  so failed rendezvous states are replayed alongside delivery-ready,
  delivery-miss, station-stall, and wrong-delivery snippets.
- `--bc-pipeline-proactive-bad-action-labels` is an experimental opt-in that
  adds trusted-plan negative labels for picking up unneeded resources,
  wrong-station interacts, and dropping a still-needed resource, plus
  wrong-item station recovery labels. Keep runs with this flag separate until
  tuned.
- `--bc-pipeline-interact-gate-*` trains a binary station-interaction gate from
  action logits: positive labels are station states where `INTERACT` would
  deliver, sync, or complete a stage; negative labels are station no-ops or
  wrong-resource interactions. The recurrent actor also trains a learned
  hidden-state gate from the same labels. W&B logs `bc/pipeline_interact_gate_*`
  and `bc/pipeline_interact_head_*` metrics, including positive/negative counts,
  mean probabilities, and predicted interact rates.
- `--bc-pipeline-plan-head-loss-weight` trains a separate recurrent
  `pipeline_plan_policy` head from the trusted plan-action labels. This is an
  experimental distillation path for testing whether a model can learn the
  local pickup/navigation/delivery action implied by the decoded Pipeline plan
  without directly enabling the rule navigation assist.
- `--bc-pipeline-option-loss-weight` trains a separate recurrent
  `pipeline_option_policy` head over high-level Pipeline options:
  `none`, `pickup`, `deliver`, `sync`, `drop`, `nav_resource`, `nav_station`,
  and `wait`. The labels come from the trusted Pipeline plan and current active
  stage, so this is the first hierarchical distillation path above primitive
  actions.
- `--obs-pipeline-feedback` appends recurrent-only Pipeline progress feedback:
  self-local event bits plus compact metadata for the last Pipeline stage,
  resource type, and station direction exposed by the previous event. This helps
  RNN policies learn to stop repeating already-satisfied deliveries and to
  respond to sync/dependency events without changing the base environment API.
  Requesting Pipeline feedback implies the parent recurrent `--obs-feedback`
  block in the training CLI.
- `--obs-pipeline-shared-feedback` keeps the same feedback width but fills those
  Pipeline event slots from all agents' previous Pipeline events. This gives
  recurrent policies public progress feedback while private blueprints still
  have to move through observations/messages. It also implies
  `--obs-pipeline-feedback`.
- `--bc-calibrate-pipeline-interact-gate-threshold` calibrates
  `--eval-pipeline-interact-gate-threshold` after BC from the learned
  hidden-state gate probabilities. By default it matches the demo valid-interact
  label rate; use `--bc-pipeline-interact-gate-threshold-target-rate` to tune a
  more permissive or stricter gated decoder.
- `--obs-pipeline-features` adds a compact recurrent observation block decoded
  from local observations, private Pipeline hints, and planner messages:
  active-station direction, needed resource types, held-resource match,
  held-target direction, pickup/delivery affordances, and wrong-station interact
  context.
- `--obs-pipeline-progress-features` appends an opt-in durable Pipeline progress
  block for recurrent policies: completed-stage fractions, current-stage
  dependency/readiness/sync flags, delivered/remaining requirement fractions,
  and remaining/delivered resource masks. It uses event-derived progress state,
  not simulator-only hidden stage objects.
- `--pipeline-wrong-delivery-penalty` controls the environment reward penalty
  for invalid station interactions with carried resources. It is part of the
  Pipeline task surface and should be recorded in comparable eval specs.
- `--eval-pipeline-navigation-assist` is an opt-in diagnostic decoder assist:
  it trusts private Pipeline hints by default and uses the local action mask to
  steer required-resource pickup, active-station delivery, and wrong-station
  suppression. Add `--eval-pipeline-navigation-assist-trust-messages` only when
  you want the assist to trust learned planner messages without a matching hint.
  Treat assisted scores separately from plain benchmark scores.
- `--eval-pipeline-interact-gate-threshold` is an opt-in recurrent eval decoder
  guard for Pipeline. If set to `>=0`, station `INTERACT` actions below the
  learned hidden-state gate probability are converted to `STAY`. This is useful
  for diagnosing whether the policy has learned when interaction is actually
  valid, but it should be reported as a gated-decoder score.
- `--eval-pipeline-station-interact-guard` is an opt-in deterministic eval
  guard for Pipeline station actions. It only touches attempted station
  `INTERACT` actions, suppressing them with trusted private-hint/progress
  state when they cannot deliver, sync, or complete. It is narrower than full
  navigation assist and should still be reported separately from plain scores.
- `--eval-pipeline-plan-head-threshold` is an opt-in recurrent eval decoder for
  the learned plan head. If set to `>=0`, the plan head may replace the base
  action with a non-`STAY` action when a trusted Pipeline plan is visible and
  the head confidence clears the threshold. Report these results separately
  from plain benchmark scores.
- `--eval-pipeline-option-threshold` is an opt-in recurrent eval decoder for
  the learned option head. If set to `>=0`, confident Pipeline options are
  decoded into allowed low-risk primitive actions using the visible trusted
  plan. Direct option-driven DELIVER/SYNC station `INTERACT` promotion is
  disabled by default; add `--eval-pipeline-option-allow-interact` only for
  ablations that intentionally test that high-risk decoder path. Report these
  results separately from plain benchmark scores.
- `--rl-rollout-pipeline-navigation-assist` applies the same Pipeline assist
  only during recurrent PPO rollout collection. This is the preferred guided
  fine-tuning mode when you want unassisted eval plots/checkpoint selection but
  still want high-reward assisted trajectories to shape the policy. Pair it with
  `--rl-rollout-eval-decoding`; use
  `--rl-rollout-pipeline-navigation-assist-trust-messages` only for runs that
  intentionally train from teammate-message plans.
- `--rl-eval-decoding-action-loss-weight` adds an opt-in PPO auxiliary loss on
  actions changed by rollout eval-decoding. This is useful when assisted PPO
  rollouts produce good trajectories but plain eval does not inherit the
  corrected actions strongly enough.
- `--rl-pipeline-assisted-action-loss-weight` adds the broader Pipeline version
  of that distillation signal: it trains on the final post-assist rollout action
  across trusted plan/navigation/sync/guard labels plus actual corrections. Use
  it to teach the plain actor the behavior produced by navigation and station
  assists.
- `--rl-pipeline-interact-gate-loss-weight` adds a PPO auxiliary on Pipeline
  station-tile interact/no-interact decisions. It trains both the raw actor
  interact logit and the auxiliary interact-gate head, with optional
  `--rl-pipeline-interact-gate-pos-weight` and
  `--rl-pipeline-interact-gate-neg-weight`. W&B logs positive and negative
  station-interact rates under `train/pipeline_interact_gate_*`, plus rollout
  label counts under `rollout/pipeline_interact_gate_*`.
- `--rl-pipeline-pickup-gate-loss-weight` adds the matching PPO auxiliary for
  resource-tile pickup/no-pickup decisions. This is useful for Pipeline runs
  that pick resources needed by blocked future stages and then produce wrong
  deliveries. W&B logs `train/pipeline_pickup_gate_*` and
  `rollout/pipeline_pickup_gate_*`.
- `--rl-pipeline-delivery-progress-action-loss-weight` adds a PPO action
  auxiliary for agents already carrying a resource required by the trusted
  active Pipeline stage. It trains movement toward the matching station and
  same-station delivery `INTERACT`, which is useful when station guards reduce
  wrong delivery but audits show many missed delivery opportunities.
- `--rl-pipeline-navigation-action-loss-weight` adds a movement-only PPO action
  auxiliary from trusted Pipeline navigation labels. It distills walking toward
  visible required resources or active stations without directly supervising
  high-risk `INTERACT` actions. W&B logs `train/pipeline_navigation_action_*`
  and `rollout/pipeline_navigation_action_labels`.
- `--rl-pipeline-sync-action-loss-weight` adds a PPO auxiliary for sync-stage
  rendezvous labels. Empty-handed agents are trained to move toward or
  `INTERACT` at a sync station once every remaining required resource for that
  stage is already carried by the team, even before the final delivery happens.
  W&B logs `train/pipeline_sync_action_*` and
  `rollout/pipeline_sync_action_labels`.
- `--rl-pipeline-ready-interact-action-loss-weight` adds a sparse positive PPO
  auxiliary for Pipeline rollout states where `INTERACT` is immediately valid:
  delivering a held needed resource at the active station or completing a ready
  sync stage. W&B logs `train/pipeline_ready_interact_action_*` plus
  `rollout/pipeline_ready_interact_action_labels`.
- `--rl-pipeline-station-guard-action-loss-weight` adds a denser PPO auxiliary
  on Pipeline rollout states where station `INTERACT` would be unsafe. It uses
  the same station-guard target action, including safe `STAY`/escape/drop
  actions, and logs `train/pipeline_station_guard_action_*` plus
  `rollout/pipeline_station_guard_action_labels`.
- `--rl-pipeline-wrong-station-recovery-action-loss-weight` adds a narrower
  PPO auxiliary for Pipeline rollout states where an agent is holding a
  resource on a station tile that cannot use it. The target leaves the current
  station toward the known station that still needs that resource, and W&B logs
  `train/pipeline_wrong_station_recovery_action_*` plus
  `rollout/pipeline_wrong_station_recovery_action_labels`.
- `--rl-pipeline-plan-action-loss-weight` adds a PPO auxiliary on Pipeline
  rollout states with a trusted visible plan, training the actor toward the
  local pickup/navigation/delivery/sync action. W&B logs
  `train/pipeline_plan_action_*` and `rollout/pipeline_plan_action_labels`.
- `--rl-pipeline-plan-head-loss-weight` trains the auxiliary recurrent
  `pipeline_plan_policy` head on those same rollout plan-action labels. W&B
  logs `train/pipeline_plan_head_*`.
- `--rl-pipeline-option-loss-weight` trains the auxiliary recurrent
  `pipeline_option_policy` head on high-level rollout options such as pickup,
  deliver, sync, navigation, and wait. W&B logs `train/pipeline_option_*` and
  `rollout/pipeline_option_labels`.
- `--rl-rollout-pipeline-station-interact-guard` applies only the narrow
  station `INTERACT` guard during PPO rollout collection. The guarded recurrent
  PPO sweep profile enables this with `--rl-rollout-eval-decoding` so rollouts
  avoid the clearest Pipeline safety failures while benchmark eval can remain
  unassisted or use its own separately reported guard setting.
- `--rl-pipeline-bad-pickup-penalty`,
  `--rl-pipeline-bad-interact-penalty`, and
  `--rl-pipeline-unneeded-drop-bonus` shape recurrent PPO rewards for confirmed
  Pipeline safety events: no-longer-needed pickups, station interactions that
  cannot deliver/sync/complete, and recovery drops. W&B logs
  `rollout/pipeline_bad_pickups`, `rollout/pipeline_bad_interacts`,
  `rollout/pipeline_unneeded_drops`, and `rollout/pipeline_wrong_deliveries`
  so these runs can be compared against the trajectory audit.
- The trajectory audit reports Pipeline failure types such as `missed_pickup`,
  `missed_delivery`, `sync_wait`, `dependency_blocked`, `wrong_delivery`, and
  `partial_pipeline`, plus aggregate stage-completion and delivery ratios. It
  also reports pickup need-status counts, station delivery-decision counts, and
  wrong-delivery provenance so you can distinguish bad pickups from wrong
  station choices.

PPO stability controls:

- `--rl-restore-best` keeps the best eval checkpoint after PPO fine-tuning.
- `--rl-eval-use-eval-seeds` selects PPO checkpoints using the main eval
  seed/list instead of the separate PPO eval seed/list. This is recommended
  when PPO is part of a benchmark-facing curriculum stage.
- `--rl-early-stop-eval-patience N` stops PPO after `N` eval checkpoints fail
  to improve the best eval score, which is useful when PPO starts degrading a
  strong BC/DAgger policy.
- `--no-dagger-restore-best` makes PPO continue from the latest DAgger round
  instead of restoring the best short-eval DAgger checkpoint. This is useful
  when later DAgger rounds add important failure-state coverage even if their
  immediate eval score is noisier.

Seed schedules:

- `--dagger-seed-list 3000,3001,3002` cycles global DAgger collection seeds.
- `--dagger-seed-list 16:3000,13000+32:7000,17000` cycles collection seeds
  separately per training map size.
- `--eval-seed-list 16:3000,13000+32:7000,17000` applies the same map-specific
  schedule to recurrent init, BC, and DAgger eval.
- `--rl-eval-seed-list ...` overrides `--eval-seed-list` for PPO eval only.

Conservative replay controls:

- `--dagger-max-replay-snippets-per-episode` caps snippets cut from each parent
  rollout.
- `--dagger-max-failed-parent-replay-snippets-per-episode` optionally applies a
  stricter cap to snippets from failed model-visited rollouts.
- `--dagger-failed-parent-replay-weight-scale` downweights those failed-parent
  snippets without changing successful-parent or expert replay snippets.
- `--dagger-failed-episode-weight` controls the full failed parent rollout
  weight itself.

Large-map Signal Hunt search auxiliary:

- The sweep-level `--recurrent-signal-preset large_map` preset inherits the
  current Signal specialist defaults, then enables the large-map discovery and
  decoy-discrimination path for new 32x32 ablations. It resolves to visible
  clue action loss `0.25` for maps at least 32x32, keeps frontier exploration
  action loss at `0.25` for maps at least 16x16, adds decoy-drift action loss
  `0.1`, decoy-scan action loss `0.25`, and DAgger focus/replay for
  `visible_clue_miss`, `decoy_scan`, and `rejected_target_scan`. A first
  heavier version over-weighted rejected-target drift and regressed DAgger
  eval, so the current preset is deliberately conservative. The tuned run
  `recurrent_signal_multisize_16_32_large_map_conservative_seed0` reached
  88/100 on the independent 16x16 audit and 55/100 on the independent 32x32
  audit, with W&B run
  `https://wandb.ai/orion8/syncorsink-core-training/runs/bggn5dip`.
  A follow-up scan-conversion ablation,
  `recurrent_signal_multisize_16_32_large_map_scanconvert_seed0`, appended
  `first_target_scan`/`joint_target_scan` positive replay and moderately raised
  target-scan action weights. It regressed the full audits to 84/100 at 16x16
  and 49/100 at 32x32 despite a promising 20-episode eval slice, so it is not
  part of the preset.
- `--bc-signal-target-pursuit-trust-exact-memory` lets target-pursuit action
  labels use trusted exact target messages retained by scan-state memory. This
  is an opt-in ablation rather than a default because the first 16x16 run did
  not improve the promoted Signal profile. `--bc-signal-target-pursuit-max-agents`
  optionally caps those labels to the closest responders; `0` keeps all
  eligible agents. The 16/32 large-map target-memory runs also underperformed:
  broad exact-memory labels (`8ibvfhg2`) reached 65% mixed eval, and the cap-1
  variant (`zw2fadn1`) reached 62.5% mixed eval with only 30% 32x32 DAgger
  success, so neither is promoted.
- `--recurrent-obs-exploration-age` forwards per-cell exploration recency into
  recurrent training. It remains opt-in: the large-map age ablation
  `recurrent_signal_multisize_16_32_large_map_age_seed0` logged to
  `https://wandb.ai/orion8/syncorsink-core-training/runs/ekvcpsb2` but reached
  only 57.5% mixed eval, with 40% 32x32 split success.
- `--bc-signal-constraint-frontier-bias` is an opt-in Signal search-label
  ablation that scores frontier targets by their distance to bounded inferred
  target candidates when the agent has strong partial clue evidence but no
  exact target. It intentionally reuses the target-decoding candidate cap so
  weak one-clue or parity-only states keep using broad coordinated search. The
  first permissive 16/32 run
  `recurrent_signal_multisize_16_32_large_map_constraintfrontier_seed0` logged
  to `https://wandb.ai/orion8/syncorsink-core-training/runs/bs4stt2m`, but it
  reached only 60% mixed eval with 45% 32x32 split success and lower target-scan
  counts, so the tightened bias remains diagnostic rather than part of
  `large_map`. The tightened rerun
  `recurrent_signal_multisize_16_32_large_map_constraintfrontier_tight_seed0`
  logged to `https://wandb.ai/orion8/syncorsink-core-training/runs/xy4w9cpc`
  and improved mixed eval to 67.5% by raising 16x16 to 90%, but 32x32 remained
  at 45%.
- `--dagger-signal-target-rendezvous-labels` is an opt-in post-discovery
  Signal Hunt label for large maps. When at least two agents have exact target
  evidence through private clues or scan-state memory, DAgger/demo collection
  routes the closest target-informed pair toward the true target. If one
  selected agent is already on the target but the partner cannot still arrive
  and scan inside the scan window, the label makes the early scanner wait
  instead of converting a likely solo target scan. The sweep wrapper exposes it
  as `--recurrent-dagger-signal-target-rendezvous-labels`, with
  `--recurrent-dagger-signal-target-rendezvous-min-map-size` and
  `--recurrent-dagger-signal-target-rendezvous-max-agents` controlling where
  and how many target-informed agents are selected. It is not part of
  `large_map`: the isolated 16/32 run
  `recurrent_signal_multisize_16_32_large_map_targetrendezvous_seed0` logged to
  `https://wandb.ai/orion8/syncorsink-core-training/runs/vde2xcu5` and reached
  only 60% mixed eval, with 80% on 16x16 and 40% on 32x32. It reduced decoy
  scanning but also reduced 32x32 true-target reach, so no full audit was run.
- A discovery-side `clue_found` positive-replay ablation,
  `recurrent_signal_multisize_16_32_large_map_cluepositive_seed0`, logged to
  `https://wandb.ai/orion8/syncorsink-core-training/runs/tuudojwi` and also
  reached only 60% mixed eval, with 80% on 16x16 and 40% on 32x32. The event
  path was active (`36` positive replay events and `11` replay triggers in
  round 2), but 32x32 decoy scans rose to `6.7`, so clue replay is not promoted
  into the large-map preset. Retesting this idea after the negative-memory scan
  guard did not rescue it:
  `recurrent_signal_multisize_16_32_large_map_cluepositive_negguard_seed0`
  logged to `https://wandb.ai/orion8/syncorsink-core-training/runs/uazym681`
  and fell to 47.5% mixed eval with a 65%/30% 16x16/32x32 split. Wrong scans
  stayed at zero, but first-target-scan misses rose sharply, so clue replay
  alone creates target visits without reliable scan conversion.
- `--bc-signal-ambiguous-target-decision-negatives` is an opt-in target-decision
  label ablation. It treats visible true-target scans as negative target
  decisions when the observation cannot uniquely justify that center target,
  avoiding hidden-oracle positives for locally ambiguous target hypotheses.
  The sweep flag is
  `--recurrent-bc-signal-ambiguous-target-decision-negatives`, with
  `--recurrent-bc-signal-ambiguous-target-decision-min-map-size` controlling
  where it starts. The isolated 16/32 run
  `recurrent_signal_multisize_16_32_large_map_ambiguousdecision_seed0` logged
  to `https://wandb.ai/orion8/syncorsink-core-training/runs/7dr55x28` and
  matched the conservative 70% mixed eval headline, but did not improve 32x32:
  the best split was 90% on 16x16 and 50% on 32x32. It can suppress decoy scans
  but raises first-target-scan misses, so no full audit was run. A moderate
  scan-pressure follow-up,
  `recurrent_signal_multisize_16_32_large_map_ambiguousdecision_scanpush_seed0`,
  logged to `https://wandb.ai/orion8/syncorsink-core-training/runs/8tbyh3s6`
  and regressed to 65% mixed eval with at most 45% on 32x32.
- `--bc-signal-ambiguous-target-search-labels` is an opt-in follow-up to the
  ambiguous target-decision ablation. When an agent is standing on a compatible
  but locally non-unique target, visible-clue labels may still fire, and if no
  clue is visible the frontier label may move the agent off the ambiguous target
  to gather more evidence. The sweep flag is
  `--recurrent-bc-signal-ambiguous-target-search-labels`, with
  `--recurrent-bc-signal-ambiguous-target-search-min-map-size` controlling
  where it starts. The 32x32-only run
  `recurrent_signal_multisize_16_32_large_map_ambiguoussearch32_seed0` logged to
  `https://wandb.ai/orion8/syncorsink-core-training/runs/5g5q8b1l` and matched
  the prior 70% mixed best eval with a 90%/50% 16x16/32x32 split, but did not
  improve the 32x32 slice. The final round regressed to 62.5% mixed eval and
  reintroduced 2.15 average wrong-target scans, so the flag remains diagnostic.
- `--obs-signal-confidence-features` appends a 14-value Signal observation block
  describing the current center target hypothesis: exact, unique/safe,
  compatible, ambiguous, rejected, unknown, whether target information and
  constraints are present, and normalized evidence counts. The sweep flag is
  `--recurrent-obs-signal-confidence-features`. The run
  `recurrent_signal_multisize_16_32_large_map_confidence_seed0` logged to
  `https://wandb.ai/orion8/syncorsink-core-training/runs/0mycw4ng` and reached
  only 60% mixed eval with a 75%/45% 16x16/32x32 split. It reduced wrong-target
  scans to zero, but did not improve large-map discovery, so it remains
  diagnostic rather than a promoted large-map default.
- `--bc-signal-target-hypothesis-loss-weight` enables an opt-in recurrent
  Signal auxiliary head that predicts whether the current evidence is enough to
  commit, how ambiguous the target hypothesis remains, and the normalized true
  target coordinate. Labels are emitted only when an observation or received
  message contains target evidence compatible with the true target; no-info
  observations stay masked out. Use
  `--bc-signal-target-hypothesis-min-map-size` or the sweep flag
  `--recurrent-bc-signal-target-hypothesis-min-map-size` to restrict this to
  larger maps. The commit, ambiguity, and coordinate sub-losses can be balanced
  with `--bc-signal-target-hypothesis-commit-loss-weight`,
  `--bc-signal-target-hypothesis-ambiguity-loss-weight`, and
  `--bc-signal-target-hypothesis-xy-loss-weight` while the top-level loss weight
  remains the master switch. W&B logs `bc/signal_target_hypothesis_count`,
  `bc/signal_target_hypothesis_loss`,
  `bc/signal_target_hypothesis_commit_acc`,
  `bc/signal_target_hypothesis_ambiguity_l1`, and
  `bc/signal_target_hypothesis_xy_l1`, plus demo/DAgger label mix fields for
  commit labels, ambiguous labels, commit rate, and mean ambiguity. This is
  implemented for measurement and ablation; it is not yet used as a decoding
  override. The all-components 0.05 run
  `recurrent_signal_multisize_16_32_large_map_targethypothesis005_seed0` reached
  65% mixed eval. The coordinate-only follow-up
  `recurrent_signal_multisize_16_32_large_map_targethypothesis_xyonly005_seed0`
  logged to `https://wandb.ai/orion8/syncorsink-core-training/runs/kwpy80zv` and
  reached 70% mixed eval with an 85%/55% 16x16/32x32 split. Its DAgger label mix
  was about 16.2k labels, 69.6% commit labels, and 0.115 mean ambiguity, so the
  component split remains useful for ablation but is not a promoted default yet.
- `--obs-signal-sector-features` appends a 10-value Signal observation block for
  role-aware search: assigned-frontier direction, global and assigned-sector
  explored fractions, whether an assigned frontier exists, whether the agent is
  currently inside its assigned sector, and normalized anchor/frontier
  distances. The sweep flag is `--recurrent-obs-signal-sector-features`. The
  isolated run `recurrent_signal_multisize_16_32_large_map_sector_seed0` logged
  to `https://wandb.ai/orion8/syncorsink-core-training/runs/lqyp8kkf` and
  reached 60% mixed eval with a 75%/45% 16x16/32x32 split. It also drove
  wrong-target scans to zero but did not improve the 32x32 success slice, so the
  flag remains diagnostic.
- `--bc-signal-constraint-message-loss-weight` adds an auxiliary token/length
  BC loss for non-initial Signal messages that carry structured clue/constraint
  segments (`21`-`25`). This targets the post-clue broadcast bottleneck without
  mixing it into the step-0 private-hint loss. The base trainer default is
  `0.0`; the recurrent Signal specialist sweep preset defaults to `4.0`.
  W&B logs `bc/signal_constraint_message_count`,
  `bc/signal_constraint_message_loss`,
  `bc/signal_constraint_message_exact_rate`, and
  `bc/signal_constraint_message_token_acc`. The isolated large-map run
  `recurrent_signal_multisize_16_32_large_map_constraintmsg_seed0` logged to
  `https://wandb.ai/orion8/syncorsink-core-training/runs/rc4pmvoq` and reached
  67.5% mixed eval with an 85%/50% 16x16/32x32 split. It improved clue finding
  but reintroduced 32x32 wrong-target scans (`7.55` average), so the next
  Signal target is confidence-gated scanning/decoy suppression rather than more
  message-token supervision.
- `--bc-signal-visible-clue-action-weight` enables a BC action loss for agents
  that can locally see an unclaimed clue but do not yet have a unique known
  target. The label moves toward the nearest visible clue or interacts if the
  agent is already standing on it. This is currently opt-in because the first
  16x16 validation improved clue collection but regressed final success.
- `--bc-signal-visible-clue-min-map-size` defaults to `16` in the trainer CLI;
  the sweep `large_map` preset raises it to `32` so mixed 16/32 runs target the
  visible-clue auxiliary at 32x32 without changing the promoted 16x16 profile.
- `--bc-signal-frontier-exploration-action-weight` enables an additional BC
  action loss for states with exploration memory, no visible clue/target, and
  no unique known target. On large multi-agent maps, labels use coordinated
  frontier anchors to spread agents across the search area instead of sending
  every agent to the same nearest frontier.
- `--bc-signal-frontier-exploration-min-map-size` defaults to `16`, keeping the
  auxiliary focused on larger maps where no-clue/no-target timeouts dominate.
- `--eval-signal-frontier-exploration-assist` is an opt-in eval/rollout assist
  that applies the same Signal frontier policy at decode time when no visible
  clue, target, or unique target evidence exists. Use the sweep flag
  `--recurrent-eval-signal-frontier-exploration-assist` to diagnose whether
  remaining large-map failures come from learned exploration execution versus
  downstream clue/scan conversion. It remains default-off after a 16x16 audit
  reached 56/100 versus the 57/100 rolefix-handoff baseline.

Checkpoint workflow:

- Use `--recurrent-init CHECKPOINT --recurrent-init-for-dagger` to continue
  BC/DAgger from an existing recurrent actor checkpoint.
- Use `--skip-bc --recurrent-init CHECKPOINT` for pure PPO fine-tuning from a
  checkpoint.
- Checkpoints should stay out of git; store public model files in an external
  artifact store and reference them from result artifacts.

`examples/core_training_sweep.py` launches the public training CLIs across the
core 8x8 cases and writes one manifest:

- `suite_summary.json`
- per-run `run_summary.json`
- per-run `stdout.log` and `stderr.log`
- per-run checkpoint under `checkpoints/`
- aggregate mean eval metrics per algorithm/scenario across `--seeds`
- per-run W&B status, including captured init failures

The child trainers do not all expose a `--wandb-mode` flag, so the sweep runner
sets `WANDB_MODE` for them. Use `--wandb-mode disabled` for a pure local
checkpoint pipeline smoke, `offline` for local W&B runs, and `online` after
`wandb login`. Add `--strict-wandb` when a requested W&B run must fail fast if
the W&B process cannot initialize.

For MAPPO runs, `examples/core_training_sweep.py` also forwards architecture,
exploration-memory, and eval-decoding controls through `--mappo-*` flags. Use
`--mappo-obs-exploration-memory` for Signal Hunt memory baselines and record
send-gate choices with `--mappo-eval-send-mode` / `--mappo-eval-send-threshold`.

The sweep also supports `--algorithms recurrent_bc_rl` for the recurrent
BC/DAgger/PPO trainer. Recurrent runs use `--recurrent-*` flags for oracle,
demo count, BC epochs, DAgger rounds/focus settings, Pipeline wrong-delivery
provenance labels, checkpoint init, train/eval map schedules, Signal Hunt
specialist observation/eval assists, and PPO stability controls.
For Signal Hunt specialist runs, the sweep defaults
`--recurrent-bc-signal-target-aux-weight` to `0.25`, adding auxiliary
supervision for fusing private clues into the exact target location. It also
defaults Signal action auxiliaries to the current Signal specialist profile:
target-pursuit `0.4`, target-match `0.4`, first target scan `0.8`, refresh scan
`0.3`, joint target scan `0.5`, target opportunity `0.4`, sync response `0.2`,
and frontier exploration `0.25` for maps at least 16x16. The preset also turns
on scan-decision, scan-gate, target-validity, and target-decision auxiliaries;
their loss/positive/negative weights can be overridden with
`--recurrent-bc-signal-scan-*`, `--recurrent-bc-signal-target-validity-*`, and
`--recurrent-bc-signal-target-decision-*` flags for ablations. Specialist runs turn
on `--obs-agent-id-features` so a shared recurrent actor can learn stable
search-sector roles on large maps during both BC/DAgger training and eval, and
set
`--eval-signal-target-scan-threshold 0.0`, which only forces `INTERACT` when the
agent is standing on an observation-safe center target candidate. The guarded
profile replays Signal target-discovery, decoy-drift, frontier-exploration, and
target-handoff misses, and includes positive target-handoff snippets so larger
maps emphasize both search recovery and post-discovery scan completion.
`--recurrent-bc-signal-active-scan-response-action-weight` is an opt-in
post-discovery ablation. When a teammate's true-target scan is still active, it
labels the closest trusted exact-target-informed responder to join/scan the
target, with minimum-map-size and max-responder caps exposed by the matching
`--recurrent-bc-signal-active-scan-response-*` flags. It is disabled by default
after the matched active-scan run improved the learned label fit but reached
only 65% mixed eval, with 87.5% on 16x16 and 42.5% on 32x32.
`--recurrent-bc-signal-clue-interact-action-weight` is an opt-in acquisition
ablation for agents already standing on an unclaimed clue. It adds a local
`INTERACT` action label, with
`--recurrent-bc-signal-clue-interact-min-map-size` controlling where it starts.
The matched 16/32 run
`recurrent_signal_multisize_16_32_large_map_clueinteract_negguard_seed0`
logged to `https://wandb.ai/orion8/syncorsink-core-training/runs/ofakn8vg`.
The auxiliary fired (`296` and `211` DAgger collection labels across the two
rounds) and the final BC pass reached `0.767` label action-match, but the run
ended at 67.5% mixed eval with an 82.5%/52.5% 16x16/32x32 split. It remains
opt-in because it did not beat the conservative 55% 32x32 baseline.
`--recurrent-bc-signal-scan-bridge-action-weight` is an opt-in post-discovery
ablation for an agent already standing on the true target while its own scan
window is partially expired and an exact-target-informed teammate is nearby.
It logs `bc/signal_scan_bridge_*` counts, action-match, action probability, and
configuration thresholds. The matched run
`recurrent_signal_multisize_16_32_large_map_scanbridge_negguard_seed0` logged
to `https://wandb.ai/orion8/syncorsink-core-training/runs/29z8hpul`. The
bridge labels fired (`51` and `42` DAgger collection labels) and were learned
quickly, but the saved best checkpoint reached only 53.75% mixed eval with a
70%/37.5% 16x16/32x32 split; later DAgger rounds improved clue collection but
fell to 51.25% mixed and 25% at 32x32. It remains a negative diagnostic rather
than a large-map default.
Signal specialist runs also default
`--recurrent-eval-signal-constraint-message-copy-assist` on, which forwards
`--eval-signal-constraint-message-copy-assist` and replaces sent Signal clue
messages with canonical structured constraints from the sender's current
`goal_hint`. This prevents wrong generated clue tokens from causing teammates to
reject a true target that satisfies the actual private/collected clues. On the
rolefix-handoff 16x16 checkpoint, the matched seed-3000 audit improved from
56/100 to 77/100 and reduced `no_target_scan` failures from 23 to 9.
The separate `--recurrent-eval-signal-constraint-message-guard` forwards
`--eval-signal-constraint-message-guard` and drops learned structured
constraint-message segments when the sender has no current private/collected
constraint to support them. It now defaults on for `--recurrent-signal-preset
large_map`: the matched xy-only target-hypothesis eval improved mixed 16/32
success from 70.0% to 77.5%, held 16x16 at 85%, raised 32x32 from 55% to 70%,
and removed the 32x32 true-target rejected-observation bucket.
Large-map Signal also defaults
`--recurrent-eval-signal-exact-target-message-copy-assist` on, forwarding
`--eval-signal-exact-target-message-copy-assist`. This broadcasts `[26, x, y]`
from any single trusted exact target, not only at step 0 or after an active scan.
On the same xy-only checkpoint it improved mixed 16/32 eval to 80.0%, holding
16x16 at 85% and lifting 32x32 to 75% (`a3lqu8q6`). Combining it with
`--recurrent-eval-signal-scan-refresh-assist` remained flat at 77.5%
(`gfcsjawy`), so scan refresh stays opt-in despite the decode integration fix
that lets exact-target navigation honor expiring solo-scan refreshes.
A fresh comparable rolefix-profile training run with this default enabled
(`recurrent_signal16_constraintcopy_rolefix_profile_seed0`) reached 79/100 on
the same 100-episode seed-3000 audit, with failure mix
`no_clue_or_target_scan`: 17, `no_target_scan`: 2, `solo_target_scan`: 2, and no
decoy failures. It logged online to W&B at
`https://wandb.ai/orion8/syncorsink-core-training/runs/04h8pmkn`. A 32x32,
4-agent audit of that checkpoint reached 41/100, with failures dominated by
`no_clue_or_target_scan`: 51. Signal inferred-target features and eval decoding
now compile each observation's structured clue/message constraints once and
reuse that compiled state for target filtering; after that optimization the
32x32 100-episode audit completed in 39.90s. Large-map discovery is therefore
the next implementation bottleneck rather than audit runtime.
The official 2-agent Signal 8x8 case now has a current recurrent checkpoint:
`recurrent_signal8_constraintcopy_rolefix_profile_seed0` reached 98/100 on a
100-episode seed-3000 trajectory audit, with W&B run
`https://wandb.ai/orion8/syncorsink-core-training/runs/hzm8kztx`.
A first 16/32 mixed curriculum
(`recurrent_signal_multisize_16_32_constraintcopy_seed0_v3`) logged to
`https://wandb.ai/orion8/syncorsink-core-training/runs/79ukojb9` and reached
84/100 on the independent 16x16 audit but stayed at 41/100 on 32x32. The 32x32
failure mix shifted toward `no_clue_or_target_scan`, `no_target_scan`, and
`decoy_scan`, so `--recurrent-signal-preset large_map` was added instead of
relying on map-size mixing alone. The tuned large-map run
(`recurrent_signal_multisize_16_32_large_map_conservative_seed0`) reached
88/100 on 16x16 and 55/100 on 32x32. The remaining 32x32 failure mix is
`no_clue_or_target_scan`: 18, `no_target_scan`: 22, `solo_target_scan`: 2, and
`decoy_scan`: 3, making post-discovery target conversion and large-map teammate
routing the next bottleneck. The direct scan-positive ablation
`recurrent_signal_multisize_16_32_large_map_scanconvert_seed0` logged to
`https://wandb.ai/orion8/syncorsink-core-training/runs/duosomim` but regressed
the independent audits to 84/100 at 16x16 and 49/100 at 32x32; this suggests
the next fix should be teammate routing/role assignment after target evidence
rather than simply larger scan-action weights.
The opt-in `--recurrent-bc-signal-target-pursuit-trust-exact-memory` ablation
lets target-pursuit action labels consume trusted exact target messages retained
by the Signal scan state, so DAgger rows can teach agents to keep moving toward
a teammate-broadcast target after the raw inbox message expires. It is not part
of the default specialist preset after the first 16x16 run landed below the
current rolefix-handoff baseline. The sweep-level
`--recurrent-bc-signal-target-pursuit-max-agents` flag forwards the optional
nearest-responder cap; the 16/32 cap-1 run
`recurrent_signal_multisize_16_32_large_map_targetmemory_cap1_seed0` logged to
`https://wandb.ai/orion8/syncorsink-core-training/runs/zw2fadn1` but reached
only 62.5% mixed eval, so it is diagnostic rather than a default.
The exploration-age ablation `recurrent_signal_multisize_16_32_large_map_age_seed0`
logged to `https://wandb.ai/orion8/syncorsink-core-training/runs/ekvcpsb2`; it
raised 32x32 clue count but lowered 32x32 success to 40%, so it is also
diagnostic rather than a large-map preset default.
Constraint-compatible frontier labels
(`recurrent_signal_multisize_16_32_large_map_constraintfrontier_seed0`,
`https://wandb.ai/orion8/syncorsink-core-training/runs/bs4stt2m`) reached only
60% mixed eval and 45% 32x32 split success, so
`--recurrent-bc-signal-constraint-frontier-bias` stays opt-in; the current
implementation now gates the bias through the same bounded inferred-target
candidate rule as target decoding. The tightened rerun
(`recurrent_signal_multisize_16_32_large_map_constraintfrontier_tight_seed0`,
`https://wandb.ai/orion8/syncorsink-core-training/runs/xy4w9cpc`) improved mixed
eval to 67.5% but left the 32x32 split at 45%, so the flag is still diagnostic
rather than a large-map default. The matched Signal-anchor fallback isolation run
(`recurrent_signal_multisize_16_32_large_map_signalfrontier_seed0`,
`https://wandb.ai/orion8/syncorsink-core-training/runs/fil29d02`) reached 62.5%
mixed eval but only 30% on the 32x32 split with many decoy target visits. The
default frontier fallback therefore remains the conservative anchored behavior
unless the constraint-frontier flag is explicitly enabled.
The opt-in `--recurrent-eval-signal-frontier-exploration-assist` ablation
forwards `--eval-signal-frontier-exploration-assist`, reusing the same
coordinated frontier policy as the large-map search labels during eval/rollout
decoding. Its first rolefix-handoff checkpoint audit was essentially flat
negative (56/100 versus 57/100 baseline), so it is diagnostic rather than
promoted.
Visible-clue pursuit/replay remains available as an explicit ablation flag, but
is not promoted by default. The `--recurrent-eval-signal-target-scan-lock` sweep
flag forwards `--eval-signal-target-scan-lock`, preserving target-scan actions
against learned scan/validity/decision suppressors; it remains opt-in because
the first 16x16 audit kept success flat while increasing decoy-scan events. Set
any aux weight to `0.0` for ablations, pass
`--no-recurrent-obs-agent-id-features` to remove role cues, or set the scan
threshold to `-1` to disable this narrow decode correction. The
`--recurrent-dagger-target-handoff-requires-exact-target` ablation forwards
`--dagger-target-handoff-requires-exact-target`, which keeps target-handoff
labels only when the responder has trusted exact target evidence rather than a
merely compatible visible target. It remains opt-in: the first 16x16 audit
reduced success from 57/100 to 53/100 and shifted failures toward missed
discovery/no-scan cases.
`--recurrent-eval-signal-compatible-target-scan-assist` is available as an
opt-in diagnostic for visible target tiles that satisfy non-unique clue
constraints, but it did not improve the first 16x16 rolefix-handoff audit once
constraint-message copy was enabled. `--recurrent-eval-signal-negative-memory-scan-guard`
is a narrower opt-in follow-up: it forwards
`--eval-signal-negative-memory-scan-guard`, suppressing a target-scan `INTERACT`
when the visible center target is still in Signal negative memory as a prior
decoy unless exact target-scan lock trusts it. This is intended for the next
32x32 over-scan ablation after
`recurrent_signal_multisize_16_32_large_map_constraintmsg_seed0`
(`https://wandb.ai/orion8/syncorsink-core-training/runs/rc4pmvoq`), where
constraint-message supervision improved the mixed 16/32 score to 67.5% but left
32x32 at 50% with 7.55 wrong scans per episode. The matched guarded rerun
`recurrent_signal_multisize_16_32_large_map_constraintmsg_negguard_seed0`
(`https://wandb.ai/orion8/syncorsink-core-training/runs/8i0re2bp`) kept mixed
success at 67.5% with the same 85%/50% 16x16/32x32 split, but reduced 32x32
wrong-target scans to 0.5 per episode. The remaining 32x32 bottleneck is
therefore true-target acquisition/coordinated scanning rather than decoy-scan
suppression alone. The next narrow sweep lever is
`--recurrent-eval-signal-scan-refresh-assist`, which forwards
`--eval-signal-scan-refresh-assist` plus
`--eval-signal-scan-refresh-threshold`; it re-triggers an expiring own target
scan from scan-state feedback when no teammate scan is active. Direct 32x32
checkpoint audits on the guarded run kept success at 10/20 both without refresh
(`logs/trajectory_audit/signal_32_guard_baseline_eval_seed3000`) and with
refresh (`logs/trajectory_audit/signal_32_guard_refresh_eval_seed3000`);
refresh raised target scans slightly (`2.1` to `2.35`) but did not change the
failure mix (`9` never reached target, `4` no clue/target scan, `3` no target
scan, `1` solo target scan, `2` decoy-scan failures). This points the next
implementation target back to large-map clue/target acquisition.
`--recurrent-eval-signal-target-probe-assist` is another opt-in diagnostic for
this failure mode. It forwards `--eval-signal-target-probe-assist`, which
probe-scans a visible center target only when it is not rejected, not in active
negative memory, and not already under this agent's active scan. Direct 32x32
audits showed it is not enough by itself: the guarded checkpoint stayed at
10/20 successes (`logs/trajectory_audit/signal_32_guard_probe_eval_seed3000`),
and the clue-positive checkpoint stayed weak at 6/20
(`logs/trajectory_audit/signal_32_cluepositive_probe_eval_seed3000`).
Use `--recurrent-backbone mlp` for the legacy two-layer flat encoder or
`--recurrent-backbone residual_mlp` for the LayerNorm residual flat encoder.
Use `--recurrent-backbone local_cnn` to encode the local grid/resource/node/
energy FOV planes with a small CNN before fusing them with message, hint,
memory, feedback, and action-mask features. Checkpoints record the recurrent
backbone and resume only into matching backbone configs.
By default, `--recurrent-oracle auto` uses the Signal Hunt specialist oracle for
Signal Hunt and the scenario planner communication teachers for Energy Grid and
Pipeline Assembly. The default `--recurrent-ppo-profile guarded` applies the
safer PPO recipe from the Signal Hunt ablation: lower LR/clip, stronger
BC/communication KL, balanced rollouts, and eval-decoding rollouts. For
Pipeline Assembly it also enables pickup/delivery/progress supervision,
plan/option/message distillation, station-guard action supervision,
  interact-gate supervision with calibrated threshold selection, proactive
  bad-action labels, focused DAgger replay for delivery misses/wrong deliveries,
  pre-delivery ready-state replay via `pipeline_delivery_ready`,
  wrong-delivery provenance labels, and rollout station-interact guarding. Use
`--recurrent-ppo-profile standard` to recover the trainer defaults, or override
individual knobs such as `--recurrent-rl-lr`, `--recurrent-clip`,
`--recurrent-bc-kl-coeff`, and `--recurrent-bc-comm-kl-coeff`. Use
`--recurrent-bc-pipeline-bad-action-margin-loss-weight` for explicit Pipeline
bad-action ablations; it is exposed for wrong-station `INTERACT` experiments but
is not part of the guarded default profile.
Use `--recurrent-bc-pipeline-wrong-station-recovery-action-loss-weight` for
explicit wrong-station movement-recovery ablations; it is logged but not part of
the guarded default profile.
Use `--recurrent-rl-pipeline-wrong-station-recovery-action-loss-weight` for the
matching PPO rollout-state recovery auxiliary after DAgger has exposed
wrong-station failures.
Use `--recurrent-bc-pipeline-ready-interact-action-loss-weight` and
`--recurrent-rl-pipeline-ready-interact-action-loss-weight` for positive
ready-delivery/sync interaction ablations when policies avoid `INTERACT` too
often after station guards are enabled.
Use `--recurrent-bc-pipeline-navigation-action-loss-weight` for explicit
trusted-plan movement distillation ablations; it is logged but not part of the
guarded default profile.
Use `--recurrent-rl-pipeline-delivery-progress-action-loss-weight` and
`--recurrent-rl-pipeline-navigation-action-loss-weight` for the matching PPO
rollout-state action auxiliaries when audits show carried resources or movement
plans are visible but the plain actor still misses delivery windows.
Use `--recurrent-bc-pipeline-frontier-exploration-action-loss-weight` for
explicit Pipeline resource-search ablations with exploration memory. It labels
frontier moves when the current trusted plan needs an unseen resource and is
logged separately from visible-resource navigation.
Use `--recurrent-bc-pipeline-sync-action-loss-weight` for explicit sync-stage
rendezvous ablations. It labels empty agents once every remaining required
resource for a sync stage is already carried by the team, then supervises
movement to the station and same-tile `INTERACT`; it is logged but not part of
the guarded default profile.
Use `--recurrent-rl-pipeline-sync-action-loss-weight` for the matching PPO
rollout-state rendezvous auxiliary when assisted rollouts reach sync stations
but the plain actor fails to retain those actions.
Use `--recurrent-rl-eval-decoding-action-loss-weight` with rollout eval-decoding
ablations when you want PPO to imitate decoder-corrected actions directly.
Use
`--recurrent-rl-early-stop-eval-patience` to tune PPO early stopping; the
guarded profile defaults this to `4`, while the standard profile disables it.
Use `--no-recurrent-dagger-retrain-from-scratch` only as an explicit ablation;
the guarded profile keeps scratch DAgger retraining by default after current
Pipeline runs showed continuation was weaker.
Use `--recurrent-init-template` with `{seed}` to fine-tune each seed from its own
BC/DAgger checkpoint. For non-Signal scenarios,
`--recurrent-bc-calibrate-send-threshold` passes the recurrent trainer's
post-BC communication threshold calibration into the sweep;
`--recurrent-bc-send-threshold-target-rate` and
`--recurrent-bc-comm-send-rate-*` expose the trainer's existing send-rate
controls. For audit-panel PPO selection, `--recurrent-eval-seed-range 3000:40`
expands to `--eval-seed-list 3000,...,3039`, sets recurrent final/PPO evals to
one episode per seed unless explicitly overridden, and enables PPO best
checkpoint selection on the same eval seed panel. Map-specific ranges use
`MAP_SIZE=START:COUNT`, such as `16=13000:40+32=17000:40`. Recurrent JSON eval
output and saved checkpoint eval
metadata are normalized into the same aggregate `success_rate`, `return`, and
`steps` fields as MAPPO/Comm-MAT/TarMAC. When recurrent PPO restores the best
checkpoint before saving, `eval_metrics` describes that saved checkpoint;
recurrent summaries also preserve separate `final_eval_metrics` and
`best_eval_metrics` so PPO regressions are visible. Recurrent trajectory audits
inherit the checkpoint observation contract and send threshold; recurrent
trainer runs that load `--recurrent-init` also inherit checkpoint observation
settings and `eval_send_threshold` when the caller leaves those defaults in
place.
Use `--recurrent-send-threshold` in audits or `--recurrent-eval-send-threshold`
in sweeps only for intentional decode overrides. The sweep exposes recurrent
observation-contract flags such as `--recurrent-obs-exploration-memory`,
`--recurrent-obs-feedback`, `--recurrent-obs-normalize-tokens`,
`--recurrent-obs-navigation-features`, and `--recurrent-obs-signal-*` so PPO
fine-tuning can match checkpoints trained with richer observation blocks.
Recurrent PPO W&B logs also report `rollout/completed_episodes` and
`rollout/partial_segments`; when a
rollout chunk has no completed episodes, `rollout/mean_ep_return`,
`rollout/mean_ep_len`, and `rollout/mean_ep_comm_tokens` fall back to partial
segment means so plots do not show misleading zeros.

`python -m syncorsink.train.recurrent_curriculum` runs staged recurrent
BC/DAgger curricula and can optionally fine-tune each stage with recurrent PPO.
For Pipeline Assembly, use `--scenario pipeline_assembly --oracle-type
planner_comm --agents 3` plus the per-stage Pipeline difficulty schedules:
`--pipeline-stage-count-schedule`, `--pipeline-required-per-stage-*-schedule`,
`--pipeline-sync-probability-schedule`, and
`--pipeline-dependency-probability-schedule`. The runner writes
`summary.json`, one checkpoint per stage, and a `_best.pt` RL checkpoint when
`--rl-updates > 0` and `--rl-save-best` is enabled. It also forwards the
Pipeline observation default `--obs-pipeline-features`, plus the generic
rare-action BC controls `--bc-action-class-balance` and
`--bc-event-action-weight`; the default event list includes `picked_resource`,
`dropped_resource`, `delivered`, `stage_completed`, and `sync_complete` for
Pipeline learning.
Use the Pipeline-specific action losses above when audits show persistent
missed pickups, missed deliveries, excessive drops, or wrong-station
interactions.

Curriculum PPO can be gated per stage with `--rl-updates-schedule`, for example
`--rl-updates-schedule 0,0,60` to keep early stages BC/DAgger-only and reserve
PPO for the harder stage. The curriculum runner defaults
`--rl-eval-use-eval-seeds` on so PPO best-checkpoint selection follows each
stage's final eval seeds. PPO eval seeds are still offset by
`--rl-eval-seed-stage-stride` per stage by default so W&B plots do not reuse the
same eval seed base across curriculum stages; set the stride to `0` to keep the
old fixed-seed behavior when `--no-rl-eval-use-eval-seeds` is set.

`examples/train_eval_workbench.py` is the single-run MAPPO checkpoint
round-trip smoke. It exposes the MAPPO trainer's communication send-rate
curriculum knobs, observation-memory flags, backbone choice, and final eval
decoding modes; the saved `summary.json` records the train config and loaded
policy metadata used for evaluation.

## Benchmark/spec configuration

### `benchmark_run.py`

Flags:

- `--spec` (required): benchmark JSON file
- `--wandb`, `--wandb-project`, `--wandb-run`
- `--policy-entrypoint`, `--policy-checkpoint`, `--policy-kwargs` for external policies
- `--allow-centralized-external-policy` for local debugging only

For Comm-MAT in spec cases:

- `policy: "comm_mat"`
- optional `policy_checkpoint`
- optional `comm_mat_deterministic`
- optional `comm_mat_send_threshold`

MARL benchmark/spec runners fail fast on unknown policy names. Supported non-LLM policies include `random`, `scripted`, `oracle`, `oracle_strong`, `oracle_planner`, `oracle_comm`, the `pipeline_planner_*` communication planners, `energy_planner_comm`, `signal_hunt_planner_comm`, and `comm_mat`.

### `eval_from_spec.py`

Flag:

- `--spec` (required)

Same spec keys as above are supported for Comm-MAT selection. Specs may also set `map_size`, `agents` or `num_agents`, `fov_preset`, `max_steps`, `comm_mode`, `track`, `energy_preset`, and `energy_private_monitor`.

For `energy_grid`, `energy_private_monitor` defaults to `true`. Set it to
`false` only for the legacy symmetric-information ablation.

## Recommended command patterns

LLM eval with full trace + W&B table + artifact + video:

```bash
python examples/eval_llm.py \
  --scenario signal_hunt \
  --provider openai-chat \
  --mode text \
  --planner executor \
  --model gpt-4o-mini \
  --episodes 5 \
  --trace-jsonl /tmp/syncorsink_llm_trace.jsonl \
  --trace-local-obs \
  --record-video \
  --video-episodes 2 \
  --render-split-view \
  --wandb \
  --wandb-log-trace-table \
  --wandb-log-trace-artifact \
  --wandb-log-video
```

Comm-MAT benchmark preset run:

```bash
python examples/benchmark_run.py --spec benchmarks/transformer_presets.json --wandb
```

Communication necessity sweep:

```bash
python examples/communication_ablation_sweep.py \
  --episodes 8 \
  --map-sizes 8 16 \
  --output-json logs/communication_ablation_sweep/latest.json \
  --wandb \
  --wandb-mode offline
```

Core training sweep:

```bash
python examples/core_training_sweep.py \
  --algorithms mappo comm_mat tarmac \
  --scenarios signal_hunt energy_grid pipeline_assembly \
  --updates 3 \
  --rollout-steps 64 \
  --epochs 2 \
  --minibatch 32 \
  --eval-every 3 \
  --eval-episodes 2 \
  --seeds 0 1 2 \
  --wandb \
  --wandb-mode offline
```

To run a fixed official benchmark case through the same sweep/W&B workbench,
use `--benchmark-spec` and `--benchmark-cases`. This bypasses the built-in
8x8 scenario defaults and uses the manifest's map size, agent count, FOV preset,
and horizon:

```bash
python examples/core_training_sweep.py \
  --algorithms recurrent_bc_rl \
  --benchmark-spec benchmarks/syncorsink_v0_1.json \
  --benchmark-cases signal_hunt_16x16_scaled_search \
  --seeds 0 \
  --recurrent-demo-episodes 30 \
  --recurrent-bc-epochs 4 \
  --recurrent-dagger-rounds 2 \
  --recurrent-dagger-episodes 30 \
  --recurrent-rl-updates 0 \
  --recurrent-eval-episodes 100 \
  --wandb \
  --wandb-mode online
```

Trajectory audit summaries include each policy's actual evaluated `env_config`
and a quick `config.policy_envs` index. This matters for recurrent checkpoints:
the audit wrapper may inherit checkpoint-required observation settings such as
agent count or FOV while preserving requested compatible fields such as map size
and max steps.

The transformer preset expects local checkpoint artifacts:

- `checkpoints/comm_mat_pipeline.pt`
- `checkpoints/comm_mat_energy.pt`
- `checkpoints/comm_mat_signal.pt`

These checkpoint files are not tracked in the repository. Train or restore them before running `benchmarks/transformer_presets.json`.

End-to-end checkpoint smoke test:

```bash
mkdir -p checkpoints

python examples/comm_mat_train.py \
  --scenario pipeline_assembly \
  --map-size 8 \
  --agents 3 \
  --fov-preset easy \
  --updates 1 \
  --rollout-steps 32 \
  --epochs 1 \
  --minibatch 32 \
  --device cpu \
  --eval-every 0 \
  --save checkpoints/comm_mat_pipeline.pt

python examples/comm_mat_train.py \
  --scenario energy_grid \
  --map-size 8 \
  --agents 3 \
  --fov-preset easy \
  --updates 1 \
  --rollout-steps 32 \
  --epochs 1 \
  --minibatch 32 \
  --device cpu \
  --eval-every 0 \
  --save checkpoints/comm_mat_energy.pt

python examples/comm_mat_train.py \
  --scenario signal_hunt \
  --map-size 8 \
  --agents 3 \
  --fov-preset easy \
  --updates 1 \
  --rollout-steps 32 \
  --epochs 1 \
  --minibatch 32 \
  --device cpu \
  --eval-every 0 \
  --save checkpoints/comm_mat_signal.pt

python examples/benchmark_run.py --spec benchmarks/transformer_presets.json
```

This verifies train-save-load-eval plumbing only. One-update checkpoints are not meaningful baselines.

Fresh-checkout smoke checks that do not require checkpoints:

```bash
pytest tests
python examples/benchmark_run.py --spec benchmarks/pipeline_presets.json
```

Locally verified on July 2, 2026:

```text
pytest tests
18 passed, 2 warnings

python examples/benchmark_run.py --spec benchmarks/pipeline_presets.json
case pipeline_easy_expert_comm success 1.0 return 37.6
case pipeline_hard_coord success 0.0 return -0.23399999999999999
case energy_easy_expert_comm success 0.6 return -1.6800000000000002
case signal_hunt_expert_comm success 1.0 return 29.910000000000004
```

## Practical note on diagrams/charts

W&B automatically builds line charts from logged scalar series (losses, returns, success rate, comm metrics).  
Trace tables and artifacts provide step-level data for building custom diagrams (message timelines, task transitions, prompt-response flow) outside or inside W&B dashboards.
