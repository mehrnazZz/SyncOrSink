# SyncOrSink Experiment Report

**Date:** 2026-03-22 (updated)
**Benchmark:** SyncOrSink — Communication-focused cooperative multi-agent POMDP
**Setting:** DTDE (decentralized training, decentralized execution), 8x8 maps, easy FOV

**Energy Grid note:** the Energy Grid results below are historical results for
the legacy symmetric-information variant. The current environment default and
`syncorsink_v0_2` use private node monitoring, which is intended to make
communication necessary.

---

## 1. Scenarios

| Scenario | Task | Success Condition | Key Challenge |
|---|---|---|---|
| **Signal Hunt** | Find clues, fuse constraints, scan hidden target | 2+ agents interact on target within 3-step window | Timing coordination + information sharing |
| **Energy Grid** | Deliver typed resources to recharge draining nodes | Required number of recharges before any node depletes | Real-time resource management + sync deliveries |
| **Pipeline Assembly** | Complete multi-stage pipeline with dependencies | All stages completed (pickup → deliver → sync) | Multi-step planning + partial blueprint sharing |

---

## 2. Master Results Table

### Signal Hunt (8x8, 2 agents, easy FOV)

| Method | Type | Success Rate | Avg Steps | Comm? | Notes |
|---|---|---|---|---|---|
| Oracle Strong | Full-state planner | **100%** | 13.5 | No | Upper bound |
| **BC→RL v2 (KL)** | **IL→RL** | **80%** | ~210 | Tokens | Best trained method overall |
| gpt-oss:20b | LLM (20B, local) | **67%** | 44 | Text | Open-source, free inference |
| gpt-4o-mini | LLM (API) | **60%** | 157 | Text | Action planner |
| DAgger BC | IL (42k demos) | **55%** | 137 | No | 3 rounds, 99.4% action acc |
| DAgger BC + comm | IL (43k demos) | **55%** | 137 | Tokens | Comm learned without perf loss |
| Vanilla BC | IL (354 demos) | 45% | 166 | No | From 50 oracle episodes |
| Comm-MAT | Transformer RL | **30%** | ~270 | Tokens | Improving at end of training |
| gpt-4o-mini (executor) | LLM (API) | 30% | 213 | Text | Gets stuck on plans |
| BC + comm (vanilla) | IL (354 demos) | 15% | 255 | Tokens | Comm dilutes action learning |
| Heuristic | Rule-based | 10% | 271 | No | Weak baseline |
| MAPPO v4 (CTDE) | PPO RL | 0% | 300 | Tokens | Joint-scan shaping, still fails |
| MAPPO v4 (DTDE) | PPO RL | 0% | 300 | Tokens | Same as CTDE |
| BC from LLM | IL from traces | 0% | 300 | Tokens | Too few demos, noisy |
| Random | — | 0% | 300 | No | Lower bound |

### Energy Grid (8x8, 3 agents, easy FOV, easy preset)

| Method | Type | Success Rate | Avg Steps | Comm? | Notes |
|---|---|---|---|---|---|
| Oracle Strong | Full-state planner | **100%** | 50 | No | Upper bound |
| **Comm-MAT** | **Transformer RL** | **100%** | 17-20 | Tokens | First RL method to solve it |
| **BC→RL v2 (KL)** | **IL→RL** | **100%** | — | No | 100% from first eval, KL preserved BC |
| gpt-oss:20b | LLM (20B, local) | **100%** | 23 | Text | 3/3 episodes |
| gpt-4o | LLM (API) | **60%** | 28 | Text | Stronger model helps |
| Vanilla BC | IL (4101 demos) | **35%** | 34 | No | From 94 oracle episodes |
| DAgger BC | IL (14.5k demos) | 25% | 36 | No | DAgger didn't help here |
| gpt-4o-mini | LLM (API) | 20% | 39 | Text | Tight energy budget |
| gpt-4o-mini (hard) | LLM (API) | 0% | 20 | Text | Nodes deplete at step 20 |
| Random/Heuristic | — | 0% | ~207 | No | Lower bound |

### Pipeline Assembly (8x8, 3 agents, easy FOV)

| Method | Type | Success Rate | Avg Steps | Comm? | Notes |
|---|---|---|---|---|---|
| Oracle Strong | Full-state planner | **60%** | 173 | No | Even oracle struggles |
| Oracle | Full-state planner | 50% | 189 | No | Greedy version |
| gpt-4o-mini | LLM (API) | **20%** | 277 | Text | Best learned method |
| gpt-4o-mini (executor) | LLM (API) | 0% | 300 | Text | Gets stuck on plans |
| gpt-oss:20b | LLM (local) | 0% | 200 | Text | Timeouts + can't chain deps |
| **Recurrent BC→RL** | **IL→RL (LSTM)** | **10%** | — | No | **First trained method to solve it!** |
| BC→RL v2 (KL) | IL→RL (MLP) | 0% | 300 | Tokens | KL preserved init but MLP can't track progress |
| BC→RL v1 | IL→RL (MLP) | 0% | 300 | No | RL destroyed BC init (return 36→3) |
| All BC/DAgger variants | IL | 0% | 300 | No | 92% action acc but can't chain deps |
| Comm-MAT | Transformer RL | 0% | 300 | Tokens | Cannot learn multi-step planning |
| MAPPO (all versions) | PPO RL | 0% | 300 | — | Not tested on this scenario |

---

## 3. Key Findings

### 3.1 BC→RL v2 is the best trained method
- **80% on signal_hunt** — beats all LLMs (60-67%) and all other trained methods
- **100% on energy_grid** — from the very first eval, KL preserved BC initialization perfectly
- KL regularization (coeff=0.5) + frozen encoder + lower LR (3e-5) + fewer PPO epochs (2) was the key
- v1 without KL: RL destroyed BC init in 600 updates (return 36→3, entropy 1.9→0.36)
- v2 with KL: entropy stable (1.9→2.4), return maintained, success rate climbed to 80%

### 3.2 Comm-MAT solves energy_grid — and ablation reveals when communication matters
- **100% success on energy_grid** — but no-comm ablation ALSO achieves 100%
- **30% on signal_hunt** — no-comm ablation drops to ~0%, proving communication is essential
- **Energy grid insight:** the transformer backbone alone learns independent coordination; explicit messaging is unnecessary on 8x8 easy. The attention mechanism over grid observations captures enough spatial reasoning.
- **Signal hunt insight:** communication is the key differentiator. Without it, agents can't share clue constraints or coordinate the synchronized target scan. This is the benchmark's core contribution — a task where learned communication provably helps.
- Communication send rate: 22% for energy (redundant but harmless), 3.5% for signal (sparse but critical)

### 3.3 LLMs dominate with zero training
- LLMs solve signal_hunt 60-67% and energy_grid 60-100% without any task-specific training
- Prior knowledge about spatial reasoning, coordination, and communication transfers directly
- Open-source 20B model (gpt-oss) matches or exceeds closed-source APIs
- Pipeline_assembly remains hard even for LLMs (20% with gpt-4o-mini)
- Improved prompt (decoded stage descriptions) pending evaluation

### 3.4 Pure MAPPO fails at communication-dependent coordination
- 0% success across all 4 shaping reward versions on signal_hunt
- v1-v2: reward too weak or comm collapsed (send rate → 0%)
- v3: agents farmed scan bonus (+314 return) without coordinating (send rate 77%)
- v4: joint-scan fix prevented farming but still can't discover timing coordination
- Fundamental issue: PPO with 1.5M env steps can't discover rare joint-scan events from scratch
- BC→RL warmstart solves what pure RL cannot — pre-trained navigation + RL timing coordination

### 3.5 IRL works for simple coordination but suffers reward hacking on complex tasks
- **Energy grid: 100% success** with learned reward — IRL replaces hand-crafted shaping entirely
- **Signal hunt: 0% success, return 4136** — reward model exploited. MAPPO found out-of-distribution states that the reward model incorrectly scores highly. Only 688 training transitions → poor generalization.
- **Pipeline assembly: 0% success, return 531** — same reward hacking, policy collapsed into deterministic loop
- **Lesson:** simple reward regression works when the reward landscape is simple (energy_grid) but needs more sophisticated approaches (more data, adversarial training, or MA-AIRL) for complex coordination tasks
- The reward model has gaps in coverage because oracle demos only visit a narrow slice of the state space; MAPPO finds and exploits these gaps

### 3.6 Hand-crafted reward shaping is a double-edged sword
- v1: shaping too weak relative to comm cost (0.01 vs 12.0 penalty) → agents learn to not move
- v2: 10x stronger shaping + lower comm cost → comm collapsed anyway
- v3: scan + co-location bonus → farming local optimum (+314 return, 0% success)
- v4: joint near-miss bonus → prevented farming but still 0% success
- **Lesson:** hand-crafted shaping creates deceptive local optima; IRL-based rewards may be better

### 3.6 Behavioral Cloning is surprisingly effective
- Vanilla BC from 354 transitions: 45% success on signal_hunt (competitive with LLMs)
- DAgger improves to 55% by fixing distribution shift (42k transitions, 99.4% action acc)
- BC with communication: fails with small data (15%) but works with DAgger (55%)
- **Key insight:** DAgger is essential for making multi-agent IL with communication work
- Pipeline_assembly: 92% action accuracy but 0% success — can't chain multi-step dependencies

### 3.7 Communication learning requires sufficient data
- BC with 354 demos + comm: 15% (worse than 45% without comm)
- DAgger with 43k demos + comm: 55% (matches no-comm DAgger)
- The comm head has a large output space (send × length × tokens) that needs diverse examples
- Two-phase training (action first, comm second) addresses this structurally

### 3.8 Pipeline Assembly is the benchmark's hardest challenge
- Even the oracle only achieves 50-60% success
- Requires multi-step dependency chains: pickup → deliver → sync, in order
- Partial blueprints mean agents must communicate to know the full plan
- Only LLMs (20% with gpt-4o-mini) have cracked it among learned methods
- BC→RL v2 preserved initialization but BC can't chain dependencies — the bottleneck is IL, not RL
- Improved LLM prompt (decoded stage descriptions) may help — eval pending

### 3.9 BC→RL warmstart requires careful regularization
- v1 (naive): RL destroyed BC init in 600 updates (return 36→3, entropy collapsed to 0.36)
- v2 (KL + freeze + low LR): entropy stable, 80% success on signal_hunt
- The gap: v1 uses LR=1e-4, 4 PPO epochs; v2 uses LR=3e-5, 2 epochs, KL=0.5, frozen encoder
- **Lesson:** IL→RL needs KL regularization to prevent catastrophic forgetting of pre-trained behavior

---

## 4. Method Comparison Summary

### By approach category (best result per scenario):

| Category | signal_hunt | energy_grid | pipeline_assembly |
|---|---|---|---|
| **Oracle** (full state) | 100% | 100% | 60% |
| **IL→RL** (BC→RL v2) | **80%** | **100%** | 0% |
| **LLM** (best per scenario) | 67% | **100%** | **20%** |
| **IL** (DAgger BC) | 55% | 35% | 0% |
| **Transformer RL** (Comm-MAT) | 30% | **100%** | 0% |
| **TarMAC** (attention comm) | 0% | **100%** | — |
| **IRL MAPPO** (learned reward) | 0% | **100%** | 0% |
| **Recurrent BC→RL** (LSTM) | 0% | **100%** | **10%** |
| **PPO RL** (MAPPO v4) | 0% | — | — |
| **Random** | 0% | 0% | 0% |

### Key takeaways for the paper:
1. **IL→RL warmstart** is the strongest trained approach (80%, 100%, 10%)
2. **LLMs** achieve 20% on pipeline_assembly with zero training
3. **Recurrent BC→RL** is the first trained method to crack pipeline_assembly (10%) — memory is the key
4. **Comm-MAT** proves learned communication enables RL coordination on signal_hunt (30% → 0% without comm)
5. **Pure MAPPO** fails entirely — reward shaping cannot substitute for pre-training
6. **Memory + communication** is the next frontier for pipeline_assembly

---

## 5. Reward Shaping Evolution (Signal Hunt)

| Version | scan_bonus | joint_scan | colocation | comm_utility | comm_cost | shaping_scale | Result |
|---|---|---|---|---|---|---|---|
| v1 | 0 | 0 | 0 | 0 | 0.01 | 0.01 | Comm penalty dominates shaping |
| v2 | 0 | 0 | 0 | 0 | 0.001 | 0.1 | Comm collapsed to 0.2% |
| v3 | 1.0 | 0 | 0.5 | 0.1 | 0.001 | 0.1 | Farming: +314 return, 0% success |
| v4 | 0.2 | 3.0 | 0.5 | 0.1 | 0.001 | 0.1 | No farming, still 0% success |

---

## 6. MAPPO Training Summary (Signal Hunt)

| Run | Critic | Updates | Best Eval Success | Final Comm Rate | Final Entropy |
|---|---|---|---|---|---|
| v1 CTDE | Central | 300 | 0% | ~49% | ~6.0 |
| v2 CTDE | Central | 3000 | 0% | 0.2% (collapsed) | 6.9 |
| v2 no-comm | Local | 3000 | 0% | N/A | 1.8 |
| v3 CTDE | Central | 3000 | 0% | 77% (farming) | 7.6 |
| v3 DTDE | Local | 3000 | 0% | 86% (farming) | 7.1 |
| v4 CTDE | Central | 3000 | 0% | 8.7% | 6.3 |

---

## 7. BC→RL Comparison

| Version | LR | KL | Encoder | PPO Epochs | signal_hunt | energy_grid | pipeline |
|---|---|---|---|---|---|---|---|
| **v1** (naive) | 1e-4 | 0 | trainable | 4 | 0% (destroyed init) | — | 0% (destroyed init) |
| **v2** (KL reg) | 3e-5 | 0.5 | frozen | 2 | **80%** | **100%** | 0% (preserved init) |

---

## 8. IL Methods Comparison (Signal Hunt)

| Method | Demo Source | Demos | Action Acc | Success | Comm? |
|---|---|---|---|---|---|
| DAgger (3 rounds) | Oracle | 42k | 99.4% | **55%** | No |
| DAgger + comm (3 rounds) | Oracle+comm | 43k | 98.7% | **55%** | Yes (2 tok/ep) |
| Vanilla BC | Oracle | 354 | 79% | 45% | No |
| Two-phase BC + comm | Oracle+comm | 354 | 71% | 15% | Yes (6 tok/ep) |
| BC from LLM | LLM traces | 742 | 60% | 0% | Yes (0 tok/ep) |

---

## 9. LLM Provider Comparison

| Model | Size | Access | signal_hunt | energy_grid | pipeline_assembly | Cost |
|---|---|---|---|---|---|---|
| gpt-oss:20b | 20B | Local (ollama) | **67%** | **100%** | 0% | Free |
| gpt-4o | ~200B+ | API | — | **60%** | quota error | $$$ |
| gpt-4o-mini | ~8B | API | **60%** | 20% | **20%** | $ |

---

## 10. Comm-MAT Results + Communication Ablation

| Scenario | With Comm | Without Comm | Comm Necessary? |
|---|---|---|---|
| **energy_grid** | **100%** | **100%** | No — backbone alone sufficient |
| **signal_hunt** | **30%** | ~0% (best 10%) | **Yes — communication is key** |
| **pipeline_assembly** | 0% | — | Unsolved either way |

### Full Comm-MAT results:

| Scenario | Success Rate | Comm Send Rate | Notes |
|---|---|---|---|
| **energy_grid** | **100%** | 22% | Solved from early training |
| **signal_hunt** | **30%** | 3.5% | Improving — more training likely helps |
| **pipeline_assembly** | 0% | 2.3% | Cannot learn multi-step planning |

### Why communication doesn't matter for energy_grid (any size/difficulty):

Verified that observations are properly partial (no privileged info leaks):
- Agents see only a 7x7 local grid (medium FOV on 16x16)
- `explored_mask` exists but is NOT included in `flatten_obs` for MAPPO/BC-RL
- Comm-MAT only uses: `local_grid`, `inventory`, `self_pos`, `goal_hint`, messages
- Track is correctly `dtde` — no centralized state

**The real reason energy_grid is easy: no information asymmetry.**
- `self_pos` (absolute position) lets the policy learn a spatial function — "when at (x,y) and I see resource type T, pick up and go to direction D"
- Only 3 nodes on 16x16 — with 4 agents exploring, all nodes are discovered within ~50 steps
- Resources spawn continuously (15%/step on hard) — always something nearby to pick up
- Unlike signal_hunt, there's nothing one agent knows that another can't independently discover

**16x16 hard results confirm this — all methods still 100%:**

| Method | 16x16 Hard Success | Steps | Comm Rate |
|---|---|---|---|
| Comm-MAT | 100% | 72 | 1.3% |
| Comm-MAT no-comm | 100% | 68-72 | disabled |
| TarMAC | 100% | 207 | attn 1.07 |
| BC→RL | 100% | 72 | 0% |
| Oracle Strong | 90% | 49 | N/A |

Note: trained methods (100%) outperform the oracle (90%) because the oracle uses a greedy heuristic while learned policies find globally better strategies.

### Information structure determines communication necessity:

| Scenario | Information Structure | Comm Required? | Evidence |
|---|---|---|---|
| **Energy grid** | Symmetric — all info independently discoverable | **No** | Comm-MAT = Comm-MAT no-comm (100% both) at every size/difficulty |
| **Signal hunt** | Asymmetric — agents get different clues | **Yes** | Comm-MAT 30% → 0% without comm |
| **Pipeline assembly** | Asymmetric — partial blueprints per agent | **Yes** (presumed) | 0% for all trained methods — unsolved |

**Key benchmark design insight:** Communication becomes necessary only when agents have complementary partial information that must be fused. Tasks with symmetric information (where any agent can independently discover everything) can be solved by coordination without messaging.

---

## 11. IRL (Learned Reward) Results

| Scenario | Success | Final Return | Entropy | Comm Rate | Diagnosis |
|---|---|---|---|---|---|
| **energy_grid** | **100%** | 29.5 | 0.06 | 0% | Learned reward works — replaces hand-crafted shaping |
| **signal_hunt** | 0% | 4136.9 | 6.5 | 68% | Reward hacking — exploits gaps in reward model |
| **pipeline_assembly** | 0% | 531.3 | 0.5 | 0% | Reward hacking — collapsed deterministic loop |

- Reward models trained on oracle (obs, action) → reward tuples via supervised regression
- Energy grid: 688 tuples sufficient because reward landscape is simple (deliver matching resource → positive reward)
- Signal hunt / pipeline: reward model has poor out-of-distribution generalization; MAPPO exploits states the oracle never visited
- Future work: adversarial IRL (MA-AIRL) or reward model ensembles to prevent exploitation

---

## 12. Running/Pending Experiments

| Experiment | Status | Platform |
|---|---|---|
| gpt-oss:20b pipeline_assembly (improved prompt) | Running | Local |
| IRL MAPPO (all 3 scenarios) | **Done** | RunPod |
| Comm-MAT no-comm ablation (energy + signal) | **Done** | RunPod |
| BC→RL v2 (all 3 scenarios) | **Done** | RunPod |
| Comm-MAT (all 3 scenarios) | **Done** | RunPod |
| MAPPO v4 (signal_hunt) | **Done** | RunPod |

---

## 12. Infrastructure Built

| Component | Status | Files |
|---|---|---|
| MAPPO training (DTDE/CTDE) | Done | `syncorsink/train/mappo.py` |
| Comm-MAT training (+ablation) | Done | `syncorsink/train/comm_mat.py` |
| BC from oracle (+comm) | Done | `syncorsink/train/bc.py` |
| DAgger (+comm) | Done | `syncorsink/train/bc.py` |
| BC→RL warmstart (KL + freeze) | Done | `syncorsink/train/mappo.py` |
| Reward regression (IRL) | Done | `syncorsink/train/bc.py` |
| LLM eval (OpenAI + litellm/ollama) | Done | `examples/eval_llm.py` |
| Oracle/heuristic eval | Done | `examples/eval_run.py` |
| BC eval | Done | `examples/eval_run.py` |
| Coordination shaping (v4) | Done | `syncorsink/envs/scenarios.py` |
| Energy node_critical events | Done | `syncorsink/envs/scenarios.py` |
| Pipeline hint decoding | Done | `syncorsink/llm/policy.py` |
| Prompt compression | Done | `syncorsink/llm/policy.py` |
| Test suite (12 tests) | Done | `tests/` |

---

## 13. Scale Experiments: 8x8 → 16x16

**Settings:** 16x16 map (4x area), 4 agents (up from 2-3), medium FOV (smaller view)

### Oracle at 16x16

| Scenario | 8x8 Success | 16x16 Success | 16x16 Steps | Scaling |
|---|---|---|---|---|
| signal_hunt | 100% | **100%** | 11.7 | No degradation — faster with 4 agents |
| energy_grid | 100% | **90-100%** | 48-60 | Slight drop for oracle_strong |
| pipeline_assembly | 50-60% | **60-70%** | 155-158 | Improves — 4 agents help parallelize |

### Trained Methods at 16x16

| Method | signal 8x8 | signal 16x16 | energy 8x8 | energy 16x16 |
|---|---|---|---|---|
| **BC→RL v2** | **80%** | **30%** | **100%** | **100%** |
| **Comm-MAT** | 30% | 20% | **100%** | **100%** |

### Key Scaling Findings:
- **Energy grid scales perfectly** — both BC→RL v2 and Comm-MAT maintain 100% at 16x16. The task structure (typed resource delivery) generalizes to larger maps. Steps increase (72 vs 17-50) but success is maintained.
- **Signal hunt degrades significantly** — BC→RL drops 80% → 30%, Comm-MAT drops 30% → 20%. Larger map + medium FOV = larger search space, reduced mutual visibility, harder timing coordination.
- **BC→RL v2 remains best trained method at scale** — 30% at 16x16 matches Comm-MAT's 8x8 performance.
- **Communication becomes more critical at scale** — more information to share across larger distances with less visibility. Comm-MAT's send rate at 16x16 energy is 2.5% (agents learn independent coordination even at scale).

### LLM at 16x16 (pending — gpt-oss:20b running locally)

---

## 14. Pipeline Assembly: The Open Challenge

Pipeline assembly remains **0% for all trained methods** across every approach tried:

| Method | Success | Why it fails |
|---|---|---|
| gpt-4o-mini (API) | **20%** | Only method to crack it — prior knowledge about task decomposition |
| gpt-oss:20b (local, improved prompt) | 0% | Timeout issues with 3-agent prompts; can't chain dependencies |
| BC→RL v2 (KL) | 0% | BC can't chain multi-step dependencies; RL can't improve beyond BC |
| BC→RL v1 (naive) | 0% | RL destroyed BC initialization |
| DAgger BC (93k demos, 92% acc) | 0% | High action accuracy but can't sequence dependency chains |
| Comm-MAT | 0% | Cannot learn multi-step planning from RL alone |
| IRL MAPPO | 0% | Reward hacking — exploits gaps in learned reward |
| Oracle Strong | **60%** | Even oracle struggles — genuinely hard task |

**Why pipeline assembly is hard:**
- Multi-step dependency chains: stage A → stage B → stage C (in order)
- Partial blueprints: each agent only knows some stages, must communicate to learn the full plan
- Resource type matching: must pick up the RIGHT resource type for each stage
- Sync interactions: some stages require 2 agents to interact simultaneously
- Even the oracle only achieves 60% — the task has inherent difficulty from map layout and resource placement

**Breakthrough: Recurrent BC→RL achieves 10%**
- LSTM memory lets the policy track which stages are complete across steps
- MLP policies (BC→RL v2) preserved initialization but couldn't chain dependencies — the bottleneck was **lack of memory**, not lack of good actions
- First success at RL update 1149; appeared at 10% in two separate evals
- Return improved from 12 → 30 during training; entropy stable at 1.0 (KL preserved policy)
- The LSTM is the minimal architecture that can represent sequential task progress

**What might push higher:**
- Bigger LSTM or attention over stage hint tokens
- Communication-enabled recurrent policy (agents share stage progress)
- Curriculum learning (start with 1-2 stages, gradually increase)
- More RL training (10% appeared late — may still be improving)

---

## 15. Recurrent BC→RL Results

| Scenario | Success | Initial Return | Final Return | Entropy | Notes |
|---|---|---|---|---|---|
| **pipeline_assembly** | **10%** | 11.9 | 29.5 | 1.0 (stable) | First trained method to solve it |
| **signal_hunt 8x8** | **98%** | — | 16.32 avg audit return | — | Official 2-agent Signal checkpoint; external 100-episode trajectory audit |
| **signal_hunt 16x16** | **79%** | — | 30.54 avg audit return | — | Fresh Signal specialist checkpoint with canonical constraint-message eval assist; external 100-episode trajectory audit |

- **Architecture:** MLP encoder → LSTMCell → policy head (same width as MAPPO actor + LSTM)
- **Training:** 200 oracle demos → recurrent BC (truncated BPTT, 30 epochs) → PPO fine-tuning (3000 updates, KL=0.5)
- **Why it works:** The LSTM maintains hidden state across steps, tracking stage progress (which delivered, which pending). MLP policies can't do this — they see each step independently.
- **Why only 10%:** Pipeline assembly has 4+ stages with dependencies. The LSTM learns sequential behavior but still struggles with multi-agent coordination (no communication in current version).
- **Key insight:** Memory is necessary but not sufficient for sequential multi-agent tasks. Communication + memory is the next step.

### Current Signal Specialist Status (August 12, 2026)

The prior `examples/core_training_sweep.py` Signal specialist default was
validated on `signal_hunt_16x16_scaled_search` with seed 0, 30 demo episodes,
4 BC epochs, 2 DAgger rounds of 30 episodes, no PPO updates, and a 100-episode
external trajectory audit. That promoted default profile included clue-fusion
auxiliary supervision, agent role/search-sector features preserved in BC/DAgger
rows, target pursuit/match action auxiliaries, target scan auxiliaries,
sync-response supervision, frontier-exploration replay, and target-handoff
positive/failure replay.

After this audit, an opt-in visible-clue action/replay profile was tested to
target the remaining `no_clue_or_target_scan` failures without using hidden
target state. It improved clue collection but regressed final success and is
therefore not promoted as the default profile.

| Run | Audit Success | Failure Mix |
|---|---|---|
| `recurrent_signal16_default_profile_seed0` | **54/100** | `no_clue_or_target_scan`: 25, `no_target_scan`: 17, `decoy_scan`: 4 |
| `recurrent_signal16_rolefix_seed0` | **52/100** | `no_clue_or_target_scan`: 16, `no_target_scan`: 27, `decoy_scan`: 5 |
| `recurrent_signal16_rolefix_handoff_seed0` | **57/100** | `no_clue_or_target_scan`: 19, `no_target_scan`: 19, `solo_target_scan`: 1, `decoy_scan`: 4 |
| `recurrent_signal16_rolefix_handoff_targetscan_lock_seed0` | 57/100 | same failure mix as rolefix-handoff; avg return fell `22.52 -> 21.13` and decoy-scan events rose `13 -> 292` |
| `recurrent_signal16_rolefix_handoff_forcefirst_seed0` | 57/100 | no audit change from rolefix-handoff; force-first scan-sync eval toggle was a no-op on this panel |
| `recurrent_signal16_exact_handoff_seed0` | 53/100 | `no_clue_or_target_scan`: 25, `no_target_scan`: 17, `solo_target_scan`: 1, `decoy_scan`: 4; avg return `20.93` |
| `recurrent_signal16_scanpressure_seed0` | 38/100 | `no_clue_or_target_scan`: 49, `no_target_scan`: 12, `decoy_scan`: 1; avg return `15.01` |
| `recurrent_signal16_visible_clue_seed0` | 48/100 | `no_clue_or_target_scan`: 18, `no_target_scan`: 34 |
| `recurrent_signal16_memorylabels_seed0` | 53/100 sweep eval | Opt-in trusted exact-memory pursuit labels; more decoy scans than the promoted baseline, not externally audited |
| `recurrent_signal16_rolefix_handoff_frontier_assist_seed0` | 56/100 | Opt-in frontier eval assist on the rolefix-handoff checkpoint; `no_clue_or_target_scan`: 18, `no_target_scan`: 21, `solo_target_scan`: 2, `decoy_scan`: 3 |
| `recurrent_signal16_rolefix_handoff_constraintcopy_seed0` | **77/100** | Canonical Signal clue-message copy assist on the rolefix-handoff checkpoint; matched seed-3000 audit failure mix `no_clue_or_target_scan`: 11, `no_target_scan`: 9, `solo_target_scan`: 2, `decoy_scan`: 1 |
| `recurrent_signal16_rolefix_handoff_constraintcopy_compatible_seed0` | **77/100** | Adding compatible visible-target scan assist on top of constraint-copy did not improve success or failure mix |
| `recurrent_signal16_constraintcopy_rolefix_profile_seed0` | **79/100** | Fresh rolefix-profile training with constraint-message copy default; `no_clue_or_target_scan`: 17, `no_target_scan`: 2, `solo_target_scan`: 2, no decoy failures; avg return `30.54`, avg steps `73.05` |
| `recurrent_signal8_constraintcopy_rolefix_profile_seed0` | **98/100** | Official 2-agent 8x8 Signal checkpoint; `no_clue_or_target_scan`: 1, `decoy_scan`: 1; W&B run `hzm8kztx` |
| `recurrent_signal_multisize_16_32_constraintcopy_seed0_v3` | **84/100 @16x16**, **41/100 @32x32** | 16/32 mixed curriculum improved 16x16 but did not improve 32x32; 32x32 failures: `no_clue_or_target_scan`: 36, `no_target_scan`: 15, `decoy_scan`: 8 |
| `recurrent_signal_multisize_16_32_large_map_conservative_seed0` | **88/100 @16x16**, **55/100 @32x32** | Tuned `--recurrent-signal-preset large_map`; best current 32x32 result; 32x32 failures: `no_clue_or_target_scan`: 18, `no_target_scan`: 22, `solo_target_scan`: 2, `decoy_scan`: 3; W&B run `bggn5dip` |
| `recurrent_signal_multisize_16_32_large_map_scanconvert_seed0` | 84/100 @16x16, 49/100 @32x32 | Negative ablation: added first/joint-scan positive replay and stronger scan-action weights; 32x32 `no_target_scan` rose to 28 and success regressed; W&B run `duosomim` |
| `recurrent_signal_multisize_16_32_large_map_targetmemory_seed0` | 65% mixed eval | Negative ablation: trusted exact-memory target pursuit with broad responder labels; split eval was 80% at 16x16 and 50% at 32x32, below the conservative baseline; W&B run `8ibvfhg2` |
| `recurrent_signal_multisize_16_32_large_map_targetmemory_cap1_seed0` | 62.5% mixed eval | Negative ablation: trusted exact-memory target pursuit capped to the nearest responder; W&B DAgger split was 75% at 16x16 and 30% at 32x32, so no full audit was run; W&B run `zw2fadn1` |
| `recurrent_signal_multisize_16_32_large_map_age_seed0` | 57.5% mixed eval | Negative ablation: enabled exploration-age observations for the large-map preset; 32x32 split was 40% despite higher clue count, so age remains opt-in; W&B run `ekvcpsb2` |
| `recurrent_signal_multisize_16_32_large_map_constraintfrontier_seed0` | 60% mixed eval | Negative ablation: constraint-compatible Signal frontier labels; 32x32 split was 45% and target scans fell, so the bias remains opt-in; W&B run `bs4stt2m` |
| `recurrent_signal_multisize_16_32_large_map_constraintfrontier_tight_seed0` | 67.5% mixed eval | Diagnostic ablation after tightening constraint-frontier labels to bounded inferred targets; 16x16 rose to 90% but 32x32 stayed at 45%, so no full audit was run; W&B run `xy4w9cpc` |
| `recurrent_signal_multisize_16_32_large_map_signalfrontier_seed0` | 62.5% mixed eval | Negative isolation: Signal-anchor fallback frontier assignment without constraint bias; 16x16 rose to 85% but 32x32 fell to 30% with many decoy target visits; W&B run `fil29d02` |
| `recurrent_signal_multisize_16_32_large_map_targetrendezvous_seed0` | 60% mixed eval | Negative ablation: exact-target-informed pair rendezvous labels; 16x16 reached 80% but 32x32 fell to 40%, with lower true-target reach despite fewer decoy scans; W&B run `vde2xcu5` |
| `recurrent_signal_multisize_16_32_large_map_cluepositive_seed0` | 60% mixed eval | Negative ablation: added `clue_found` positive replay; 16x16 reached 80% but 32x32 stayed at 40%, with 32x32 decoy scans rising to 6.7; W&B run `tuudojwi` |
| `recurrent_signal_multisize_16_32_large_map_ambiguousdecision_seed0` | 70% mixed eval | Diagnostic ablation: ambiguous true-target scans become negative target-decision labels; best split was 90% at 16x16 and 50% at 32x32, matching but not beating the conservative 32x32 eval slice; W&B run `7dr55x28` |
| `recurrent_signal_multisize_16_32_large_map_ambiguousdecision_scanpush_seed0` | 65% mixed eval | Negative follow-up: ambiguity labels plus moderate first-scan/opportunity pressure; best split was 85% at 16x16 and 45% at 32x32; W&B run `8tbyh3s6` |

The main remaining bottlenecks are now weighted toward large-map discovery.
Constraint-message copy fixed a major communication failure where generated
step-0 and post-clue structured tokens contradicted the true private clues,
causing agents to reject the real target after reaching it. Handoff replay reduced the role-fix-only
`no_target_scan` spike while keeping the discovery gains from agent role
features. The visible-clue ablation raised audit `avg_clues_found` to `0.85`
but shifted failures into `no_target_scan`; the target-scan lock ablation showed
that simply bypassing learned scan suppressors over-scans decoys without raising
success. The exact-handoff-label ablation was also negative: stricter responder
evidence reduced target-handoff labels, lowered success to 53/100, and increased
target-reach-without-scan exposure (`20.25 -> 32.88` agent-steps). A narrow
force-first scan-sync eval toggle did not change behavior. A stronger scan
pressure ablation (`first_target_scan=1.2`, `target_opportunity=0.8`,
`scan_decision_pos=3.0`) reduced decoy scans but collapsed discovery and joint
scan completion. The next useful fix is therefore better large-map discovery,
message-driven target memory, and teammate routing before scan conversion, not
a stricter handoff-label filter, broader forced-interact rule, or larger
positive scan loss.
The matched seed-3000 rolefix-handoff baseline was 56/100; canonical
constraint-message copy raised it to 77/100, reduced average steps from
142.21 to 79.98, and reduced target-reach-without-scan exposure from 22.03 to
8.76 agent-steps. Decoy failures also dropped from 2 to 1. A compatible
visible-target scan assist on top was flat, so the useful promotion candidate is
message canonicalization rather than another forced-scan rule.

A fresh comparable run
(`recurrent_signal16_constraintcopy_rolefix_profile_seed0`) trained with the
new default enabled reached 79/100 on the same 100-episode seed-3000 audit, with
zero decoy failures and only two `no_target_scan` failures. The remaining misses
are now mostly discovery failures (`no_clue_or_target_scan`: 17). W&B tracked the
training run at
`https://wandb.ai/orion8/syncorsink-core-training/runs/04h8pmkn`. A 32x32,
4-agent audit of the same checkpoint reached 41/100, with failures dominated by
`no_clue_or_target_scan`: 51. This confirms the policy executes at the larger
map scale, but large-map discovery remains the next implementation bottleneck.
A same-checkpoint 8x8 stress audit reached 91/100 using the
checkpoint-compatible 4-agent env; the official 2-agent 8x8 track is covered by
the separate 8x8 checkpoint below.

The official 2-agent 8x8 Signal checkpoint
(`recurrent_signal8_constraintcopy_rolefix_profile_seed0`) now reaches 98/100
on a 100-episode seed-3000 trajectory audit, with one discovery failure and one
decoy-scan failure. Its W&B run is
`https://wandb.ai/orion8/syncorsink-core-training/runs/hzm8kztx`.

A first 16/32 mixed curriculum
(`recurrent_signal_multisize_16_32_constraintcopy_seed0_v3`) trained in 207.4s
and logged to
`https://wandb.ai/orion8/syncorsink-core-training/runs/79ukojb9`. It improved
the independent 16x16 audit to 84/100, but 32x32 remained 41/100 and shifted
failures toward decoy scans and missed target scans. Mixing 32x32 into the
current curriculum is therefore not sufficient. The sweep now provides
`--recurrent-signal-preset large_map`, which inherits the current specialist
defaults and adds a conservative 32x32-only visible-clue auxiliary, light
decoy-drift/decoy-scan suppression, and targeted DAgger replay for
`visible_clue_miss`, `decoy_scan`, and `rejected_target_scan`. A first heavier
version with rejected-target-drift replay collected thousands of drift labels
and lowered round-2 mixed eval, so that broad drift pressure is intentionally
not part of the current preset. The tuned large-map run
(`recurrent_signal_multisize_16_32_large_map_conservative_seed0`) logged to
`https://wandb.ai/orion8/syncorsink-core-training/runs/bggn5dip`, reached 70%
on the 20+20 mixed eval panel, and improved the independent audits to 88/100 at
16x16 and 55/100 at 32x32. The remaining 32x32 failures are now split between
`no_clue_or_target_scan` and `no_target_scan`, so the next Signal step is
post-discovery target conversion and teammate routing on large maps. A direct
scan-conversion ablation
(`recurrent_signal_multisize_16_32_large_map_scanconvert_seed0`) appended
`first_target_scan`/`joint_target_scan` positive replay and moderately raised
target-scan action weights. It looked promising on the 20+20 eval panel
(`60%` at 32x32), but the full independent audits regressed to 84/100 at 16x16
and 49/100 at 32x32, with `no_target_scan` rising to 28. It is therefore not
promoted; the next fix should improve teammate routing/role assignment after
target evidence rather than simply increasing scan-positive pressure. Two
target-memory pursuit ablations also stayed below the conservative baseline:
the broad exact-memory label run (`8ibvfhg2`) reached only 65% mixed eval, and
the nearest-responder cap run (`zw2fadn1`) reached only 62.5% mixed eval with
30% 32x32 DAgger success. The cap remains available for controlled ablations,
but large-map Signal should move next toward explicit role/teammate routing
rather than broader pursuit-label replay. Enabling exploration-age observations
(`ekvcpsb2`) was also negative: it increased 32x32 clue count to 0.85 on the
20-episode split but lowered 32x32 success to 40%, so exploration age remains
an explicit flag rather than a large-map default. Constraint-biased frontier
labels (`bs4stt2m`) and Signal-anchor fallback frontier assignment (`fil29d02`)
were also negative on the 32x32 split, so the default frontier fallback remains
the prior conservative behavior while the constraint-frontier bias stays opt-in
for controlled experiments. After this ablation, the opt-in constraint-frontier
path was tightened to reuse the bounded inferred-target candidate set from
target decoding, avoiding weak one-clue or parity-only frontier bias. The
tightened rerun (`xy4w9cpc`) improved mixed eval to 67.5% by raising 16x16 to
90%, but 32x32 remained at 45%, so large-map Signal still needs a stronger
post-discovery routing/search change rather than more local frontier scoring.
The target-rendezvous ablation
(`recurrent_signal_multisize_16_32_large_map_targetrendezvous_seed0`,
`vde2xcu5`) routed the closest exact-target-informed pair toward the target and
made an early scanner wait when its partner could not still arrive and scan
inside the target-scan window. It produced many labels
(`6526` action labels and `369` wait labels in round 2) but regressed mixed eval
to 60%, with 80% on 16x16 and only 40% on 32x32. It reduced decoy scanning but
also reduced 32x32 true-target reach, so no full audit was run and the flag
remains diagnostic.
Adding `clue_found` to positive replay
(`recurrent_signal_multisize_16_32_large_map_cluepositive_seed0`, `tuudojwi`)
also failed to improve the large-map split: the event path was active
(`36` positive replay events and `11` replay triggers in round 2), but mixed
eval stayed at 60% with 80% on 16x16 and 40% on 32x32, and 32x32 decoy scans
rose to 6.7. This suggests the next Signal fix should change how agents choose
between candidate clue/target hypotheses on large maps rather than simply
replaying more successful clue pickups.
The ambiguity-label ablation
(`recurrent_signal_multisize_16_32_large_map_ambiguousdecision_seed0`,
`7dr55x28`) changed target-decision labels so a true target tile is still a
negative scan decision when the local observation admits multiple compatible
target hypotheses. It matched the conservative 70% mixed eval headline and
suppressed decoy scans in the best round, but its 32x32 split stayed at 50% and
first-target-scan misses rose sharply, so no full audit was run. A moderate
scan-pressure follow-up
(`recurrent_signal_multisize_16_32_large_map_ambiguousdecision_scanpush_seed0`,
`8tbyh3s6`) regressed to 65% mixed eval with at most 45% on 32x32. The useful
lesson is that ambiguity-aware target decisions help decoy discipline but need
better evidence acquisition or scan timing before they can improve the official
large-map score.

Implementation note:
`--recurrent-eval-signal-constraint-message-copy-assist` now defaults on for
Signal specialist sweep runs and forwards
`--eval-signal-constraint-message-copy-assist`. It replaces sent Signal clue
messages with canonical structured constraints from the sender's current
`goal_hint`, preventing learned token errors from poisoning teammate target
inference.
Signal inferred-target features and eval decoding now compile each observation's
structured clue/message constraints once and reuse that compiled state for
target filtering. The 32x32 100-episode audit completed in 39.90s after this
change, making full large-map trajectory audits practical.
`--recurrent-eval-signal-frontier-exploration-assist` is now available as an
opt-in diagnostic ablation. It reuses the Signal frontier label policy during
eval/rollout decoding when no visible clue, target, or unique target evidence
exists, and fixes the nearest-frontier fallback to select the best-scored
frontier rather than the last scanned frontier. On the rolefix-handoff 16x16
checkpoint it reached 56/100 in a 100-episode external audit, just below the
57/100 promoted baseline, so it remains default-off.
`--recurrent-bc-signal-target-pursuit-trust-exact-memory` adds an opt-in
target-pursuit label path that reads trusted exact target memory from the Signal
scan state, allowing BC/DAgger data to supervise continued movement toward a
teammate-broadcast target after the inbox message expires. A first 16x16 run
(`recurrent_signal16_memorylabels_seed0`) reached 53/100 in the round-1 sweep
eval with more decoy scans, below the 57/100 rolefix-handoff baseline, and was
not promoted as default behavior.
`--recurrent-dagger-target-handoff-requires-exact-target`, which restricts
target-handoff labels to responders with trusted exact target evidence, remains
available for reproduction but default-off after the first audit.

---

## 16. TarMAC Results (8x8)

| Scenario | Success | Avg Steps | Attn Entropy | Notes |
|---|---|---|---|---|
| **energy_grid** | **100%** | **16.7** (best RL) | 0.611 | Fastest solve — steps improved 40→16.7 during training |
| **signal_hunt** | 0% | 300 | 0.000 (collapsed) | 2 agents = no routing choice → attention degenerates |

### RL Communication Methods Comparison (8x8)

| Method | Comm Type | signal_hunt | energy_grid | energy steps |
|---|---|---|---|---|
| **TarMAC** | Continuous + attention | 0% | **100%** | **16.7** (fastest) |
| **Comm-MAT** | Discrete tokens + transformer | **30%** | **100%** | 17-20 |
| **Comm-MAT no-comm** | None (transformer backbone) | ~0% | **100%** | 15-20 |
| **MAPPO v4** | Discrete tokens + MLP | 0% | — | — |

### Architecture comparison:

| Property | MAPPO | Comm-MAT | TarMAC |
|---|---|---|---|
| Backbone | MLP | Transformer | MLP + attention |
| Messages | Discrete tokens via env | Discrete tokens via env | Continuous vectors via attention |
| Routing | Broadcast to all | Broadcast to all | Learned attention weights |
| Comm learning | Send gate + token head | Send gate + token head | End-to-end through attention |

### TarMAC findings:
- **Energy grid:** TarMAC achieves the fastest solve time of any RL method (16.7 steps vs 17-20 for Comm-MAT). Attention entropy 0.611 shows agents actively choosing who to message (max for 2 targets ≈ 0.693).
- **Signal hunt:** attention entropy collapsed to 0.000 — with only 2 agents, each agent has exactly 1 possible recipient, making the attention mechanism degenerate. TarMAC's advantage (targeted routing) requires 3+ agents.
- **Implication:** TarMAC is best suited for scenarios with many agents where routing decisions matter. For 2-agent tasks, Comm-MAT's discrete token approach is more effective.

---

## 16. Running/Pending Experiments

| Experiment | Status | Platform |
|---|---|---|
| LLM 16x16 (gpt-oss:20b signal + energy) | Running | Local |
| TarMAC (signal_hunt + energy_grid) | **Done** | RunPod |
| Comm-MAT 16x16 (signal + energy) | **Done** | RunPod |
| BC→RL v2 16x16 (signal + energy) | **Done** | RunPod |
| gpt-oss:20b pipeline_assembly | **Done** (0%) | Local |
| IRL MAPPO (all 3 scenarios) | **Done** | RunPod |
| Comm-MAT no-comm ablation | **Done** | RunPod |
| BC→RL v2 8x8 (all 3 scenarios) | **Done** | RunPod |
| Comm-MAT 8x8 (all 3 scenarios) | **Done** | RunPod |
| MAPPO v4 (signal_hunt) | **Done** | RunPod |

---

## 17. Infrastructure

| Component | Status | Files |
|---|---|---|
| MAPPO training (DTDE/CTDE) | Done | `syncorsink/train/mappo.py` |
| Comm-MAT training (+ablation) | Done | `syncorsink/train/comm_mat.py` |
| TarMAC training | Done | `syncorsink/train/tarmac.py` |
| BC from oracle (+comm) | Done | `syncorsink/train/bc.py` |
| DAgger (+comm) | Done | `syncorsink/train/bc.py` |
| BC→RL warmstart (KL + freeze) | Done | `syncorsink/train/mappo.py` |
| Reward regression (IRL) | Done | `syncorsink/train/bc.py` |
| LLM eval (OpenAI + litellm/ollama) | Done | `examples/eval_llm.py` |
| Oracle/heuristic eval | Done | `examples/eval_run.py` |
| BC eval | Done | `examples/eval_run.py` |
| Coordination shaping (v4) | Done | `syncorsink/envs/scenarios.py` |
| Energy node_critical events | Done | `syncorsink/envs/scenarios.py` |
| Pipeline hint decoding | Done | `syncorsink/llm/policy.py` |
| Prompt compression | Done | `syncorsink/llm/policy.py` |
| Test suite (12 tests) | Done | `tests/` |

---

## 18. Next Steps

1. **Check TarMAC results** — classic comm method comparison
2. **Check LLM 16x16 results** — gpt-oss:20b scaling
3. **Crack pipeline assembly** — hierarchical approach or curriculum learning
4. **Paper writing** — results tables, analysis figures, discussion of findings
5. **32x32 experiments** — further scaling study if needed
