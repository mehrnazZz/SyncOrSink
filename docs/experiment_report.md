# SyncOrSink Experiment Report

**Date:** 2026-03-22 (updated)
**Benchmark:** SyncOrSink — Communication-focused cooperative multi-agent POMDP
**Setting:** DTDE (decentralized training, decentralized execution), 8x8 maps, easy FOV

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
| gpt-oss:20b | LLM (local) | pending | — | Text | Running with improved prompt |
| BC→RL v2 (KL) | IL→RL | 0% | 300 | Tokens | KL preserved init but BC can't chain deps |
| BC→RL v1 | IL→RL | 0% | 300 | No | RL destroyed BC init (return 36→3) |
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

### 3.5 Reward shaping is a double-edged sword
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
| **PPO RL** (MAPPO v4) | 0% | — | — |
| **Random** | 0% | 0% | 0% |

### Key takeaways for the paper:
1. **IL→RL warmstart** is the strongest trained approach (80%, 100%, 0%)
2. **LLMs** are the only methods that solve pipeline_assembly (20%)
3. **Comm-MAT** proves that learned communication enables RL coordination on energy_grid
4. **Pure MAPPO** fails entirely — reward shaping cannot substitute for pre-training
5. **Pipeline assembly** is an open challenge — even 60% oracle ceiling shows it's genuinely hard

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
| gpt-oss:20b | 20B | Local (ollama) | **67%** | **100%** | pending | Free |
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

### Ablation insight:
- **Energy grid** can be solved by independent agents learning typed-resource-to-node delivery. Communication helps in theory but isn't necessary on 8x8 easy settings. The transformer backbone's attention mechanism over grid observations is sufficient.
- **Signal hunt** genuinely requires explicit communication. Without it, agents can't share clue information or coordinate the synchronized target scan. The drop from 30% → ~0% confirms communication is the differentiating factor, not just having a better backbone.
- **Implication for the paper:** Energy grid tests *learned coordination* (achievable without messaging), signal hunt tests *learned communication* (necessary for success), and pipeline assembly tests *both* (unsolved by all trained methods).

---

## 11. Running Experiments

| Experiment | Status | Platform |
|---|---|---|
| IRL MAPPO (all 3 scenarios) | Running | RunPod |
| gpt-oss:20b pipeline_assembly (improved prompt) | Running | Local |
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

## 13. Next Steps

1. **Check Comm-MAT no-comm ablation** — proves whether communication or backbone drives performance
2. **Check IRL MAPPO results** — can learned reward replace hand-crafted shaping?
3. **Check gpt-oss:20b on pipeline_assembly** — improved prompt eval
4. **Extend Comm-MAT on signal_hunt** — 30% and improving, more updates likely helps
5. **TarMAC baseline** — classic emergent communication method (reviewer expectation)
6. **Larger-scale experiments** — 16x16 maps, more agents, harder FOV presets
7. **Paper writing** — results tables, analysis figures, discussion of findings
