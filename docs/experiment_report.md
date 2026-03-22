# SyncOrSink Experiment Report

**Date:** 2026-03-22
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
| gpt-oss:20b | LLM (20B, local) | **67%** | 44 | Text | Open-source, free inference |
| gpt-4o-mini | LLM (API) | **60%** | 157 | Text | Action planner |
| gpt-4o-mini | LLM (API) | 30% | 213 | Text | Executor planner |
| DAgger BC | IL (42k demos) | **55%** | 137 | No | 3 rounds, 99.4% action acc |
| DAgger BC + comm | IL (43k demos) | **55%** | 137 | Tokens | Comm learned without perf loss |
| Vanilla BC | IL (354 demos) | 45% | 166 | No | From 50 oracle episodes |
| Comm-MAT | Transformer RL | **30%** | ~270 | Tokens | Improving at end of training |
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
| gpt-oss:20b | LLM (20B, local) | **100%** | 23 | Text | 3/3 episodes, remarkable |
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
| All BC/DAgger variants | IL | 0% | 300 | No | 92% action acc but can't chain deps |
| Comm-MAT | Transformer RL | 0% | 300 | Tokens | Cannot learn multi-step planning |
| MAPPO (all versions) | PPO RL | 0% | 300 | — | Not tested on this scenario |
| BC→RL v1 | IL→RL | 0% | 300 | No | RL destroyed BC init (return 36→3) |
| BC→RL v2 | IL→RL (KL reg) | pending | — | Tokens | Running with KL + frozen encoder |

---

## 3. Key Findings

### 3.1 LLMs dominate with zero training
- LLMs solve signal_hunt 60-67% and energy_grid 60-100% without any task-specific training
- Prior knowledge about spatial reasoning, coordination, and communication transfers directly
- Open-source 20B model (gpt-oss) matches or exceeds closed-source APIs on some tasks
- Pipeline_assembly remains hard even for LLMs (20% with gpt-4o-mini)

### 3.2 Comm-MAT is the first RL method to solve any scenario
- **100% success on energy_grid** — learned typed resource delivery and node recharging
- **30% on signal_hunt** — improving at end of training, likely needs more updates
- Transformer architecture with communication heads outperforms MLP-based MAPPO
- Communication send rate settles at meaningful levels (22% for energy, 3.5% for signal)

### 3.3 MAPPO fails at communication-dependent coordination
- 0% success across all 4 shaping reward versions on signal_hunt
- v1-v2: reward too weak or comm collapsed (send rate → 0%)
- v3: agents farmed scan bonus (+314 return) without coordinating (send rate 77%)
- v4: joint-scan fix prevented farming but still can't discover timing coordination
- Fundamental issue: PPO with 1.5M env steps can't discover rare joint-scan events from scratch

### 3.4 Reward shaping is a double-edged sword
- v1: shaping too weak relative to comm cost (0.01 vs 12.0 penalty) → agents learn to not move
- v2: 10x stronger shaping + lower comm cost → comm collapsed anyway
- v3: scan + co-location bonus → farming local optimum (+314 return, 0% success)
- v4: joint near-miss bonus → prevented farming but still 0% success
- **Lesson:** hand-crafted shaping creates deceptive local optima that prevent learning actual coordination

### 3.5 Behavioral Cloning is surprisingly effective
- Vanilla BC from 354 transitions: 45% success on signal_hunt (competitive with LLMs)
- DAgger improves to 55% by fixing distribution shift (42k transitions, 99.4% action acc)
- BC with communication: fails with small data (15%) but works with DAgger (55%)
- **Key insight:** DAgger is essential for making multi-agent IL with communication work
- Pipeline_assembly: 92% action accuracy but 0% success — can't chain multi-step dependencies

### 3.6 BC→RL warmstart: promising but fragile
- v1: BC initialization strong (return 36, ep_len 106) but RL destroyed it in 600 updates
- Entropy collapsed from 1.9 → 0.36 — policy became deterministic rut
- v2 (running): KL penalty + frozen encoder + lower LR should preserve BC initialization
- **Lesson:** naive IL→RL fine-tuning needs careful regularization

### 3.7 Communication learning requires sufficient data
- BC with 354 demos + comm: 15% (worse than 45% without comm)
- DAgger with 43k demos + comm: 55% (matches no-comm DAgger)
- The comm head has a large output space (send × length × tokens) that needs diverse examples
- Two-phase training (action first, comm second) is theoretically better but not yet validated at scale

### 3.8 Pipeline Assembly is the hardest scenario
- Even the oracle only achieves 50-60% success
- Requires multi-step dependency chains: pickup → deliver → sync, in order
- Partial blueprints mean agents must communicate to know the full plan
- Only LLMs (20% with gpt-4o-mini) have cracked it among learned methods
- Improved prompt (decoded stage descriptions) should help — eval pending

---

## 4. Reward Shaping Evolution (Signal Hunt)

| Version | scan_bonus | joint_scan | colocation | comm_utility | comm_cost | shaping_scale | Result |
|---|---|---|---|---|---|---|---|
| v1 | 0 | 0 | 0 | 0 | 0.01 | 0.01 | Comm penalty dominates shaping |
| v2 | 0 | 0 | 0 | 0 | 0.001 | 0.1 | Comm collapsed to 0.2% |
| v3 | 1.0 | 0 | 0.5 | 0.1 | 0.001 | 0.1 | Farming: +314 return, 0% success |
| v4 | 0.2 | 3.0 | 0.5 | 0.1 | 0.001 | 0.1 | No farming, still 0% success |

---

## 5. MAPPO Training Summary (Signal Hunt)

| Run | Critic | Updates | Best Eval Success | Final Comm Rate | Final Entropy |
|---|---|---|---|---|---|
| v1 CTDE | Central | 300 | 0% | ~49% | ~6.0 |
| v2 CTDE | Central | 3000 | 0% | 0.2% (collapsed) | 6.9 |
| v2 no-comm | Local | 3000 | 0% | N/A | 1.8 |
| v3 CTDE | Central | 3000 | 0% | 77% (farming) | 7.6 |
| v3 DTDE | Local | 3000 | 0% | 86% (farming) | 7.1 |
| v4 CTDE | Central | 3000 | 0% | 8.7% | 6.3 |

---

## 6. IL Methods Comparison (Signal Hunt)

| Method | Demo Source | Demos | Action Acc | Success | Comm? |
|---|---|---|---|---|---|
| DAgger (3 rounds) | Oracle | 42k | 99.4% | **55%** | No |
| DAgger + comm (3 rounds) | Oracle+comm | 43k | 98.7% | **55%** | Yes (2 tok/ep) |
| Vanilla BC | Oracle | 354 | 79% | 45% | No |
| Two-phase BC + comm | Oracle+comm | 354 | 71% | 15% | Yes (6 tok/ep) |
| BC from LLM | LLM traces | 742 | 60% | 0% | Yes (0 tok/ep) |

---

## 7. LLM Provider Comparison

| Model | Size | Access | signal_hunt | energy_grid | pipeline_assembly | Cost |
|---|---|---|---|---|---|---|
| gpt-oss:20b | 20B | Local (ollama) | **67%** | **100%** | pending | Free |
| gpt-4o | ~200B+ | API | — | **60%** | quota error | $$$ |
| gpt-4o-mini | ~8B | API | **60%** | 20% | **20%** | $ |

**Key insight:** Local open-source model is competitive with or superior to closed-source APIs. The benchmark is accessible without expensive API access.

---

## 8. Running Experiments

### On RunPod (A6000, 128 cores):
- BC→RL v2 with KL regularization (all 3 scenarios, with comm)
- Includes oracle+comm demo collection + DAgger + BC→RL fine-tuning

### Local:
- gpt-oss:20b on pipeline_assembly with improved prompt

---

## 9. Infrastructure Built

| Component | Status | Files |
|---|---|---|
| MAPPO training (DTDE/CTDE) | Done | `syncorsink/train/mappo.py` |
| Comm-MAT training | Done | `syncorsink/train/comm_mat.py` |
| BC from oracle | Done | `syncorsink/train/bc.py` |
| DAgger | Done | `syncorsink/train/bc.py` |
| BC→RL warmstart (KL) | Done | `syncorsink/train/mappo.py` |
| Reward regression (IRL) | Done | `syncorsink/train/bc.py` |
| LLM eval (OpenAI) | Done | `examples/eval_llm.py` |
| LLM eval (litellm/ollama) | Done | `examples/eval_llm.py` |
| Oracle/heuristic eval | Done | `examples/eval_run.py` |
| BC eval | Done | `examples/eval_run.py` |
| Coordination shaping (v4) | Done | `syncorsink/envs/scenarios.py` |
| Energy node_critical events | Done | `syncorsink/envs/scenarios.py` |
| Prompt compression | Done | `syncorsink/llm/policy.py` |
| Pipeline hint decoding | Done | `syncorsink/llm/policy.py` |
| Test suite (12 tests) | Done | `tests/` |

---

## 10. Next Steps

1. **Check BC→RL v2 results** — KL regularization + frozen encoder on all 3 scenarios
2. **Check gpt-oss:20b on pipeline_assembly** — improved prompt eval
3. **More Comm-MAT training on signal_hunt** — currently at 30% and improving, more updates likely helps
4. **TarMAC baseline** — classic emergent communication method, reviewer expectation
5. **Larger-scale experiments** — 16x16 maps, more agents, harder FOV presets
6. **Paper writing** — results tables, analysis figures, discussion of findings
