# SyncOrSink Design

This document describes the shared world mechanics, observation/action spaces, and communication design. Scenario‑specific success conditions and rewards are in `docs/scenarios.md`.

## Coordination Scenarios

1. Pipeline Assembly (task planning)
- Agents get partial blueprints.
- Multi-stage dependency DAG across stations.
- Stages require multiple resource types and some require synchronized interaction.
- Requires semantic coordination, sequencing, and synchronized actions.

2. Energy Grid (resource sharing)
- Shared nodes drain energy over time.
- Fuel cells must be delivered to keep nodes alive.
- Nodes have types; resources must match.
- Low-energy nodes require synchronized recharge by multiple agents.
- Communication is required to coordinate coverage and delivery.

3. Signal Hunt (cooperative search)
- Hidden target with distributed clues.
- Agents share textual clues to triangulate the target.
- Clue tiles can be collected on-site for additional hints.
- Two agents must scan the target within a short window to complete.
- Map includes rooms, doors, occlusion, water, and beacon landmarks.
- Decoy targets are present; clues constrain the true target.
- Configurable toggles: rooms/doors, fog-of-war, decoy count and penalties.

## Communication

Two modes supported:
- Tokens: bounded integer sequences (for algorithmic agents)
- Text: free text strings (for LLM agents)

Both are represented in the action payload. Token budgets apply to both and are costed for communication efficiency.
Optional radius‑limited messaging via `comm_radius` (None = broadcast).

## POMDP Presets

FOV presets (radius):
- hard: 2
- medium: 3
- easy: 4

## Metrics

- Task success / time-to-success
- Communication efficiency
- Generalization across unseen maps

## Action Space

Discrete actions (shared across scenarios):
- `0`: up
- `1`: down
- `2`: left
- `3`: right
- `4`: stay
- `5`: interact
- `6`: pickup
- `7`: drop

Action payload per agent:
- `action`: discrete id
- `message_tokens`: list of ints (token comm mode)
- `message_text`: free text (text comm mode)

## Observation Space

Per‑agent observation dict:
- `local_grid`: `(2r+1, 2r+1)` grid of tile ids (unknown tiles are `TILE_UNKNOWN`)
- If `obs_onehot=True`, `local_grid` becomes `(C,H,W)` one‑hot channels
- `inventory`: `(1,)` int
- `self_pos`: `(2,)` absolute `(x,y)` position
- `local_resource_types`: `(2r+1, 2r+1)` int grid (resource type ids, `0` means none)
- `local_node_types`: `(2r+1, 2r+1)` int grid (node type ids, `0` means none)
- `local_node_energy`: `(2r+1, 2r+1)` int grid (node energy values)
- `messages_tokens`: `(max_messages, comm_token_limit)`
- `message_from`: `(max_messages,)`
- `goal_hint`: `(16,)` per-agent private hint tokens (scenario-dependent)
- `explored_mask`: `(map_size, map_size)` binary per-agent explored map (if `obs_exploration_memory=True`)
- `explored_age`: `(map_size, map_size)` steps since last seen, `-1` unseen (if `obs_exploration_age=True`)

Info dict (per step):
- `messages_text`: list of incoming message strings (text mode)
- `messages_with_sender`: list of incoming `{from, text}` entries (text mode)
- `comm_tokens`: per‑agent token counts
- `central_obs` (CTDE track only): full grid + agent positions + inventories

Shared `info` must not expose private scenario state such as Signal Hunt clue
constraints, Pipeline blueprints, or Energy Grid node assignments. DTDE policies
should treat per-agent observations and received messages as the only private
information channels.

## World Elements (Tiles)

Tiles include walls, doors, resources, stations, nodes, clues, targets, water, and beacons. Rooms/corridors are generated for structured exploration.
