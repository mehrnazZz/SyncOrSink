from __future__ import annotations

from typing import Dict

import numpy as np

from collections import deque

from syncorsink.envs.maps import (
    TILE_RESOURCE,
    TILE_STATION,
    TILE_NODE,
    TILE_CLUE,
    TILE_TARGET,
    TILE_WATER,
    TILE_BEACON,
)
from syncorsink.policies.pathing import shortest_path


def _local_grid_as_ids(local_grid: np.ndarray) -> np.ndarray:
    if local_grid.ndim == 3:
        # one-hot (C, H, W)
        return np.argmax(local_grid, axis=0).astype(np.int16)
    return local_grid.astype(np.int16)


def _find_nearest(local_ids: np.ndarray, tile_ids: set[int]):
    h, w = local_ids.shape
    cx, cy = w // 2, h // 2
    best = None
    best_dist = 1e9
    for y in range(h):
        for x in range(w):
            if int(local_ids[y, x]) in tile_ids:
                dist = abs(x - cx) + abs(y - cy)
                if dist < best_dist:
                    best_dist = dist
                    best = (x - cx, y - cy)
    return best


def _move_from_delta(dx: int, dy: int):
    if dx == 0 and dy == 0:
        return 4  # stay
    if abs(dx) >= abs(dy):
        return 3 if dx > 0 else 2  # right/left
    return 1 if dy > 0 else 0  # down/up


def _bfs_next_step(start: tuple[int, int], goals: set[tuple[int, int]], passable: set[int], mem: dict) -> tuple[int, int] | None:
    if start in goals:
        return (0, 0)
    q = deque([start])
    prev = {start: None}
    while q:
        cur = q.popleft()
        if cur in goals:
            # reconstruct next step
            node = cur
            while prev[node] is not None and prev[node] != start:
                node = prev[node]
            if prev[node] is None:
                return (0, 0)
            return (node[0] - start[0], node[1] - start[1])
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = cur[0] + dx, cur[1] + dy
            tile = mem.get((nx, ny))
            if tile is None:
                continue
            if tile not in passable:
                continue
            if (nx, ny) not in prev:
                prev[(nx, ny)] = cur
                q.append((nx, ny))
    return None


def _frontier_goals(mem: dict, passable: set[int]) -> set[tuple[int, int]]:
    goals = set()
    for (x, y), tile in mem.items():
        if tile not in passable:
            continue
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if (nx, ny) not in mem:
                goals.add((x, y))
                break
    return goals


def local_oracle(env):
    """
    Local (POMDP) oracle: uses only local observation + inventory.
    Does not access global env state; intended as a sanity check, not optimal.
    """
    def _policy(obs: Dict[int, dict], info: dict, state: dict):
        actions = {}
        for aid, ob in obs.items():
            local_ids = _local_grid_as_ids(ob["local_grid"])
            inv = int(ob["inventory"][0])
            # priority: if on interactable tile, interact or pickup
            center = local_ids[local_ids.shape[0] // 2, local_ids.shape[1] // 2]
            if center in (TILE_RESOURCE,) and inv == 0:
                actions[aid] = {"action": env.ACTION_PICKUP, "message_tokens": []}
                continue
            if center in (TILE_STATION, TILE_NODE, TILE_CLUE, TILE_TARGET) and inv != 0:
                actions[aid] = {"action": env.ACTION_INTERACT, "message_tokens": []}
                continue
            if center in (TILE_CLUE, TILE_TARGET) and env.config.scenario == "signal_hunt":
                actions[aid] = {"action": env.ACTION_INTERACT, "message_tokens": []}
                continue

            # pick a visible goal based on scenario
            if env.config.scenario == "signal_hunt":
                target = _find_nearest(local_ids, {TILE_TARGET, TILE_CLUE})
            elif env.config.scenario == "energy_grid":
                if inv == 0:
                    target = _find_nearest(local_ids, {TILE_RESOURCE})
                else:
                    target = _find_nearest(local_ids, {TILE_NODE})
            else:
                if inv == 0:
                    target = _find_nearest(local_ids, {TILE_RESOURCE})
                else:
                    target = _find_nearest(local_ids, {TILE_STATION})
            if target is None:
                actions[aid] = {"action": env.ACTION_STAY, "message_tokens": []}
            else:
                dx, dy = target
                actions[aid] = {"action": _move_from_delta(dx, dy), "message_tokens": []}
        return actions

    return _policy


def local_oracle_comm(env):
    """
    Local oracle with minimal communication: sends a short message when a
    high-priority tile is newly observed.
    """
    seen = {aid: set() for aid in range(env.num_agents)}
    policy = local_oracle(env)

    def _policy(obs: Dict[int, dict], info: dict, state: dict):
        actions = policy(obs, info, state)
        for aid, ob in obs.items():
            local_ids = _local_grid_as_ids(ob["local_grid"])
            # find best visible tile
            priority = [TILE_TARGET, TILE_NODE, TILE_STATION, TILE_CLUE, TILE_RESOURCE]
            found = None
            for t in priority:
                if np.any(local_ids == t):
                    found = t
                    break
            if found is None:
                continue
            if found not in seen[aid]:
                seen[aid].add(found)
                actions[aid]["message_tokens"] = [9, int(found)]
        return actions

    return _policy


def local_oracle_plus(env):
    """
    Stronger local oracle: keeps a per-agent local map with dead-reckoned pose
    and explores frontiers when no target is known.
    """
    passable = {0, TILE_RESOURCE, TILE_STATION, TILE_NODE, TILE_CLUE, TILE_TARGET}
    poses = {aid: (0, 0) for aid in range(env.num_agents)}
    mem = {aid: {} for aid in range(env.num_agents)}
    last_action = {aid: env.ACTION_STAY for aid in range(env.num_agents)}
    last_step = -1

    def _update_pose(aid: int):
        x, y = poses[aid]
        act = last_action[aid]
        if act == env.ACTION_UP:
            poses[aid] = (x, y - 1)
        elif act == env.ACTION_DOWN:
            poses[aid] = (x, y + 1)
        elif act == env.ACTION_LEFT:
            poses[aid] = (x - 1, y)
        elif act == env.ACTION_RIGHT:
            poses[aid] = (x + 1, y)

    def _policy(obs: Dict[int, dict], info: dict, state: dict):
        nonlocal last_step
        step = int(state.get("step", 0))
        if step == 0 and last_step != 0:
            for aid in range(env.num_agents):
                poses[aid] = (0, 0)
                mem[aid] = {}
                last_action[aid] = env.ACTION_STAY
        if step != last_step:
            for aid in range(env.num_agents):
                _update_pose(aid)
        last_step = step

        actions = {}
        for aid, ob in obs.items():
            local_ids = _local_grid_as_ids(ob["local_grid"])
            inv = int(ob["inventory"][0])
            px, py = poses[aid]

            # update memory
            h, w = local_ids.shape
            cx, cy = w // 2, h // 2
            for y in range(h):
                for x in range(w):
                    gx, gy = px + (x - cx), py + (y - cy)
                    mem[aid][(gx, gy)] = int(local_ids[y, x])

            # immediate action if on resource or interactable
            center = local_ids[cy, cx]
            if center == TILE_RESOURCE and inv == 0:
                actions[aid] = {"action": env.ACTION_PICKUP, "message_tokens": []}
                last_action[aid] = env.ACTION_PICKUP
                continue
            if center in (TILE_STATION, TILE_NODE) and inv != 0:
                actions[aid] = {"action": env.ACTION_INTERACT, "message_tokens": []}
                last_action[aid] = env.ACTION_INTERACT
                continue
            if env.config.scenario == "signal_hunt" and center in (TILE_CLUE, TILE_TARGET):
                actions[aid] = {"action": env.ACTION_INTERACT, "message_tokens": []}
                last_action[aid] = env.ACTION_INTERACT
                continue

            # select goal set from memory
            goals = set()
            if env.config.scenario == "signal_hunt":
                goals = {pos for pos, t in mem[aid].items() if t in (TILE_TARGET, TILE_CLUE)}
            elif env.config.scenario == "energy_grid":
                if inv == 0:
                    goals = {pos for pos, t in mem[aid].items() if t == TILE_RESOURCE}
                else:
                    goals = {pos for pos, t in mem[aid].items() if t == TILE_NODE}
            else:
                if inv == 0:
                    goals = {pos for pos, t in mem[aid].items() if t == TILE_RESOURCE}
                else:
                    goals = {pos for pos, t in mem[aid].items() if t == TILE_STATION}

            step_delta = None
            if goals:
                step_delta = _bfs_next_step((px, py), goals, passable, mem[aid])
            if step_delta is None:
                # explore frontier
                frontier = _frontier_goals(mem[aid], passable)
                if frontier:
                    step_delta = _bfs_next_step((px, py), frontier, passable, mem[aid])
            if step_delta is None:
                actions[aid] = {"action": env.ACTION_STAY, "message_tokens": []}
                last_action[aid] = env.ACTION_STAY
            else:
                dx, dy = step_delta
                act = _move_from_delta(dx, dy)
                actions[aid] = {"action": act, "message_tokens": []}
                last_action[aid] = act

        return actions

    return _policy


def local_oracle_plus_comm(env):
    policy = local_oracle_plus(env)
    seen = {aid: set() for aid in range(env.num_agents)}

    def _policy(obs: Dict[int, dict], info: dict, state: dict):
        actions = policy(obs, info, state)
        for aid, ob in obs.items():
            local_ids = _local_grid_as_ids(ob["local_grid"])
            priority = [TILE_TARGET, TILE_NODE, TILE_STATION, TILE_CLUE, TILE_RESOURCE]
            found = None
            for t in priority:
                if np.any(local_ids == t):
                    found = t
                    break
            if found is None:
                continue
            if found not in seen[aid]:
                seen[aid].add(found)
                actions[aid]["message_tokens"] = [9, int(found)]
        return actions

    return _policy


def local_oracle_team_comm(env):
    """
    Team-level local oracle with shared global map built from self_pos + local observations.
    Uses a simple comm protocol to share discovered tiles with absolute coords.
    """
    passable = {0, TILE_RESOURCE, TILE_STATION, TILE_NODE, TILE_CLUE, TILE_TARGET}
    shared_mem: dict[tuple[int, int], int] = {}
    last_sent = {aid: set() for aid in range(env.num_agents)}

    def _decode_messages(obs):
        # message format: [10, tile_id, x, y]
        tokens = obs.get("messages_tokens")
        if tokens is None:
            return
        for msg in tokens:
            if len(msg) < 4:
                continue
            if int(msg[0]) != 10:
                continue
            tile = int(msg[1])
            x = int(msg[2])
            y = int(msg[3])
            shared_mem[(x, y)] = tile

    def _policy(obs: Dict[int, dict], info: dict, state: dict):
        actions = {}
        # integrate received messages into shared map
        for aid, ob in obs.items():
            _decode_messages(ob)

        for aid, ob in obs.items():
            local_ids = _local_grid_as_ids(ob["local_grid"])
            inv = int(ob["inventory"][0])
            px, py = int(ob["self_pos"][0]), int(ob["self_pos"][1])

            # update shared map with visible tiles
            h, w = local_ids.shape
            cx, cy = w // 2, h // 2
            for y in range(h):
                for x in range(w):
                    gx, gy = px + (x - cx), py + (y - cy)
                    tile = int(local_ids[y, x])
                    shared_mem[(gx, gy)] = tile
                    if tile in (TILE_RESOURCE, TILE_STATION, TILE_NODE, TILE_CLUE, TILE_TARGET):
                        key = (tile, gx, gy)
                        if key not in last_sent[aid]:
                            last_sent[aid].add(key)
                            actions.setdefault(aid, {"action": env.ACTION_STAY, "message_tokens": []})
                            actions[aid]["message_tokens"] = [10, tile, gx, gy]

            # action selection
            if aid not in actions:
                actions[aid] = {"action": env.ACTION_STAY, "message_tokens": []}

            center = local_ids[cy, cx]
            if center == TILE_RESOURCE and inv == 0:
                actions[aid]["action"] = env.ACTION_PICKUP
                continue
            if center in (TILE_STATION, TILE_NODE) and inv != 0:
                actions[aid]["action"] = env.ACTION_INTERACT
                continue
            if env.config.scenario == "signal_hunt" and center in (TILE_CLUE, TILE_TARGET):
                actions[aid]["action"] = env.ACTION_INTERACT
                continue

            # pick goals from shared memory
            goals = set()
            if env.config.scenario == "pipeline_assembly":
                if inv == 0:
                    goals = {pos for pos, t in shared_mem.items() if t == TILE_RESOURCE}
                else:
                    goals = {pos for pos, t in shared_mem.items() if t == TILE_STATION}
            elif env.config.scenario == "energy_grid":
                if inv == 0:
                    goals = {pos for pos, t in shared_mem.items() if t == TILE_RESOURCE}
                else:
                    goals = {pos for pos, t in shared_mem.items() if t == TILE_NODE}
            else:
                goals = {pos for pos, t in shared_mem.items() if t in (TILE_TARGET, TILE_CLUE)}

            step_delta = None
            if goals:
                step_delta = _bfs_next_step((px, py), goals, passable, shared_mem)
            if step_delta is None:
                frontier = _frontier_goals(shared_mem, passable)
                if frontier:
                    step_delta = _bfs_next_step((px, py), frontier, passable, shared_mem)
            if step_delta is not None:
                dx, dy = step_delta
                actions[aid]["action"] = _move_from_delta(dx, dy)

        return actions

    return _policy


def local_pipeline_policy(env):
    """
    Scenario-specific local policy for pipeline assembly using goal hints + local types.
    """
    passable = {0, TILE_RESOURCE, TILE_STATION}
    shared_mem: dict[tuple[int, int], int] = {}
    shared_resource_types: dict[tuple[int, int], int] = {}

    def _decode_hint(hint: np.ndarray):
        # layout: [stage, sx, sy, req_len, req0, req1, deps_len, dep0, sync] ...
        if hint is None or len(hint) < 9 or hint[0] < 0:
            return None
        stage = int(hint[0])
        sx, sy = int(hint[1]), int(hint[2])
        req_len = int(hint[3])
        reqs = []
        if req_len > 0:
            reqs.append(int(hint[4]))
        if req_len > 1:
            reqs.append(int(hint[5]))
        sync = int(hint[8]) == 1
        return {"stage": stage, "station": (sx, sy), "reqs": reqs, "sync": sync}

    def _policy(obs: Dict[int, dict], info: dict, state: dict):
        actions = {}
        for aid, ob in obs.items():
            local_ids = _local_grid_as_ids(ob["local_grid"])
            inv = int(ob["inventory"][0])
            px, py = int(ob["self_pos"][0]), int(ob["self_pos"][1])
            local_resource = ob.get("local_resource_types")

            # update shared maps
            h, w = local_ids.shape
            cx, cy = w // 2, h // 2
            for y in range(h):
                for x in range(w):
                    gx, gy = px + (x - cx), py + (y - cy)
                    tile = int(local_ids[y, x])
                    shared_mem[(gx, gy)] = tile
                    if local_resource is not None and int(local_resource[y, x]) > 0:
                        shared_resource_types[(gx, gy)] = int(local_resource[y, x])

            hint = _decode_hint(ob.get("goal_hint"))
            target_station = hint["station"] if hint else None
            required = hint["reqs"] if hint else []

            center = local_ids[cy, cx]
            if center == TILE_RESOURCE and inv == 0:
                actions[aid] = {"action": env.ACTION_PICKUP, "message_tokens": []}
                continue
            if center == TILE_STATION and inv != 0:
                actions[aid] = {"action": env.ACTION_INTERACT, "message_tokens": []}
                continue

            # choose goal
            goals = set()
            if inv == 0 and required:
                goals = {pos for pos, t in shared_resource_types.items() if t in required}
            elif inv == 0:
                goals = {pos for pos, t in shared_mem.items() if t == TILE_RESOURCE}
            else:
                if target_station is not None:
                    goals = {target_station}
                else:
                    goals = {pos for pos, t in shared_mem.items() if t == TILE_STATION}

            step_delta = None
            if goals:
                step_delta = _bfs_next_step((px, py), goals, passable, shared_mem)
            if step_delta is None:
                frontier = _frontier_goals(shared_mem, passable)
                if frontier:
                    step_delta = _bfs_next_step((px, py), frontier, passable, shared_mem)
            if step_delta is None:
                actions[aid] = {"action": env.ACTION_STAY, "message_tokens": []}
            else:
                dx, dy = step_delta
                actions[aid] = {"action": _move_from_delta(dx, dy), "message_tokens": []}
        return actions

    return _policy


def local_energy_policy(env):
    """
    Scenario-specific local policy for energy grid using node/resource types.
    """
    passable = {0, TILE_RESOURCE, TILE_NODE}
    shared_mem: dict[tuple[int, int], int] = {}
    shared_resource_types: dict[tuple[int, int], int] = {}
    shared_node_types: dict[tuple[int, int], int] = {}
    shared_node_energy: dict[tuple[int, int], int] = {}

    def _policy(obs: Dict[int, dict], info: dict, state: dict):
        actions = {}
        for aid, ob in obs.items():
            local_ids = _local_grid_as_ids(ob["local_grid"])
            inv = int(ob["inventory"][0])
            px, py = int(ob["self_pos"][0]), int(ob["self_pos"][1])
            local_resource = ob.get("local_resource_types")
            local_node = ob.get("local_node_types")
            local_energy = ob.get("local_node_energy")

            # update shared maps
            h, w = local_ids.shape
            cx, cy = w // 2, h // 2
            for y in range(h):
                for x in range(w):
                    gx, gy = px + (x - cx), py + (y - cy)
                    tile = int(local_ids[y, x])
                    shared_mem[(gx, gy)] = tile
                    if local_resource is not None and int(local_resource[y, x]) > 0:
                        shared_resource_types[(gx, gy)] = int(local_resource[y, x])
                    if local_node is not None and int(local_node[y, x]) > 0:
                        shared_node_types[(gx, gy)] = int(local_node[y, x])
                    if local_energy is not None and int(local_energy[y, x]) > 0:
                        shared_node_energy[(gx, gy)] = int(local_energy[y, x])

            center = local_ids[cy, cx]
            if center == TILE_RESOURCE and inv == 0:
                actions[aid] = {"action": env.ACTION_PICKUP, "message_tokens": []}
                continue
            if center == TILE_NODE and inv != 0:
                actions[aid] = {"action": env.ACTION_INTERACT, "message_tokens": []}
                continue

            # choose goal
            goals = set()
            if inv == 0:
                # pick any resource, prefer one with a known matching node type
                goals = set(shared_resource_types.keys())
            else:
                goals = {pos for pos, t in shared_node_types.items() if t == inv}
                if not goals:
                    goals = {pos for pos, t in shared_mem.items() if t == TILE_NODE}
                # prefer lowest-energy node if known
                if goals and shared_node_energy:
                    goals = {min(goals, key=lambda p: shared_node_energy.get(p, 999))}

            step_delta = None
            if goals:
                step_delta = _bfs_next_step((px, py), goals, passable, shared_mem)
            if step_delta is None:
                frontier = _frontier_goals(shared_mem, passable)
                if frontier:
                    step_delta = _bfs_next_step((px, py), frontier, passable, shared_mem)
            if step_delta is None:
                actions[aid] = {"action": env.ACTION_STAY, "message_tokens": []}
            else:
                dx, dy = step_delta
                actions[aid] = {"action": _move_from_delta(dx, dy), "message_tokens": []}
        return actions

    return _policy


def local_signal_policy(env):
    """
    Scenario-specific local policy for signal hunt using shared map + hints.
    """
    passable = {0, TILE_CLUE, TILE_TARGET, TILE_WATER, TILE_BEACON}
    shared_mem: dict[tuple[int, int], int] = {}
    claimed_clues: set[tuple[int, int]] = set()
    rejected_targets: set[tuple[int, int]] = set()
    def _new_constraints():
        return {
            "exact_targets": set(),
            "near": [],
            "parity": None,
            "quadrant": None,
            "quadrant_size": env.map_size,
            "x_parity": None,
            "y_parity": None,
        }

    constraints = _new_constraints()
    last_sent: dict[int, list[int]] = {}
    sent_messages: dict[int, set[tuple[int, ...]]] = {}
    last_target_interact_pos: dict[int, tuple[int, int]] = {}
    last_step = -1

    def _clip_msg(tokens: list[int]) -> list[int]:
        return [max(0, min(env.token_vocab_size - 1, int(t))) for t in tokens][: env.comm_token_limit]

    def _add_exact_target(x: int, y: int):
        if 0 <= x < env.map_size and 0 <= y < env.map_size:
            constraints["exact_targets"].add((x, y))

    def _signal_segments(raw) -> list[list[int]]:
        if raw is None:
            return []
        try:
            tokens = [int(v) for v in raw.tolist()]
        except AttributeError:
            tokens = [int(v) for v in raw]
        segments: list[list[int]] = []
        i = 0
        while i < len(tokens):
            code = tokens[i]
            if code < 0:
                break
            length = {21: 5, 22: 6, 23: 4, 24: 2, 25: 2, 26: 3}.get(code)
            if length is None or i + length > len(tokens):
                break
            segment = tokens[i: i + length]
            segments.append(segment)
            i += length
        return segments

    def _parse_signal_tokens(raw):
        for segment in _signal_segments(raw):
            code = segment[0]
            if code == 21:
                obj = int(segment[1])
                ox, oy = int(segment[2]), int(segment[3])
                dist = int(segment[4])
                if 0 <= ox < env.map_size and 0 <= oy < env.map_size:
                    item = (obj, (ox, oy), dist)
                    if item not in constraints["near"]:
                        constraints["near"].append(item)
            elif code == 22:
                ox, oy = int(segment[2]), int(segment[3])
                dx, dy = int(segment[4]), int(segment[5])
                _add_exact_target(ox + dx, oy + dy)
            elif code == 23:
                constraints["parity"] = int(segment[1])
                for name, value in {"NW": 0, "NE": 1, "SW": 2, "SE": 3}.items():
                    if int(segment[2]) == value:
                        constraints["quadrant"] = name
                        break
                constraints["quadrant_size"] = int(segment[3])
            elif code == 24:
                constraints["x_parity"] = int(segment[1])
            elif code == 25:
                constraints["y_parity"] = int(segment[1])
            elif code == 26:
                _add_exact_target(int(segment[1]), int(segment[2]))
            else:
                break

    def _parse_inbox(ob: dict):
        tokens = ob.get("messages_tokens")
        if tokens is None:
            return
        for msg in tokens:
            _parse_signal_tokens(msg)

    def _record_feedback(info: dict):
        events = (info or {}).get("events", {})
        if not isinstance(events, dict):
            return
        for aid_raw, agent_events in events.items():
            try:
                aid = int(aid_raw)
            except (TypeError, ValueError):
                aid = aid_raw
            if not isinstance(agent_events, list):
                continue
            for event in agent_events:
                if not isinstance(event, dict):
                    continue
                if event.get("event") == "decoy_scan" and aid in last_target_interact_pos:
                    rejected_targets.add(last_target_interact_pos[aid])

    def _in_quadrant(pos: tuple[int, int]) -> bool:
        q = constraints["quadrant"]
        if q is None:
            return True
        x, y = pos
        half = float(constraints.get("quadrant_size") or env.map_size) / 2.0
        if q == "NW":
            return x < half and y < half
        if q == "NE":
            return x >= half and y < half
        if q == "SW":
            return x < half and y >= half
        return x >= half and y >= half

    def _matches_constraints(pos: tuple[int, int]) -> bool:
        exact_targets = constraints["exact_targets"]
        if exact_targets and pos not in exact_targets:
            return False
        if constraints["parity"] is not None and (pos[0] + pos[1]) % 2 != constraints["parity"]:
            return False
        if constraints["x_parity"] is not None and pos[0] % 2 != constraints["x_parity"]:
            return False
        if constraints["y_parity"] is not None and pos[1] % 2 != constraints["y_parity"]:
            return False
        if not _in_quadrant(pos):
            return False
        for _obj, near_pos, dist in constraints["near"]:
            if abs(pos[0] - near_pos[0]) + abs(pos[1] - near_pos[1]) > dist:
                return False
        return True

    def _candidate_targets() -> set[tuple[int, int]]:
        exact_targets = set(constraints["exact_targets"]) - rejected_targets
        observed = {
            pos for pos, tile in shared_mem.items()
            if tile == TILE_TARGET and pos not in rejected_targets
        }
        if exact_targets:
            return exact_targets
        return {pos for pos in observed if _matches_constraints(pos)}

    def _team_candidate(candidates: set[tuple[int, int]]) -> tuple[int, int] | None:
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda pos: (
                sum(abs(pos[0] - ax) + abs(pos[1] - ay) for ax, ay in env.agent_positions),
                pos[1],
                pos[0],
            ),
        )

    def _message_for_agent(aid: int, ob: dict) -> list[int]:
        exact_targets = sorted(constraints["exact_targets"])
        if exact_targets:
            tx, ty = exact_targets[0]
            return _clip_msg([26, tx, ty])
        seen = sent_messages.setdefault(aid, set())
        for segment in reversed(_signal_segments(ob.get("goal_hint"))):
            key = tuple(segment)
            if key not in seen:
                seen.add(key)
                return _clip_msg(segment)
        return []

    def _move_toward(agent_id: int, pos: tuple[int, int], goals: set[tuple[int, int]], mem: dict):
        if not goals:
            return None
        blocked_positions = {p for idx, p in enumerate(env.agent_positions) if idx != agent_id and p not in goals}
        dx, dy, reached = shortest_path(env.grid, pos, goals, blocked_positions=blocked_positions)
        if reached is not None:
            return dx, dy
        return _bfs_next_step(pos, goals, passable, mem)

    def _policy(obs: Dict[int, dict], info: dict, state: dict):
        nonlocal constraints, last_step
        step = int(state.get("step", 0))
        if step == 0 and last_step != 0:
            shared_mem.clear()
            claimed_clues.clear()
            rejected_targets.clear()
            constraints = _new_constraints()
            last_sent.clear()
            sent_messages.clear()
            last_target_interact_pos.clear()
        last_step = step
        _record_feedback(info)
        actions = {}
        for aid, ob in obs.items():
            _parse_signal_tokens(ob.get("goal_hint"))
            _parse_inbox(ob)
            local_ids = _local_grid_as_ids(ob["local_grid"])
            px, py = int(ob["self_pos"][0]), int(ob["self_pos"][1])

            # update shared map
            h, w = local_ids.shape
            cx, cy = w // 2, h // 2
            for y in range(h):
                for x in range(w):
                    gx, gy = px + (x - cx), py + (y - cy)
                    tile = int(local_ids[y, x])
                    shared_mem[(gx, gy)] = tile

            center = local_ids[cy, cx]
            message = _message_for_agent(aid, ob)
            send = message and message != last_sent.get(aid)
            if send:
                last_sent[aid] = message
            else:
                message = []

            exact_targets = set(constraints["exact_targets"])
            candidate_targets = _candidate_targets()
            unclaimed_clues = {
                pos for pos, tile in shared_mem.items()
                if tile == TILE_CLUE and pos not in claimed_clues
            }
            frontier = _frontier_goals(shared_mem, passable)
            safe_targets = exact_targets or (candidate_targets if len(candidate_targets) == 1 else set())
            commit_targets: set[tuple[int, int]] = set()
            if not safe_targets and not unclaimed_clues and not frontier:
                candidate = _team_candidate(candidate_targets)
                if candidate is not None:
                    commit_targets = {candidate}
            scan_targets = safe_targets | commit_targets

            if center == TILE_CLUE and (px, py) not in claimed_clues:
                claimed_clues.add((px, py))
                actions[aid] = {"action": env.ACTION_INTERACT, "message_tokens": message}
                last_target_interact_pos.pop(aid, None)
                continue
            if center == TILE_TARGET and (px, py) in scan_targets:
                actions[aid] = {"action": env.ACTION_INTERACT, "message_tokens": message}
                last_target_interact_pos[aid] = (px, py)
                continue

            if safe_targets:
                goals = safe_targets
            elif unclaimed_clues:
                goals = unclaimed_clues
            elif commit_targets:
                goals = commit_targets
            else:
                goals = set()

            step_delta = None
            if goals:
                step_delta = _move_toward(aid, (px, py), goals, shared_mem)
            if step_delta is None:
                if frontier:
                    step_delta = _bfs_next_step((px, py), frontier, passable, shared_mem)
            if step_delta is None:
                actions[aid] = {"action": env.ACTION_STAY, "message_tokens": message}
                last_target_interact_pos.pop(aid, None)
            else:
                dx, dy = step_delta
                actions[aid] = {"action": _move_from_delta(dx, dy), "message_tokens": message}
                last_target_interact_pos.pop(aid, None)
        return actions

    return _policy
