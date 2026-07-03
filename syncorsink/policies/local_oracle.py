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
)


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
    passable = {0, TILE_CLUE, TILE_TARGET}
    shared_mem: dict[tuple[int, int], int] = {}
    constraints = {"water": set(), "beacon": set(), "parity": None, "quadrant": None}

    def _parse_goal_hint_tokens(raw):
        if raw is None:
            return
        try:
            tokens = [int(v) for v in raw.tolist()]
        except AttributeError:
            tokens = [int(v) for v in raw]
        i = 0
        while i < len(tokens):
            code = tokens[i]
            if code < 0:
                break
            if code == 21 and i + 4 < len(tokens):
                constraints["water"].add("near")
                i += 5
            elif code == 22 and i + 5 < len(tokens):
                constraints["beacon"].add("east2")
                i += 6
            elif code == 23 and i + 3 < len(tokens):
                constraints["parity"] = int(tokens[i + 1])
                for name, value in {"NW": 0, "NE": 1, "SW": 2, "SE": 3}.items():
                    if int(tokens[i + 2]) == value:
                        constraints["quadrant"] = name
                        break
                i += 4
            elif code == 24 and i + 1 < len(tokens):
                i += 2
            elif code == 25 and i + 1 < len(tokens):
                i += 2
            else:
                break

    def _policy(obs: Dict[int, dict], info: dict, state: dict):
        actions = {}
        for aid, ob in obs.items():
            _parse_goal_hint_tokens(ob.get("goal_hint"))
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
            if center in (TILE_CLUE, TILE_TARGET):
                actions[aid] = {"action": env.ACTION_INTERACT, "message_tokens": []}
                continue

            # if target known from shared map, go there
            goals = {pos for pos, t in shared_mem.items() if t == TILE_TARGET}
            if not goals:
                goals = {pos for pos, t in shared_mem.items() if t == TILE_CLUE}
            # apply weak constraints when target unknown
            if not goals and constraints["quadrant"] is not None:
                q = constraints["quadrant"]
                def in_quad(p):
                    x, y = p
                    half = env.map_size / 2
                    if q == "NW":
                        return x < half and y < half
                    if q == "NE":
                        return x >= half and y < half
                    if q == "SW":
                        return x < half and y >= half
                    return x >= half and y >= half
                goals = {pos for pos, t in shared_mem.items() if t in (TILE_CLUE, TILE_TARGET) and in_quad(pos)}

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
