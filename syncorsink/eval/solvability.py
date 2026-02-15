from __future__ import annotations

from typing import Dict, List, Tuple
from collections import deque


def _bfs_reachable(grid, start):
    size = grid.shape[0]
    blocked = {1, 9}  # walls, doors
    q = deque([start])
    seen = {start}
    while q:
        x, y = q.popleft()
        for dx, dy in ((1,0), (-1,0), (0,1), (0,-1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < size and 0 <= ny < size:
                if grid[ny, nx] in blocked:
                    continue
                if (nx, ny) not in seen:
                    seen.add((nx, ny))
                    q.append((nx, ny))
    return seen


def _any_reachable(grid, starts, targets):
    for s in starts:
        reach = _bfs_reachable(grid, s)
        if any(t in reach for t in targets):
            return True
    return False


def check_pipeline_feasible(env) -> Tuple[bool, str]:
    starts = env.meta.get("agent_starts", [])
    stations = env.meta.get("stations", [])
    resources = list(env.scenario_state.data.get("resource_types", {}).keys())
    if not stations or not resources:
        return False, "missing stations or resources"
    if not _any_reachable(env.grid, starts, stations):
        return False, "stations unreachable"
    if not _any_reachable(env.grid, starts, resources):
        return False, "resources unreachable"
    return True, "ok"


def check_energy_feasible(env) -> Tuple[bool, str]:
    starts = env.meta.get("agent_starts", [])
    nodes = env.meta.get("nodes", [])
    resources = list(env.scenario_state.data.get("resource_types", {}).keys())
    if not nodes or not resources:
        return False, "missing nodes or resources"
    if not _any_reachable(env.grid, starts, nodes):
        return False, "nodes unreachable"
    if not _any_reachable(env.grid, starts, resources):
        return False, "resources unreachable"
    return True, "ok"


def check_signal_hunt_feasible(env) -> Tuple[bool, str]:
    starts = env.meta.get("agent_starts", [])
    targets = env.meta.get("targets", [])
    clues = env.meta.get("clues", [])
    if not targets or not clues:
        return False, "missing targets or clues"
    if not _any_reachable(env.grid, starts, targets):
        return False, "targets unreachable"
    if not _any_reachable(env.grid, starts, clues):
        return False, "clues unreachable"
    # ensure constraints identify true target vs decoys
    target = env.scenario_state.data.get("target")
    decoys = env.scenario_state.data.get("decoys", [])
    constraints = env.scenario_state.data.get("constraints", [])
    if target is None:
        return False, "missing target"
    if _constraints_match(target, constraints) is False:
        return False, "constraints don't match target"
    for d in decoys:
        if _constraints_match(d, constraints):
            return False, "constraints match decoy"
    return True, "ok"


def _constraints_match(pos, constraints) -> bool:
    x, y = pos
    for c in constraints:
        if c["type"] == "near":
            px, py = c["pos"]
            dist = abs(px - x) + abs(py - y)
            if dist > c["dist"]:
                return False
        elif c["type"] == "offset":
            px, py = c["pos"]
            if (px + c["dx"], py + c["dy"]) != (x, y):
                return False
        elif c["type"] == "parity_quadrant":
            parity = c["parity"]
            quadrant = c["quadrant"]
            if (x + y) % 2 != parity:
                return False
            size = c.get("size")
            if size:
                if quadrant == "NW" and not (x < size / 2 and y < size / 2):
                    return False
                if quadrant == "NE" and not (x >= size / 2 and y < size / 2):
                    return False
                if quadrant == "SW" and not (x < size / 2 and y >= size / 2):
                    return False
                if quadrant == "SE" and not (x >= size / 2 and y >= size / 2):
                    return False
        elif c["type"] == "x_parity":
            if x % 2 != c["value"]:
                return False
        elif c["type"] == "y_parity":
            if y % 2 != c["value"]:
                return False
    return True


def check_solvability(env) -> Tuple[bool, str]:
    if env.config.scenario == "pipeline_assembly":
        return check_pipeline_feasible(env)
    if env.config.scenario == "energy_grid":
        return check_energy_feasible(env)
    if env.config.scenario == "signal_hunt":
        return check_signal_hunt_feasible(env)
    return False, "unknown scenario"
