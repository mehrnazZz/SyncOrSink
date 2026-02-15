from __future__ import annotations

from collections import deque


def shortest_path(grid, start, goals, blocked=(1, 9), blocked_positions=None):
    """
    BFS shortest path from start to any goal tile.
    Returns next step (dx, dy) and target position, or (0,0,None) if unreachable.
    """
    size = grid.shape[0]
    q = deque([start])
    prev = {start: None}
    blocked_positions = blocked_positions or set()
    while q:
        x, y = q.popleft()
        if (x, y) in goals:
            # reconstruct next move
            cur = (x, y)
            while prev[cur] is not None and prev[cur] != start:
                cur = prev[cur]
            if prev[cur] is None:
                return (0, 0, (x, y))
            dx = cur[0] - start[0]
            dy = cur[1] - start[1]
            return (dx, dy, (x, y))
        for dx, dy in ((1,0), (-1,0), (0,1), (0,-1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < size and 0 <= ny < size:
                if (nx, ny) in blocked_positions:
                    continue
                if grid[ny, nx] in blocked:
                    continue
                if (nx, ny) not in prev:
                    prev[(nx, ny)] = (x, y)
                    q.append((nx, ny))
    return (0, 0, None)


def move_action_from_delta(dx, dy, env):
    if dx > 0:
        return env.ACTION_RIGHT
    if dx < 0:
        return env.ACTION_LEFT
    if dy > 0:
        return env.ACTION_DOWN
    if dy < 0:
        return env.ACTION_UP
    return env.ACTION_STAY


def shortest_path_distance(grid, start, goals, blocked=(1, 9), blocked_positions=None):
    """
    BFS distance from start to any goal tile.
    Returns distance (int) or None if unreachable.
    """
    size = grid.shape[0]
    blocked_positions = blocked_positions or set()
    q = deque([start])
    dist = {start: 0}
    while q:
        x, y = q.popleft()
        if (x, y) in goals:
            return dist[(x, y)]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < size and 0 <= ny < size:
                if (nx, ny) in blocked_positions:
                    continue
                if grid[ny, nx] in blocked:
                    continue
                if (nx, ny) not in dist:
                    dist[(nx, ny)] = dist[(x, y)] + 1
                    q.append((nx, ny))
    return None
