from __future__ import annotations

import numpy as np
from .maps import TILE_WALL, TILE_UNKNOWN, TILE_DOOR


def extract_local(grid: np.ndarray, pos: tuple[int, int], radius: int) -> np.ndarray:
    size = grid.shape[0]
    x, y = pos
    span = radius * 2 + 1
    local = np.full((span, span), TILE_UNKNOWN, dtype=grid.dtype)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            gx = x + dx
            gy = y + dy
            if 0 <= gx < size and 0 <= gy < size:
                if _visible(grid, x, y, gx, gy):
                    local[dy + radius, dx + radius] = grid[gy, gx]
    return local


def _visible(grid: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> bool:
    # Bresenham-style line of sight; walls and doors block visibility.
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    if dx == 0 and dy == 0:
        return True
    err = dx - dy
    while not (x == x1 and y == y1):
        if (x, y) != (x0, y0) and grid[y, x] in (TILE_WALL, TILE_DOOR):
            return False
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    return True


def visible_mask(grid: np.ndarray, agent_positions: list[tuple[int, int]], radius: int) -> np.ndarray:
    size = grid.shape[0]
    mask = np.zeros((size, size), dtype=bool)
    for (x, y) in agent_positions:
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                gx, gy = x + dx, y + dy
                if 0 <= gx < size and 0 <= gy < size:
                    if _visible(grid, x, y, gx, gy):
                        mask[gy, gx] = True
    return mask
