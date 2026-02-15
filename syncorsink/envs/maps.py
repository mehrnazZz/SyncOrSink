from __future__ import annotations

from dataclasses import dataclass
import numpy as np


TILE_EMPTY = 0
TILE_WALL = 1
TILE_RESOURCE = 2
TILE_STATION = 3
TILE_NODE = 4
TILE_CLUE = 5
TILE_TARGET = 6
TILE_WATER = 7
TILE_BEACON = 8
TILE_DOOR = 9
TILE_UNKNOWN = 10


@dataclass
class MapSpec:
    size: int
    obstacle_frac: float = 0.08


def generate_base_map(spec: MapSpec, rng: np.random.Generator) -> np.ndarray:
    size = spec.size
    grid = np.full((size, size), TILE_EMPTY, dtype=np.int16)
    wall_count = int(size * size * spec.obstacle_frac)
    for _ in range(wall_count):
        x = rng.integers(0, size)
        y = rng.integers(0, size)
        grid[y, x] = TILE_WALL
    return grid


def generate_room_map(spec: MapSpec, rng: np.random.Generator, room_attempts: int = 30) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    size = spec.size
    grid = np.full((size, size), TILE_WALL, dtype=np.int16)
    rooms: list[tuple[int, int, int, int]] = []

    for _ in range(room_attempts):
        w = int(rng.integers(3, max(4, size // 3)))
        h = int(rng.integers(3, max(4, size // 3)))
        x = int(rng.integers(1, size - w - 1))
        y = int(rng.integers(1, size - h - 1))
        rect = (x, y, w, h)
        # check overlap
        overlap = False
        for rx, ry, rw, rh in rooms:
            if x < rx + rw + 1 and x + w + 1 > rx and y < ry + rh + 1 and y + h + 1 > ry:
                overlap = True
                break
        if overlap:
            continue
        for yy in range(y, y + h):
            for xx in range(x, x + w):
                grid[yy, xx] = TILE_EMPTY
        rooms.append(rect)

    # connect rooms with corridors
    for i in range(1, len(rooms)):
        x1, y1, w1, h1 = rooms[i - 1]
        x2, y2, w2, h2 = rooms[i]
        cx1, cy1 = x1 + w1 // 2, y1 + h1 // 2
        cx2, cy2 = x2 + w2 // 2, y2 + h2 // 2
        if rng.random() < 0.5:
            _carve_h_corridor(grid, cx1, cx2, cy1)
            _carve_v_corridor(grid, cy1, cy2, cx2)
        else:
            _carve_v_corridor(grid, cy1, cy2, cx1)
            _carve_h_corridor(grid, cx1, cx2, cy2)

    # add doors where corridors meet rooms
    _place_doors_from_corridors(grid, rng)

    return grid, rooms


def _carve_h_corridor(grid: np.ndarray, x1: int, x2: int, y: int):
    if x2 < x1:
        x1, x2 = x2, x1
    for x in range(x1, x2 + 1):
        if grid[y, x] == TILE_WALL:
            grid[y, x] = TILE_EMPTY


def _carve_v_corridor(grid: np.ndarray, y1: int, y2: int, x: int):
    if y2 < y1:
        y1, y2 = y2, y1
    for y in range(y1, y2 + 1):
        if grid[y, x] == TILE_WALL:
            grid[y, x] = TILE_EMPTY


def _place_doors_from_corridors(grid: np.ndarray, rng: np.random.Generator, prob: float = 0.3):
    size = grid.shape[0]
    candidates = []
    for y in range(1, size - 1):
        for x in range(1, size - 1):
            if grid[y, x] != TILE_WALL:
                continue
            # horizontal door: empty left/right, walls up/down
            if grid[y, x - 1] == TILE_EMPTY and grid[y, x + 1] == TILE_EMPTY and grid[y - 1, x] == TILE_WALL and grid[y + 1, x] == TILE_WALL:
                candidates.append((x, y))
            # vertical door: empty up/down, walls left/right
            if grid[y - 1, x] == TILE_EMPTY and grid[y + 1, x] == TILE_EMPTY and grid[y, x - 1] == TILE_WALL and grid[y, x + 1] == TILE_WALL:
                candidates.append((x, y))
    for (x, y) in candidates:
        if rng.random() < prob:
            grid[y, x] = TILE_DOOR


def place_tiles_with_constraints(
    grid: np.ndarray,
    rng: np.random.Generator,
    tile: int,
    count: int,
    min_dist: int = 0,
    avoid: list[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    size = grid.shape[0]
    positions = []
    avoid = avoid or []
    tries = 0
    while len(positions) < count and tries < count * 500:
        x = int(rng.integers(0, size))
        y = int(rng.integers(0, size))
        if grid[y, x] != TILE_EMPTY:
            tries += 1
            continue
        if any(abs(x - ax) + abs(y - ay) < min_dist for (ax, ay) in avoid):
            tries += 1
            continue
        grid[y, x] = tile
        positions.append((x, y))
        tries += 1
    return positions


def place_random_positions(grid: np.ndarray, rng: np.random.Generator, count: int) -> list[tuple[int, int]]:
    size = grid.shape[0]
    positions = []
    tries = 0
    while len(positions) < count and tries < count * 200:
        x = int(rng.integers(0, size))
        y = int(rng.integers(0, size))
        if grid[y, x] == TILE_EMPTY:
            grid[y, x] = -1  # temporary reserved
            positions.append((x, y))
        tries += 1
    for (x, y) in positions:
        grid[y, x] = TILE_EMPTY
    return positions


def place_tiles(grid: np.ndarray, rng: np.random.Generator, tile: int, count: int) -> list[tuple[int, int]]:
    size = grid.shape[0]
    positions = []
    tries = 0
    while len(positions) < count and tries < count * 200:
        x = int(rng.integers(0, size))
        y = int(rng.integers(0, size))
        if grid[y, x] == TILE_EMPTY:
            grid[y, x] = tile
            positions.append((x, y))
        tries += 1
    return positions


def largest_component_positions(grid: np.ndarray) -> list[tuple[int, int]]:
    size = grid.shape[0]
    seen = set()
    best = []
    for y in range(size):
        for x in range(size):
            if grid[y, x] != TILE_EMPTY or (x, y) in seen:
                continue
            # BFS component
            stack = [(x, y)]
            comp = []
            seen.add((x, y))
            while stack:
                cx, cy = stack.pop()
                comp.append((cx, cy))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < size and 0 <= ny < size and (nx, ny) not in seen:
                        if grid[ny, nx] == TILE_EMPTY:
                            seen.add((nx, ny))
                            stack.append((nx, ny))
            if len(comp) > len(best):
                best = comp
    return best


def place_tiles_in_positions(
    grid: np.ndarray,
    rng: np.random.Generator,
    tile: int,
    count: int,
    positions: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    positions = positions.copy()
    rng.shuffle(positions)
    placed = []
    for pos in positions:
        if len(placed) >= count:
            break
        x, y = pos
        if grid[y, x] != TILE_EMPTY:
            continue
        grid[y, x] = tile
        placed.append((x, y))
    return placed
