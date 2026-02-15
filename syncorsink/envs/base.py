from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # fallback for environments without gymnasium
    import gym
    from gym import spaces

from .communication import Message, count_tokens
from .maps import (
    TILE_EMPTY,
    TILE_RESOURCE,
    TILE_STATION,
    TILE_NODE,
    TILE_CLUE,
    TILE_TARGET,
    TILE_WATER,
    TILE_BEACON,
    TILE_DOOR,
    TILE_UNKNOWN,
)
from .observation import extract_local, visible_mask
from .scenarios import SCENARIOS
from .utils import FOV_PRESETS, get_rng


@dataclass
class SyncOrSinkConfig:
    scenario: str = "pipeline_assembly"
    map_size: int = 16
    num_agents: int = 3
    fov_preset: str = "medium"
    comm_mode: str = "tokens"  # "tokens" or "text"
    comm_token_limit: int = 24
    max_messages: int = 8
    token_vocab_size: int = 256
    max_steps: int = 300
    comm_radius: int | None = None
    comm_cost: float = 0.01
    comm_len_cost: float = 0.0
    # map / rendering options
    use_rooms: bool = True
    use_doors: bool = True
    enable_fog_of_war: bool = True
    # signal_hunt options
    signal_decoy_count: int | None = None
    decoy_penalty: float = 0.5
    scan_window: int = 3
    # pipeline shaping
    pipeline_shaping: bool = False
    pipeline_shaping_scale: float = 0.01
    # energy grid shaping
    energy_shaping: bool = False
    energy_shaping_scale: float = 0.01
    # signal hunt shaping
    signal_shaping: bool = False
    signal_shaping_scale: float = 0.01
    # energy grid difficulty presets
    energy_preset: str = "hard"  # "easy" or "hard"
    # deterministic map control
    map_seed: int | None = None
    map_variant: int = 0
    split: str | None = None
    # benchmark track
    track: str = "dtde"  # "dtde" or "ctde"
    # rendering options
    render_god_view: bool = False
    render_split_view: bool = False
    render_style: str = "arcade_flat"  # "arcade_flat" or "sprite"
    obs_onehot: bool = False


class SyncOrSinkEnv(gym.Env):
    metadata = {"render_modes": ["ansi", "human", "rgb_array"]}

    ACTION_UP = 0
    ACTION_DOWN = 1
    ACTION_LEFT = 2
    ACTION_RIGHT = 3
    ACTION_STAY = 4
    ACTION_INTERACT = 5
    ACTION_PICKUP = 6
    ACTION_DROP = 7

    def __init__(self, config: SyncOrSinkConfig | None = None, render_mode: str | None = None):
        super().__init__()
        self.config = config or SyncOrSinkConfig()
        self.render_mode = render_mode
        self._pg_renderer = None
        if self.config.fov_preset not in FOV_PRESETS:
            raise ValueError(f"Unknown fov preset: {self.config.fov_preset}")
        if self.config.scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {self.config.scenario}")

        self.num_agents = self.config.num_agents
        self.map_size = self.config.map_size
        self.fov_radius = FOV_PRESETS[self.config.fov_preset].radius
        self.comm_mode = self.config.comm_mode
        self.comm_token_limit = self.config.comm_token_limit
        self.max_messages = self.config.max_messages
        self.token_vocab_size = self.config.token_vocab_size
        self.max_steps = self.config.max_steps
        self.comm_radius = self.config.comm_radius
        self.obs_onehot = self.config.obs_onehot

        self.scenario = SCENARIOS[self.config.scenario]
        self.rng = get_rng(None)
        self.steps = 0

        # rewards
        self.reward_stage = 1.0
        self.reward_complete = 10.0
        self.reward_fail = 5.0
        self.energy_start = 6
        self.energy_refill = 4
        self.comm_cost = self.config.comm_cost

        self.grid = None
        self.meta = {}
        self.agent_positions: list[tuple[int, int]] = []
        self.inventories: list[int] = []
        self.scenario_state = None
        self.inboxes: list[list[Message]] = []

        self.action_space = spaces.Dict({
            "action": spaces.Discrete(8),
            "message_tokens": spaces.Box(
                low=-1,
                high=self.token_vocab_size - 1,
                shape=(self.comm_token_limit,),
                dtype=np.int16,
            ),
            "message_tokens_aux": spaces.Box(
                low=-1,
                high=self.token_vocab_size - 1,
                shape=(self.comm_token_limit,),
                dtype=np.int16,
            ),
        })

        local_shape = (self.fov_radius * 2 + 1, self.fov_radius * 2 + 1)
        self._tile_channels = max(TILES.values()) + 1
        if self.obs_onehot:
            local_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self._tile_channels, local_shape[0], local_shape[1]),
                dtype=np.float32,
            )
        else:
            local_space = spaces.Box(
                low=0,
                high=12,
                shape=local_shape,
                dtype=np.int16,
            )
        self.observation_space = spaces.Dict({
            "local_grid": local_space,
            "inventory": spaces.Box(low=0, high=10, shape=(1,), dtype=np.int16),
            "self_pos": spaces.Box(low=0, high=self.map_size - 1, shape=(2,), dtype=np.int16),
            "local_resource_types": spaces.Box(
                low=0, high=10, shape=local_shape, dtype=np.int16
            ),
            "local_node_types": spaces.Box(
                low=0, high=10, shape=local_shape, dtype=np.int16
            ),
            "local_node_energy": spaces.Box(
                low=0, high=50, shape=local_shape, dtype=np.int16
            ),
            "messages_tokens": spaces.Box(
                low=-1,
                high=self.token_vocab_size - 1,
                shape=(self.max_messages, self.comm_token_limit),
                dtype=np.int16,
            ),
            "message_from": spaces.Box(low=-1, high=self.num_agents - 1, shape=(self.max_messages,), dtype=np.int16),
            "goal_hint": spaces.Box(low=-1, high=1024, shape=(16,), dtype=np.int16),
        })

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self.rng = get_rng(seed)
        self.steps = 0
        map_seed = self.config.map_seed if self.config.map_seed is not None else seed
        if self.config.split is not None and map_seed is None:
            from syncorsink.eval.splits import seed_for_variant
            map_seed = seed_for_variant(self.config.split, self.config.map_variant)
        if map_seed is not None:
            map_rng = get_rng(map_seed + self.config.map_variant)
        else:
            map_rng = self.rng
        self.grid, self.meta = self.scenario.build_map(self.map_size, map_rng, self.config)
        starts = self.meta.get("agent_starts", [])
        if len(starts) < self.num_agents:
            # fallback: random empty cells
            starts = starts + [
                (int(self.rng.integers(0, self.map_size)), int(self.rng.integers(0, self.map_size)))
                for _ in range(self.num_agents - len(starts))
            ]
        self.agent_positions = starts[: self.num_agents]
        self.inventories = [0 for _ in range(self.num_agents)]
        self.inboxes = [[] for _ in range(self.num_agents)]
        self.scenario_state = self.scenario.reset(self)

        obs = self._build_observations()
        info = self._build_info()
        if self.config.track == "ctde":
            from .central_obs import build_central_obs
            info["central_obs"] = build_central_obs(self)
        return obs, info

    def step(self, actions: dict[int, Any]):
        self.steps += 1
        parsed_actions = self._parse_actions(actions)

        # movement phase
        occupied = {pos: idx for idx, pos in enumerate(self.agent_positions)}
        allow_multi = {TILE_STATION, TILE_NODE, TILE_TARGET}
        for agent_id, action in parsed_actions.items():
            if action["action"] in (self.ACTION_UP, self.ACTION_DOWN, self.ACTION_LEFT, self.ACTION_RIGHT):
                x, y = self.agent_positions[agent_id]
                nx, ny = x, y
                if action["action"] == self.ACTION_UP:
                    ny -= 1
                elif action["action"] == self.ACTION_DOWN:
                    ny += 1
                elif action["action"] == self.ACTION_LEFT:
                    nx -= 1
                elif action["action"] == self.ACTION_RIGHT:
                    nx += 1
                if 0 <= nx < self.map_size and 0 <= ny < self.map_size:
                    if self.grid[ny, nx] not in (1, TILE_DOOR):
                        if (nx, ny) in occupied and self.grid[ny, nx] not in allow_multi:
                            continue
                        occupied.pop(self.agent_positions[agent_id], None)
                        self.agent_positions[agent_id] = (nx, ny)
                        occupied[(nx, ny)] = agent_id

        # interaction phase: pickup/drop
        for agent_id, action in parsed_actions.items():
            pos = self.agent_positions[agent_id]
            if action["action"] == self.ACTION_PICKUP:
                if self.inventories[agent_id] == 0:
                    if pos in self._resource_positions():
                        resource_type = self._resource_positions()[pos]
                        self.inventories[agent_id] = resource_type
                        self._remove_resource(pos)
            elif action["action"] == self.ACTION_DROP:
                if self.inventories[agent_id] != 0 and self.grid[pos[1], pos[0]] == TILE_EMPTY:
                    self._add_resource(pos, self.inventories[agent_id])
                    self.inventories[agent_id] = 0
            elif action["action"] == self.ACTION_INTERACT and self.config.use_doors:
                # open adjacent door if present
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = pos[0] + dx, pos[1] + dy
                    if 0 <= nx < self.map_size and 0 <= ny < self.map_size:
                        if self.grid[ny, nx] == TILE_DOOR:
                            self.grid[ny, nx] = TILE_EMPTY
                            break

        # communication phase
        comm_tokens = {i: 0 for i in range(self.num_agents)}
        for agent_id, action in parsed_actions.items():
            msg_tokens = action.get("message_tokens")
            msg_tokens_aux = action.get("message_tokens_aux")
            msg_text = action.get("message_text")
            token_count = count_tokens(msg_tokens, msg_text)
            aux_count = count_tokens(msg_tokens_aux, None)
            if token_count == 0 and aux_count == 0:
                continue
            if token_count > self.comm_token_limit:
                if msg_tokens is not None:
                    msg_tokens = msg_tokens[: self.comm_token_limit]
                if msg_text is not None:
                    msg_text = " ".join(msg_text.split()[: self.comm_token_limit])
                token_count = self.comm_token_limit
            if aux_count > self.comm_token_limit:
                if msg_tokens_aux is not None:
                    msg_tokens_aux = msg_tokens_aux[: self.comm_token_limit]
                aux_count = self.comm_token_limit
            comm_tokens[agent_id] = token_count + aux_count
            messages = []
            if token_count > 0:
                messages.append(Message(sender=agent_id, tokens=msg_tokens or [], text=msg_text))
            if aux_count > 0:
                messages.append(Message(sender=agent_id, tokens=msg_tokens_aux or [], text=None))
            for recv_id in range(self.num_agents):
                if recv_id == agent_id:
                    continue
                if self.comm_radius is not None:
                    ax, ay = self.agent_positions[agent_id]
                    bx, by = self.agent_positions[recv_id]
                    if abs(ax - bx) + abs(ay - by) > self.comm_radius:
                        continue
                for message in messages:
                    self.inboxes[recv_id].append(message)
                if len(self.inboxes[recv_id]) > self.max_messages:
                    self.inboxes[recv_id] = self.inboxes[recv_id][-self.max_messages :]

        # scenario-specific step
        rewards, done, scenario_info = self.scenario.step(self, parsed_actions)

        # communication penalty
        for agent_id, count in comm_tokens.items():
            rewards[agent_id] -= self.comm_cost * count
            if self.config.comm_len_cost > 0:
                rewards[agent_id] -= self.config.comm_len_cost * count

        truncated = self.steps >= self.max_steps
        obs = self._build_observations()
        info = self._build_info()
        if self.config.track == "ctde":
            from .central_obs import build_central_obs
            info["central_obs"] = build_central_obs(self)
        info["comm_tokens"] = comm_tokens
        info.update(scenario_info)
        return obs, rewards, done, truncated, info

    def render(self):
        if self.render_mode in (None, "ansi"):
            return self._render_ansi()
        if self.render_mode == "human":
            return self._render_human()
        if self.render_mode == "rgb_array":
            return self._render_rgb()
        raise NotImplementedError("Only ansi and human render are supported.")

    def _render_ansi(self) -> str:
        glyphs = {
            TILE_EMPTY: ".",
            1: "#",
            TILE_RESOURCE: "R",
            TILE_STATION: "S",
            TILE_NODE: "N",
            TILE_CLUE: "C",
            TILE_TARGET: "T",
            TILE_WATER: "W",
            TILE_BEACON: "B",
            TILE_DOOR: "D",
            TILE_UNKNOWN: "?",
        }
        grid = [[glyphs.get(int(self.grid[y, x]), "?") for x in range(self.map_size)] for y in range(self.map_size)]
        for agent_id, (x, y) in enumerate(self.agent_positions):
            grid[y][x] = str(agent_id % 10)
        lines = ["".join(row) for row in grid]
        lines.append(f"steps={self.steps} scenario={self.config.scenario} agents={self.num_agents}")
        return "\n".join(lines)

    def _render_human(self):
        if self._pg_renderer is None:
            from .pygame_render import SyncOrSinkPygameRenderer
            self._pg_renderer = SyncOrSinkPygameRenderer(
                map_size=self.map_size,
                god_view=self.config.render_god_view,
                split_view=self.config.render_split_view,
                style=self.config.render_style,
            )
        info_text = f"steps={self.steps} scenario={self.config.scenario} agents={self.num_agents}"
        mask = None
        if self.config.enable_fog_of_war:
            mask = visible_mask(self.grid, self.agent_positions, self.fov_radius)
        resource_types = self.scenario_state.data.get("resource_types")
        node_energy = self.scenario_state.data.get("node_energy")
        node_types = self.scenario_state.data.get("node_types")
        decoys = self.scenario_state.data.get("decoys")
        self._pg_renderer.render(
            self.grid,
            self.agent_positions,
            info_text=info_text,
            fov_radius=self.fov_radius,
            visible_mask=mask,
            inventories=self.inventories,
            resource_types=resource_types,
            node_energy=node_energy,
            node_types=node_types,
            decoys=decoys,
            scenario=self.config.scenario,
        )
        return None

    def _render_rgb(self):
        if self._pg_renderer is None:
            from .pygame_render import SyncOrSinkPygameRenderer
            self._pg_renderer = SyncOrSinkPygameRenderer(map_size=self.map_size)
        info_text = f"steps={self.steps} scenario={self.config.scenario} agents={self.num_agents}"
        mask = None
        if self.config.enable_fog_of_war:
            mask = visible_mask(self.grid, self.agent_positions, self.fov_radius)
        resource_types = self.scenario_state.data.get("resource_types")
        node_energy = self.scenario_state.data.get("node_energy")
        node_types = self.scenario_state.data.get("node_types")
        decoys = self.scenario_state.data.get("decoys")
        surface = self._pg_renderer.render_rgb(
            self.grid,
            self.agent_positions,
            info_text=info_text,
            fov_radius=self.fov_radius,
            visible_mask=mask,
            inventories=self.inventories,
            resource_types=resource_types,
            node_energy=node_energy,
            node_types=node_types,
            decoys=decoys,
            scenario=self.config.scenario,
        )
        return surface

    def close(self):
        if self._pg_renderer is not None:
            self._pg_renderer.close()
            self._pg_renderer = None

    def _parse_actions(self, actions: dict[int, Any]) -> dict[int, dict]:
        parsed = {}
        for agent_id in range(self.num_agents):
            act = actions.get(agent_id, {"action": self.ACTION_STAY})
            if isinstance(act, int):
                act = {"action": int(act)}
            action_id = int(act.get("action", self.ACTION_STAY))
            msg_tokens = act.get("message_tokens")
            if msg_tokens is not None:
                msg_tokens = [int(t) for t in list(msg_tokens) if int(t) >= 0][: self.comm_token_limit]
            msg_tokens_aux = act.get("message_tokens_aux")
            if msg_tokens_aux is not None:
                msg_tokens_aux = [int(t) for t in list(msg_tokens_aux) if int(t) >= 0][: self.comm_token_limit]
            msg_text = act.get("message_text")
            parsed[agent_id] = {
                "action": action_id,
                "message_tokens": msg_tokens,
                "message_tokens_aux": msg_tokens_aux,
                "message_text": msg_text if self.comm_mode == "text" else None,
            }
        return parsed

    def _resource_positions(self) -> dict[tuple[int, int], int]:
        return self.scenario_state.data.get("resource_types", {})

    def _remove_resource(self, pos: tuple[int, int]):
        if pos in self.scenario_state.data.get("resource_types", {}):
            self.scenario_state.data["resource_types"].pop(pos, None)
        if self.grid[pos[1], pos[0]] == TILE_RESOURCE:
            self.grid[pos[1], pos[0]] = TILE_EMPTY

    def _add_resource(self, pos: tuple[int, int], resource_type: int):
        self.scenario_state.data.setdefault("resource_types", {})[pos] = resource_type
        self.grid[pos[1], pos[0]] = TILE_RESOURCE

    def _build_observations(self) -> dict[int, dict]:
        obs = {}
        resource_type_grid = np.zeros((self.map_size, self.map_size), dtype=np.int16)
        for pos, rtype in self._resource_positions().items():
            resource_type_grid[pos[1], pos[0]] = int(rtype)
        node_type_grid = np.zeros((self.map_size, self.map_size), dtype=np.int16)
        for pos, ntype in self.scenario_state.data.get("node_types", {}).items():
            node_type_grid[pos[1], pos[0]] = int(ntype)
        node_energy_grid = np.zeros((self.map_size, self.map_size), dtype=np.int16)
        for pos, energy in self.scenario_state.data.get("node_energy", {}).items():
            node_energy_grid[pos[1], pos[0]] = int(energy)
        for agent_id in range(self.num_agents):
            local = extract_local(self.grid, self.agent_positions[agent_id], self.fov_radius)
            if self.obs_onehot:
                local = self._onehot(local)
            local_resource_types = extract_local(resource_type_grid, self.agent_positions[agent_id], self.fov_radius)
            local_node_types = extract_local(node_type_grid, self.agent_positions[agent_id], self.fov_radius)
            local_node_energy = extract_local(node_energy_grid, self.agent_positions[agent_id], self.fov_radius)
            messages_tokens, message_from = self._encode_messages(self.inboxes[agent_id])
            hint_tokens = self._encode_hint(agent_id)
            obs[agent_id] = {
                "local_grid": local,
                "inventory": np.array([self.inventories[agent_id]], dtype=np.int16),
                "self_pos": np.array(self.agent_positions[agent_id], dtype=np.int16),
                "local_resource_types": local_resource_types,
                "local_node_types": local_node_types,
                "local_node_energy": local_node_energy,
                "messages_tokens": messages_tokens,
                "message_from": message_from,
                "goal_hint": hint_tokens,
            }
        return obs

    def _onehot(self, local: np.ndarray) -> np.ndarray:
        # One-hot encode tiles 0..max into shape (C, H, W)
        channels = self._tile_channels
        h, w = local.shape
        out = np.zeros((channels, h, w), dtype=np.float32)
        for c in range(channels):
            out[c] = (local == c).astype(np.float32)
        return out

    def _encode_messages(self, inbox: list[Message]):
        tokens = np.full((self.max_messages, self.comm_token_limit), -1, dtype=np.int16)
        senders = np.full((self.max_messages,), -1, dtype=np.int16)
        for idx, msg in enumerate(inbox[-self.max_messages :]):
            senders[idx] = msg.sender
            for j, t in enumerate(msg.tokens[: self.comm_token_limit]):
                tokens[idx, j] = int(t)
        return tokens, senders

    def _encode_hint(self, agent_id: int) -> np.ndarray:
        hint = []
        if self.config.scenario == "pipeline_assembly":
            stages = self.scenario_state.data.get("hints", {}).get(agent_id, [])
            for stage in stages:
                required = stage.get("required", [])
                deps = stage.get("deps", [])
                hint.extend(
                    [
                        stage["stage"],
                        stage["station"][0],
                        stage["station"][1],
                        len(required),
                        required[0] if len(required) > 0 else -1,
                        required[1] if len(required) > 1 else -1,
                        len(deps),
                        deps[0] if deps else -1,
                        1 if stage.get("sync") else 0,
                    ]
                )
        hint = hint[:16]
        padded = hint + [-1] * (16 - len(hint))
        return np.array(padded, dtype=np.int16)

    def _build_info(self) -> dict:
        info = {"messages_text": {}, "goal_hint_texts": {}}
        for agent_id in range(self.num_agents):
            info["messages_text"][agent_id] = [m.text for m in self.inboxes[agent_id] if m.text]
        if self.config.scenario == "signal_hunt":
            for agent_id in range(self.num_agents):
                info["goal_hint_texts"][agent_id] = self.scenario_state.data.get("agent_hints", {}).get(agent_id)
        return info


# Expose tiles for users
TILES = {
    "empty": TILE_EMPTY,
    "resource": TILE_RESOURCE,
    "station": TILE_STATION,
    "node": TILE_NODE,
    "clue": TILE_CLUE,
    "target": TILE_TARGET,
    "water": TILE_WATER,
    "beacon": TILE_BEACON,
    "door": TILE_DOOR,
    "unknown": TILE_UNKNOWN,
}
