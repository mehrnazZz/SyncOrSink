from __future__ import annotations

try:
    import pygame
except Exception as exc:  # pragma: no cover
    raise ImportError("pygame is required for human rendering. Install with `pip install pygame`.") from exc

from .sprites import SpriteAtlas


class SyncOrSinkPygameRenderer:
    def __init__(
        self,
        map_size: int,
        tile_size: int = 28,
        fps: int = 12,
        show_legend: bool = True,
        show_hud: bool = True,
        fancy: bool = True,
        use_sprites: bool = True,
        god_view: bool = False,
        split_view: bool = False,
        style: str = "arcade_flat",
    ):
        pygame.init()
        self.map_size = map_size
        self.tile_size = tile_size
        self.fps = fps
        self.show_legend = show_legend
        self.show_hud = show_hud
        self.fancy = fancy
        self.god_view = god_view
        self.split_view = split_view
        self.style = style
        if style == "arcade_flat":
            use_sprites = False
        self.use_sprites = use_sprites
        self.clock = pygame.time.Clock()
        self.panel_w = 240 if show_legend else 0

        view_cols = 2 if split_view else 1
        w = map_size * tile_size * view_cols + self.panel_w
        h = map_size * tile_size + (70 if show_hud else 0)
        self.screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption("SyncOrSink")
        self.font = pygame.font.SysFont("Arial", 16, bold=True)
        self.small = pygame.font.SysFont("Arial", 12, bold=True)

        # Flat-arcade palette (high contrast)
        self.colors = {
            "bg": (16, 16, 20),
            "grid": (35, 35, 42),
            "wall": (60, 60, 72),
            "empty": (28, 28, 34),
            "resource": (0, 180, 140),
            "station": (130, 90, 240),
            "node": (220, 170, 0),
            "clue": (70, 150, 255),
            "target": (240, 80, 80),
            "water": (0, 130, 220),
            "beacon": (255, 140, 20),
            "door": (140, 100, 60),
            "agent": (245, 245, 245),
            "fog": (0, 0, 0, 180),
            "outline": (10, 10, 12),
            "panel": (22, 22, 28),
            "accent": (255, 220, 120),
        }

        self.atlas = SpriteAtlas(self.tile_size, self.colors) if use_sprites else None

    def render(
        self,
        grid,
        agent_positions,
        info_text: str = "",
        fov_radius: int | None = None,
        visible_mask=None,
        inventories=None,
        resource_types=None,
        node_energy=None,
        node_types=None,
        decoys=None,
        scenario: str | None = None,
    ):
        pygame.event.pump()
        self._draw_scene(
            grid,
            agent_positions,
            info_text,
            fov_radius,
            visible_mask,
            inventories,
            resource_types,
            node_energy,
            node_types,
            decoys,
            scenario,
        )
        pygame.display.flip()
        self.clock.tick(self.fps)

    def render_rgb(
        self,
        grid,
        agent_positions,
        info_text: str = "",
        fov_radius: int | None = None,
        visible_mask=None,
        inventories=None,
        resource_types=None,
        node_energy=None,
        node_types=None,
        decoys=None,
        scenario: str | None = None,
    ):
        pygame.event.pump()
        self._draw_scene(
            grid,
            agent_positions,
            info_text,
            fov_radius,
            visible_mask,
            inventories,
            resource_types,
            node_energy,
            node_types,
            decoys,
            scenario,
        )
        surface = pygame.display.get_surface()
        return pygame.surfarray.array3d(surface).swapaxes(0, 1)

    def _draw_scene(
        self,
        grid,
        agent_positions,
        info_text: str = "",
        fov_radius: int | None = None,
        visible_mask=None,
        inventories=None,
        resource_types=None,
        node_energy=None,
        node_types=None,
        decoys=None,
        scenario: str | None = None,
    ):
        self.screen.fill(self.colors["bg"])

        if self.split_view:
            self._draw_grid(grid, x_offset=0)
            if fov_radius is not None:
                self._draw_fov(agent_positions, fov_radius, x_offset=0)
            if visible_mask is not None:
                self._draw_fog(visible_mask, x_offset=0)
            self._draw_tile_overlays(resource_types, node_energy, node_types, decoys, x_offset=0)
            self._draw_agents(agent_positions, x_offset=0)

            x_off = self.map_size * self.tile_size
            self._draw_grid(grid, x_offset=x_off)
            if fov_radius is not None:
                self._draw_fov(agent_positions, fov_radius, x_offset=x_off)
            self._draw_tile_overlays(resource_types, node_energy, node_types, decoys, x_offset=x_off)
            self._draw_agents(agent_positions, x_offset=x_off)
        else:
            self._draw_grid(grid, x_offset=0)
            if fov_radius is not None:
                self._draw_fov(agent_positions, fov_radius, x_offset=0)
            if visible_mask is not None and not self.god_view:
                self._draw_fog(visible_mask, x_offset=0)
            self._draw_tile_overlays(resource_types, node_energy, node_types, decoys, x_offset=0)
            self._draw_agents(agent_positions, x_offset=0)

        if self.show_hud:
            self._draw_footer(info_text, inventories, scenario)
        if self.show_legend:
            self._draw_legend()

    def _draw_grid(self, grid, x_offset: int = 0):
        size = self.map_size
        for y in range(size):
            for x in range(size):
                tile = int(grid[y, x])
                rect = pygame.Rect(
                    x * self.tile_size + x_offset,
                    y * self.tile_size,
                    self.tile_size,
                    self.tile_size,
                )
                if self.atlas is not None:
                    self._blit_tile(tile, rect)
                else:
                    self._draw_tile_flat(tile, rect)

    def _draw_tile_flat(self, tile: int, rect: pygame.Rect):
        color = self._tile_color(tile)
        pygame.draw.rect(self.screen, self.colors["outline"], rect)
        inner = rect.inflate(-2, -2)
        pygame.draw.rect(self.screen, color, inner)
        if self.fancy:
            self._draw_tile_icon(tile, inner)

    def _blit_tile(self, tile: int, rect: pygame.Rect):
        name = self._tile_name(tile)
        self.screen.blit(self.atlas.tile(name), rect)
        pygame.draw.rect(self.screen, self.colors["grid"], rect, 1)

    def _tile_name(self, tile: int) -> str:
        return {
            1: "wall",
            2: "resource",
            3: "station",
            4: "node",
            5: "clue",
            6: "target",
            7: "water",
            8: "beacon",
            9: "door",
        }.get(tile, "empty")

    def _draw_tile_icon(self, tile: int, rect: pygame.Rect):
        cx = rect.x + rect.width // 2
        cy = rect.y + rect.height // 2
        if tile == 2:  # resource
            points = [(cx, cy - 6), (cx + 6, cy), (cx, cy + 6), (cx - 6, cy)]
            pygame.draw.polygon(self.screen, (245, 245, 245), points, 0)
        elif tile == 3:  # station
            pygame.draw.rect(self.screen, (245, 245, 245), (cx - 6, cy - 6, 12, 12), 2)
        elif tile == 4:  # node
            pygame.draw.rect(self.screen, (40, 40, 40), (cx - 6, cy - 4, 12, 8))
            pygame.draw.rect(self.screen, (40, 40, 40), (cx + 6, cy - 2, 2, 4))
        elif tile == 5:  # clue
            pygame.draw.circle(self.screen, (245, 245, 245), (cx, cy), 5, 2)
        elif tile == 6:  # target
            pygame.draw.circle(self.screen, (245, 245, 245), (cx, cy), 6, 1)
            pygame.draw.circle(self.screen, (245, 245, 245), (cx, cy), 2, 0)
        elif tile == 7:  # water
            pygame.draw.arc(self.screen, (220, 220, 255), (cx - 6, cy - 6, 12, 12), 0, 3.14, 2)
        elif tile == 8:  # beacon
            pygame.draw.polygon(self.screen, (255, 240, 200), [(cx, cy - 6), (cx + 6, cy + 6), (cx - 6, cy + 6)])
        elif tile == 9:  # door
            pygame.draw.rect(self.screen, (230, 200, 140), (cx - 3, cy - 6, 6, 12), 0)

    def _draw_agents(self, agent_positions, x_offset: int = 0):
        for agent_id, (x, y) in enumerate(agent_positions):
            cx = x * self.tile_size + self.tile_size // 2 + x_offset
            cy = y * self.tile_size + self.tile_size // 2
            radius = max(5, self.tile_size // 3)
            pygame.draw.circle(self.screen, self.colors["agent"], (cx, cy), radius)
            label = self.font.render(str(agent_id), True, (0, 0, 0))
            label_rect = label.get_rect(center=(cx, cy))
            self.screen.blit(label, label_rect)

    def _draw_fov(self, agent_positions, fov_radius: int, x_offset: int = 0):
        colors = [
            (0, 200, 255),
            (255, 180, 0),
            (180, 255, 0),
            (255, 0, 200),
            (0, 255, 140),
        ]
        for agent_id, (x, y) in enumerate(agent_positions):
            color = colors[agent_id % len(colors)]
            size = (fov_radius * 2 + 1) * self.tile_size
            rect = pygame.Rect(
                (x - fov_radius) * self.tile_size + x_offset,
                (y - fov_radius) * self.tile_size,
                size,
                size,
            )
            pygame.draw.rect(self.screen, color, rect, 2)

    def _draw_fog(self, visible_mask, x_offset: int = 0):
        size = self.map_size
        overlay = pygame.Surface((size * self.tile_size, size * self.tile_size), pygame.SRCALPHA)
        overlay.fill(self.colors["fog"])
        for y in range(size):
            for x in range(size):
                if visible_mask[y, x]:
                    rect = pygame.Rect(
                        x * self.tile_size,
                        y * self.tile_size,
                        self.tile_size,
                        self.tile_size,
                    )
                    overlay.fill((0, 0, 0, 0), rect)
        self.screen.blit(overlay, (x_offset, 0))

    def _draw_footer(self, info_text: str, inventories=None, scenario: str | None = None):
        view_cols = 2 if self.split_view else 1
        w = self.map_size * self.tile_size * view_cols + self.panel_w
        rect = pygame.Rect(0, self.map_size * self.tile_size, w, 70)
        pygame.draw.rect(self.screen, self.colors["panel"], rect)
        text = self.font.render(info_text, True, (220, 220, 220))
        self.screen.blit(text, (8, self.map_size * self.tile_size + 6))
        if inventories is not None:
            inv_text = " | ".join([f"A{i}: {inv}" for i, inv in enumerate(inventories)])
            inv = self.small.render(f"inventories: {inv_text}", True, (180, 180, 180))
            self.screen.blit(inv, (8, self.map_size * self.tile_size + 28))
        if scenario:
            scen = self.small.render(f"scenario: {scenario}", True, (160, 160, 160))
            self.screen.blit(scen, (8, self.map_size * self.tile_size + 46))
        help_text = self.small.render("WASD/arrows move | SPACE interact | E pickup | Q drop | P pause", True, (140, 140, 140))
        self.screen.blit(help_text, (8, self.map_size * self.tile_size + 58))

    def _tile_color(self, tile: int):
        if tile == 1:
            return self.colors["wall"]
        if tile == 2:
            return self.colors["resource"]
        if tile == 3:
            return self.colors["station"]
        if tile == 4:
            return self.colors["node"]
        if tile == 5:
            return self.colors["clue"]
        if tile == 6:
            return self.colors["target"]
        if tile == 7:
            return self.colors["water"]
        if tile == 8:
            return self.colors["beacon"]
        if tile == 9:
            return self.colors["door"]
        return self.colors["empty"]

    def _draw_tile_overlays(self, resource_types, node_energy, node_types, decoys, x_offset: int = 0):
        if resource_types:
            for (x, y), rtype in resource_types.items():
                if 0 <= x < self.map_size and 0 <= y < self.map_size:
                    label = self.small.render(str(rtype), True, (255, 255, 255))
                    self.screen.blit(label, (x * self.tile_size + 2 + x_offset, y * self.tile_size + 2))
        if node_energy and node_types:
            for (x, y), energy in node_energy.items():
                ntype = node_types.get((x, y), 0)
                bar_w = self.tile_size - 4
                bar_h = 4
                filled = max(0, min(bar_w, int(bar_w * (energy / 10.0))))
                base_x = x * self.tile_size + 2 + x_offset
                base_y = y * self.tile_size + self.tile_size - 6
                pygame.draw.rect(self.screen, (40, 40, 40), (base_x, base_y, bar_w, bar_h))
                pygame.draw.rect(self.screen, (0, 200, 80), (base_x, base_y, filled, bar_h))
                label = self.small.render(str(ntype), True, (0, 0, 0))
                self.screen.blit(label, (x * self.tile_size + 2 + x_offset, y * self.tile_size + 2))
        if decoys:
            for (x, y) in decoys:
                cx = x * self.tile_size + self.tile_size // 2 + x_offset
                cy = y * self.tile_size + self.tile_size // 2
                pygame.draw.line(self.screen, (255, 80, 80), (cx - 6, cy - 6), (cx + 6, cy + 6), 2)
                pygame.draw.line(self.screen, (255, 80, 80), (cx - 6, cy + 6), (cx + 6, cy - 6), 2)

    def _draw_legend(self):
        view_cols = 2 if self.split_view else 1
        x0 = self.map_size * self.tile_size * view_cols + 10
        y0 = 10
        items = [
            ("Wall", self.colors["wall"]),
            ("Empty", self.colors["empty"]),
            ("Resource", self.colors["resource"]),
            ("Station", self.colors["station"]),
            ("Node", self.colors["node"]),
            ("Clue", self.colors["clue"]),
            ("Target", self.colors["target"]),
            ("Water", self.colors["water"]),
            ("Beacon", self.colors["beacon"]),
            ("Door", self.colors["door"]),
        ]
        title = self.font.render("Legend", True, (220, 220, 220))
        self.screen.blit(title, (x0, y0))
        y = y0 + 24
        for name, color in items:
            rect = pygame.Rect(x0, y, 12, 12)
            pygame.draw.rect(self.screen, color, rect)
            if self.atlas is not None:
                tile = self.atlas.tile(name.lower())
                self.screen.blit(pygame.transform.scale(tile, (12, 12)), rect)
            label = self.small.render(name, True, (200, 200, 200))
            self.screen.blit(label, (x0 + 18, y - 2))
            y += 18

    def close(self):
        pygame.quit()
