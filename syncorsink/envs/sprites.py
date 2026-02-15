from __future__ import annotations

import pygame


class SpriteAtlas:
    def __init__(self, tile_size: int, colors: dict[str, tuple[int, int, int]]):
        self.tile_size = tile_size
        self.colors = colors
        self.tiles = {}
        self._build_tiles()

    def _build_tiles(self):
        ts = self.tile_size
        self.tiles["empty"] = self._solid(self.colors["empty"])
        self.tiles["wall"] = self._wall()
        self.tiles["resource"] = self._resource()
        self.tiles["station"] = self._station()
        self.tiles["node"] = self._node()
        self.tiles["clue"] = self._clue()
        self.tiles["target"] = self._target()
        self.tiles["water"] = self._water()
        self.tiles["beacon"] = self._beacon()
        self.tiles["door"] = self._door()

    def _solid(self, color):
        surf = pygame.Surface((self.tile_size, self.tile_size))
        surf.fill(color)
        return surf

    def _wall(self):
        surf = self._solid(self.colors["wall"])
        ts = self.tile_size
        for i in range(0, ts, 4):
            pygame.draw.line(surf, (90, 90, 100), (i, 0), (i, ts), 1)
        return surf

    def _resource(self):
        surf = self._solid(self.colors["resource"])
        ts = self.tile_size
        pygame.draw.polygon(surf, (230, 230, 230), [(ts//2, 4), (ts-4, ts//2), (ts//2, ts-4), (4, ts//2)])
        return surf

    def _station(self):
        surf = self._solid(self.colors["station"])
        ts = self.tile_size
        pygame.draw.rect(surf, (220, 220, 220), (4, 4, ts-8, ts-8), 2)
        return surf

    def _node(self):
        surf = self._solid(self.colors["node"])
        ts = self.tile_size
        pygame.draw.rect(surf, (40, 40, 40), (4, ts//2-4, ts-8, 8))
        pygame.draw.rect(surf, (40, 40, 40), (ts-6, ts//2-2, 4, 4))
        return surf

    def _clue(self):
        surf = self._solid(self.colors["clue"])
        ts = self.tile_size
        pygame.draw.circle(surf, (240, 240, 240), (ts//2, ts//2), 5, 2)
        return surf

    def _target(self):
        surf = self._solid(self.colors["target"])
        ts = self.tile_size
        pygame.draw.circle(surf, (240, 240, 240), (ts//2, ts//2), 6, 1)
        pygame.draw.circle(surf, (240, 240, 240), (ts//2, ts//2), 2, 0)
        return surf

    def _water(self):
        surf = self._solid(self.colors["water"])
        ts = self.tile_size
        pygame.draw.arc(surf, (220, 220, 255), (4, 4, ts-8, ts-8), 0, 3.14, 2)
        return surf

    def _beacon(self):
        surf = self._solid(self.colors["beacon"])
        ts = self.tile_size
        pygame.draw.polygon(surf, (255, 240, 200), [(ts//2, 4), (ts-4, ts-4), (4, ts-4)])
        return surf

    def _door(self):
        surf = self._solid(self.colors["door"])
        ts = self.tile_size
        pygame.draw.rect(surf, (200, 180, 120), (ts//2-3, 4, 6, ts-8), 0)
        return surf

    def tile(self, name: str):
        return self.tiles[name]
