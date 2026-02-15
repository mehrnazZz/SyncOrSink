from __future__ import annotations

import random

try:
    import pygame
except Exception as exc:  # pragma: no cover
    raise ImportError("pygame is required for human control. Install with `pip install pygame`.") from exc


class HumanController:
    def __init__(self, env, human_agent_id: int = 0):
        if not pygame.get_init():
            pygame.init()
        self.env = env
        self.human_agent_id = human_agent_id
        self.last_action = env.ACTION_STAY
        self.paused = False

    def collect_actions(self):
        self._poll_events()
        actions = {}
        for agent_id in range(self.env.num_agents):
            if agent_id == self.human_agent_id:
                actions[agent_id] = {"action": self.last_action, "message_tokens": []}
            else:
                actions[agent_id] = {"action": self._random_action(), "message_tokens": []}
        if self.paused:
            for agent_id in range(self.env.num_agents):
                actions[agent_id] = {"action": self.env.ACTION_STAY, "message_tokens": []}
        return actions

    def _poll_events(self):
        if not pygame.get_init():
            pygame.init()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.env.close()
                raise SystemExit(0)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w or event.key == pygame.K_UP:
                    self.last_action = self.env.ACTION_UP
                elif event.key == pygame.K_s or event.key == pygame.K_DOWN:
                    self.last_action = self.env.ACTION_DOWN
                elif event.key == pygame.K_a or event.key == pygame.K_LEFT:
                    self.last_action = self.env.ACTION_LEFT
                elif event.key == pygame.K_d or event.key == pygame.K_RIGHT:
                    self.last_action = self.env.ACTION_RIGHT
                elif event.key == pygame.K_SPACE:
                    self.last_action = self.env.ACTION_INTERACT
                elif event.key == pygame.K_e:
                    self.last_action = self.env.ACTION_PICKUP
                elif event.key == pygame.K_q:
                    self.last_action = self.env.ACTION_DROP
                elif event.key == pygame.K_p:
                    self.paused = not self.paused
                elif event.key == pygame.K_x:
                    self.last_action = self.env.ACTION_STAY

    def _random_action(self):
        return random.randint(0, 7)
