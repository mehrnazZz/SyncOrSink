from __future__ import annotations

import torch
import torch.nn as nn

from syncorsink.models import MLPEncoder, TransformerEncoder, PolicyHead, ValueHead


class MAPPOActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        backbone: str = "mlp",
        comm_enabled: bool = False,
        comm_token_limit: int = 0,
        comm_vocab_size: int = 0,
    ):
        super().__init__()
        self.backbone = backbone
        self.comm_enabled = comm_enabled
        self.comm_token_limit = comm_token_limit
        self.comm_vocab_size = comm_vocab_size
        if backbone == "transformer":
            self.encoder = TransformerEncoder(token_dim=obs_dim, hidden_dim=hidden_dim)
        else:
            self.encoder = MLPEncoder(obs_dim, hidden_dim=hidden_dim, depth=2)
        self.policy = PolicyHead(hidden_dim, action_dim)
        if comm_enabled:
            self.comm_send = nn.Linear(hidden_dim, 1)
            self.comm_tokens = nn.Linear(hidden_dim, comm_token_limit * comm_vocab_size)
            self.comm_len = nn.Linear(hidden_dim, comm_token_limit + 1)

    def forward(self, obs):
        h = self.encoder(obs)
        logits = self.policy(h)
        if not self.comm_enabled:
            return logits
        send_logits = self.comm_send(h)
        token_logits = self.comm_tokens(h).view(-1, self.comm_token_limit, self.comm_vocab_size)
        len_logits = self.comm_len(h)
        return logits, send_logits, token_logits, len_logits


class MAPPOCentralValue(nn.Module):
    def __init__(self, joint_obs_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.encoder = MLPEncoder(joint_obs_dim, hidden_dim=hidden_dim, depth=2)
        self.value = ValueHead(hidden_dim)

    def forward(self, obs):
        h = self.encoder(obs)
        return self.value(h)
