from __future__ import annotations

import torch
import torch.nn as nn

from syncorsink.models import MLPEncoder, TransformerEncoder, PolicyHead, ValueHead


class MAPPOActor(nn.Module):
    """Actor network for MAPPO with optional communication heads.

    Outputs action logits and, when comm is enabled, three additional heads:
      - send gate (Bernoulli): whether to send a message
      - token logits (Categorical per position): content of each token slot
      - length logits (Categorical): how many tokens to actually send
    """

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

    def forward(self, obs: torch.Tensor):
        h = self.encoder(obs)
        logits = self.policy(h)
        if not self.comm_enabled:
            return logits
        send_logits = self.comm_send(h)
        token_logits = self.comm_tokens(h).view(
            h.shape[0], self.comm_token_limit, self.comm_vocab_size
        )
        len_logits = self.comm_len(h)
        return logits, send_logits, token_logits, len_logits


class MAPPORecurrentActor(nn.Module):
    """Actor with LSTM memory for sequential tasks like pipeline assembly.

    The LSTM maintains hidden state across steps within an episode,
    allowing the policy to track progress (which stages are done,
    what resources have been delivered, etc.) without explicit memory
    in the observation.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        comm_enabled: bool = False,
        comm_token_limit: int = 0,
        comm_vocab_size: int = 0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.comm_enabled = comm_enabled
        self.comm_token_limit = comm_token_limit
        self.comm_vocab_size = comm_vocab_size

        self.encoder = MLPEncoder(obs_dim, hidden_dim=hidden_dim, depth=2)
        self.lstm = nn.LSTMCell(hidden_dim, hidden_dim)
        self.policy = PolicyHead(hidden_dim, action_dim)
        if comm_enabled:
            self.comm_send = nn.Linear(hidden_dim, 1)
            self.comm_tokens = nn.Linear(hidden_dim, comm_token_limit * comm_vocab_size)
            self.comm_len = nn.Linear(hidden_dim, comm_token_limit + 1)

    def init_hidden(self, batch_size: int, device: torch.device):
        """Initialize LSTM hidden state for a new episode."""
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

    def forward(self, obs: torch.Tensor, hidden: tuple[torch.Tensor, torch.Tensor]):
        """Forward pass with LSTM state.

        Args:
            obs: (B, obs_dim)
            hidden: (h, c) each (B, hidden_dim)

        Returns:
            Same as MAPPOActor but with updated hidden state as last element.
        """
        enc = self.encoder(obs)
        h, c = self.lstm(enc, hidden)
        logits = self.policy(h)
        if not self.comm_enabled:
            return logits, (h, c)
        send_logits = self.comm_send(h)
        token_logits = self.comm_tokens(h).view(
            h.shape[0], self.comm_token_limit, self.comm_vocab_size
        )
        len_logits = self.comm_len(h)
        return logits, send_logits, token_logits, len_logits, (h, c)


class MAPPOCritic(nn.Module):
    """Value network for MAPPO supporting both DTDE and CTDE modes.

    - critic_mode="local": input is a single agent's observation (DTDE).
    - critic_mode="central": input is concatenated observations of all agents (CTDE).

    The caller is responsible for passing the right input dimension and tensors.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.encoder = MLPEncoder(input_dim, hidden_dim=hidden_dim, depth=2)
        self.value = ValueHead(hidden_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h = self.encoder(obs)
        return self.value(h)


# Keep the old name as an alias for backward compatibility with checkpoints
MAPPOCentralValue = MAPPOCritic
