from __future__ import annotations

import torch


def grid_to_tokens(grid: torch.Tensor, vocab_size: int = 16) -> torch.Tensor:
    # grid: (H, W)
    flat = grid.view(-1).long()
    return torch.clamp(flat, 0, vocab_size - 1)


def tokens_to_onehot(tokens: torch.Tensor, vocab_size: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(tokens, num_classes=vocab_size).float()
