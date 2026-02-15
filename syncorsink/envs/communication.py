from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Message:
    sender: int
    tokens: list[int]
    text: str | None


def count_tokens(tokens: list[int] | None, text: str | None) -> int:
    if tokens is not None:
        return len([t for t in tokens if t >= 0])
    if text is None:
        return 0
    # Simple whitespace token count for text mode.
    return len([t for t in text.strip().split() if t])
