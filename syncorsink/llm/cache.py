from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PromptCache:
    path: Path

    def __post_init__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("{}", encoding="utf-8")

    def _load(self) -> dict:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _save(self, data: dict):
        self.path.write_text(json.dumps(data), encoding="utf-8")

    def _key(self, prompt: str) -> str:
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    def get(self, prompt: str):
        data = self._load()
        return data.get(self._key(prompt))

    def set(self, prompt: str, response: str):
        data = self._load()
        data[self._key(prompt)] = response
        self._save(data)
