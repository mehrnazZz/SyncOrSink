from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List

from .spec_validate import validate_spec


@dataclass
class BenchmarkCase:
    name: str
    spec: Dict[str, Any]
    weight: float = 1.0
    tags: List[str] | None = None


@dataclass
class BenchmarkSpec:
    name: str
    cases: List[BenchmarkCase]
    version: str = "unversioned"
    description: str = ""
    metadata: Dict[str, Any] | None = None


def load_benchmark(path: str) -> BenchmarkSpec:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "name" not in data or "cases" not in data:
        raise ValueError("Benchmark spec must have name and cases")
    cases = []
    for case in data["cases"]:
        if "name" not in case or "spec" not in case:
            raise ValueError("Each case must have name and spec")
        validate_spec(case["spec"])
        cases.append(
            BenchmarkCase(
                name=case["name"],
                spec=case["spec"],
                weight=float(case.get("weight", 1.0)),
                tags=list(case.get("tags", [])),
            )
        )
    return BenchmarkSpec(
        name=data["name"],
        cases=cases,
        version=str(data.get("version", "unversioned")),
        description=str(data.get("description", "")),
        metadata=dict(data.get("metadata", {})),
    )
