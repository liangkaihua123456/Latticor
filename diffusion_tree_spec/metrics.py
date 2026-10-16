"""Metrics containers and aggregation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional


@dataclass
class RunMetrics:
    mode: str
    batch_size: int = 1
    num_tokens: int = 0
    wall_ms: float = 0.0
    draft_ms_sum: float = 0.0
    verify_ms_sum: float = 0.0
    steps: int = 0
    accepted_lens: List[float] = field(default_factory=list)
    tree_nodes_list: List[int] = field(default_factory=list)
    nfe_list: List[int] = field(default_factory=list)
    decisions: List[str] = field(default_factory=list)
    text: str = ""

    @property
    def per_token_latency_ms(self) -> float:
        if self.num_tokens <= 0:
            return float("inf")
        return self.wall_ms / self.num_tokens

    @property
    def avg_accepted_len(self) -> float:
        if not self.accepted_lens:
            return 0.0
        return sum(self.accepted_lens) / len(self.accepted_lens)

    @property
    def avg_tree_nodes(self) -> float:
        if not self.tree_nodes_list:
            return 0.0
        return sum(self.tree_nodes_list) / len(self.tree_nodes_list)

    @property
    def avg_nfe(self) -> float:
        if not self.nfe_list:
            return 0.0
        return sum(self.nfe_list) / len(self.nfe_list)

    def summary(self) -> Dict[str, float]:
        return {
            "mode": self.mode,  # type: ignore[dict-item]
            "batch_size": self.batch_size,
            "num_tokens": self.num_tokens,
            "per_token_latency_ms": self.per_token_latency_ms,
            "avg_accepted_len": self.avg_accepted_len,
            "draft_ms_sum": self.draft_ms_sum,
            "verify_ms_sum": self.verify_ms_sum,
            "avg_tree_nodes": self.avg_tree_nodes,
            "avg_nfe": self.avg_nfe,
            "steps": self.steps,
            "wall_ms": self.wall_ms,
        }


def speedup(baseline_pt: float, other_pt: float) -> float:
    if other_pt <= 0:
        return 0.0
    return baseline_pt / other_pt
