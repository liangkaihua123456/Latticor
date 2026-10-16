"""Innovation B: Timestep-as-Tree adaptive NFE / width controller."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional


@dataclass
class AdaptiveDecision:
    nfe: int
    top_k: int
    draft_len: int
    max_width: int
    reason: str


class TimestepAsTreeController:
    """Map denoising compute to tree depth/width using online feedback.

    Signals
    -------
    - recent accepted length (alpha)
    - optional batch size / load proxy
    - optional lattice entropy (caller may pass)
    """

    def __init__(
        self,
        *,
        min_nfe: int = 1,
        max_nfe: int = 6,
        min_width: int = 1,
        max_width: int = 5,
        base_draft_len: int = 8,
        target_alpha: float = 3.0,
        window: int = 16,
        enabled: bool = True,
    ) -> None:
        self.min_nfe = min_nfe
        self.max_nfe = max_nfe
        self.min_width = min_width
        self.max_width = max_width
        self.base_draft_len = base_draft_len
        self.target_alpha = target_alpha
        self.enabled = enabled
        self._alpha_hist: Deque[float] = deque(maxlen=window)
        self._cur_nfe = max(2, (min_nfe + max_nfe) // 2)
        self._cur_width = max(2, (min_width + max_width) // 2)

    def observe(self, accepted_len: float, mean_entropy: Optional[float] = None) -> None:
        self._alpha_hist.append(float(accepted_len))
        del mean_entropy  # reserved for future use

    def decide(self, *, batch_size: int = 1) -> AdaptiveDecision:
        if not self.enabled:
            return AdaptiveDecision(
                nfe=self.max_nfe,
                top_k=self.max_width,
                draft_len=self.base_draft_len,
                max_width=self.max_width,
                reason="fixed",
            )

        avg_alpha = (
            sum(self._alpha_hist) / len(self._alpha_hist) if self._alpha_hist else self.target_alpha
        )

        # Load-aware: larger batch → smaller trees (less spare compute).
        load_penalty = 0
        if batch_size >= 8:
            load_penalty = 2
        elif batch_size >= 4:
            load_penalty = 1

        reason_parts = []
        if avg_alpha < self.target_alpha - 0.5:
            self._cur_nfe = min(self.max_nfe, self._cur_nfe + 1)
            self._cur_width = min(self.max_width, self._cur_width + 1)
            reason_parts.append("raise_for_alpha")
        elif avg_alpha > self.target_alpha + 0.75:
            self._cur_nfe = max(self.min_nfe, self._cur_nfe - 1)
            self._cur_width = max(self.min_width, self._cur_width - 1)
            reason_parts.append("lower_for_alpha")
        else:
            reason_parts.append("hold")

        nfe = max(self.min_nfe, self._cur_nfe - load_penalty)
        width = max(self.min_width, self._cur_width - load_penalty)
        # Shorter draft when load is high.
        draft_len = self.base_draft_len if batch_size < 4 else max(4, self.base_draft_len - 2)
        if batch_size >= 8:
            draft_len = max(4, self.base_draft_len - 4)

        if load_penalty:
            reason_parts.append(f"load_bs{batch_size}")

        return AdaptiveDecision(
            nfe=nfe,
            top_k=width,
            draft_len=draft_len,
            max_width=width,
            reason="+".join(reason_parts),
        )
