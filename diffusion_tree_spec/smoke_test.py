#!/usr/bin/env python3
"""Minimal smoke test: use llama-68m as both target and draft (no 7B needed)."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from diffusion_tree_spec.engine import DiffusionTreeEngine, EngineConfig


def main() -> None:
    # Weights may live in a local snapshot without tokenizer files.
    weights = (
        "/root/autodl-tmp/hf_home/hub/models--JackFram--llama-68m/"
        "snapshots/9de84537b6aa98de634ad0fbb1608e9d6a019355"
    )
    cfg = EngineConfig(
        target_model=weights,
        draft_model=weights,
        tokenizer_name="JackFram/llama-68m",
        mode="diff_tree",
        max_new_tokens=16,
        draft_len=6,
        top_k=3,
        nfe=2,
        max_width=2,
        adaptive=True,
        dtype="float16",
        device="cuda",
    )
    print("Loading...", flush=True)
    eng = DiffusionTreeEngine(cfg)
    m = eng.generate("Hello, speculative decoding is", batch_size=1)
    print(m.summary())
    print("text:", m.text[:200])
    print("SMOKE_OK")


if __name__ == "__main__":
    main()
