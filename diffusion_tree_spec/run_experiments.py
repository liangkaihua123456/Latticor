#!/usr/bin/env python3
"""Single-GPU SmartCom experiments for Diffusion Lattice → Tree (A+B).

Usage (inside conda flexflow)::

    source /path/to/activate_specinfer.sh
    python -m diffusion_tree_spec.run_experiments --quick
    python -m diffusion_tree_spec.run_experiments --full
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

# Allow running as script or module.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from diffusion_tree_spec.engine import DiffusionTreeEngine, EngineConfig
from diffusion_tree_spec.metrics import RunMetrics, speedup


DEFAULT_PROMPTS = [
    "Explain the idea of speculative decoding for large language models in one paragraph.",
    "What are the advantages of tree-based verification compared to sequence speculation?",
    "Write a short Python function that computes Fibonacci numbers iteratively.",
    "Summarize why GPU memory bandwidth often bottlenecks LLM inference.",
]


def _dir_has_weights(p: Path) -> bool:
    if not p.exists():
        return False
    if (p / "config.json").exists():
        return True
    return any(p.glob("**/config.json")) or any(p.glob("*.safetensors")) or any(
        p.glob("pytorch_model*.bin")
    )


def resolve_model(name: str) -> str:
    """Prefer local HF / ModelScope caches when present."""
    # ModelScope snapshot layout: cache_dir/modelscope/xxx/yyy
    ms_roots = [
        Path("/root/autodl-tmp/hf_home/modelscope"),
        Path("/root/autodl-tmp/hf_home/modelscope/models"),
    ]
    local_map = {
        "huggyllama/llama-7b": [
            "/root/autodl-tmp/hf_home/models/llama-7b",
            "/root/autodl-tmp/hf_home/hub/models--huggyllama--llama-7b",
        ],
        "JackFram/llama-68m": [
            "/root/autodl-tmp/hf_home/hub/models--JackFram--llama-68m/snapshots/9de84537b6aa98de634ad0fbb1608e9d6a019355",
            "JackFram/llama-68m",
        ],
        "Qwen/Qwen2.5-7B-Instruct": [
            "/root/autodl-tmp/hf_home/modelscope/Qwen/Qwen2.5-7B-Instruct",
            "/root/autodl-tmp/hf_home/modelscope/models/Qwen/Qwen2.5-7B-Instruct",
        ],
        "Qwen/Qwen2.5-3B-Instruct": [
            "/root/autodl-tmp/hf_home/modelscope/Qwen/Qwen2.5-3B-Instruct",
            "/root/autodl-tmp/hf_home/modelscope/models/Qwen/Qwen2.5-3B-Instruct",
        ],
    }
    for path in local_map.get(name, []):
        p = Path(path)
        if _dir_has_weights(p):
            if "hub/models--" in path and not (p / "config.json").exists():
                return name
            return str(p)
    # Generic ModelScope fallback: search by last path component.
    leaf = name.split("/")[-1]
    for root in ms_roots:
        if not root.exists():
            continue
        hits = list(root.glob(f"**/{leaf}"))
        for h in hits:
            if _dir_has_weights(h):
                return str(h)
    return name


def aggregate(metrics_list: List[RunMetrics]) -> Dict[str, float]:
    assert metrics_list
    keys = [
        "per_token_latency_ms",
        "avg_accepted_len",
        "draft_ms_sum",
        "verify_ms_sum",
        "avg_tree_nodes",
        "avg_nfe",
        "num_tokens",
        "wall_ms",
        "steps",
    ]
    out: Dict[str, float] = {}
    for k in keys:
        vals = [float(m.summary()[k]) for m in metrics_list]
        out[k] = sum(vals) / len(vals)
    out["batch_size"] = metrics_list[0].batch_size
    return out


def run_mode(
    mode: str,
    prompts: List[str],
    *,
    target: str,
    draft: str,
    batch_size: int,
    adaptive: bool,
    max_new_tokens: int,
    top_k: int,
    nfe: int,
    max_width: int,
) -> Dict[str, float]:
    cfg = EngineConfig(
        target_model=resolve_model(target),
        draft_model=resolve_model(draft),
        mode=mode,
        max_new_tokens=max_new_tokens,
        draft_len=8,
        top_k=top_k,
        nfe=nfe,
        max_width=max_width,
        adaptive=adaptive,
        device="cuda",
        dtype="float16",
    )
    # Local weight directories often lack tokenizer files.
    tm = cfg.target_model
    if isinstance(tm, str) and tm.startswith("/"):
        low = tm.lower()
        if "68m" in low or "jackfram" in low:
            cfg.tokenizer_name = "JackFram/llama-68m"
        elif "llama-7b" in low or "huggyllama" in low:
            cfg.tokenizer_name = "huggyllama/llama-7b"
        else:
            cfg.tokenizer_name = draft if not str(draft).startswith("/") else target
    else:
        cfg.tokenizer_name = target

    print(f"\n=== mode={mode} bs={batch_size} adaptive={adaptive} target={cfg.target_model} ===")
    engine = DiffusionTreeEngine(cfg)
    results = []
    for i, prompt in enumerate(prompts):
        print(f"  prompt[{i}] ...", flush=True)
        m = engine.generate(prompt, batch_size=batch_size)
        print(
            f"    tokens={m.num_tokens} pt={m.per_token_latency_ms:.2f}ms "
            f"alpha={m.avg_accepted_len:.2f} tree={m.avg_tree_nodes:.1f}"
        )
        results.append(m)
    # Free GPU before next mode.
    del engine
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    summary = aggregate(results)
    summary["mode"] = mode
    summary["adaptive"] = float(adaptive)
    summary["top_k"] = float(top_k)
    summary["nfe"] = float(nfe)
    summary["max_width"] = float(max_width)
    return summary


def write_csv(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Smoke test (few tokens/prompts)")
    parser.add_argument("--full", action="store_true", help="Full single-GPU experiment suite")
    parser.add_argument("--target", default=os.environ.get("DTS_TARGET", "huggyllama/llama-7b"))
    parser.add_argument("--draft", default=os.environ.get("DTS_DRAFT", "JackFram/llama-68m"))
    parser.add_argument(
        "--out-dir",
        default=str(
            _ROOT / "FlexFlow" / "inference" / "output" / "diffusion_tree_spec"
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.quick:
        prompts = DEFAULT_PROMPTS[:2]
        max_new = args.max_new_tokens or 24
        batch_sizes = [1]
        modes = ["incr", "ar_seq", "ar_tree", "diff_tree"]
        do_ablation = True
    else:
        # Default to full suite when --full or neither flag.
        prompts = DEFAULT_PROMPTS
        max_new = args.max_new_tokens or 64
        batch_sizes = [1, 2, 4, 8]
        modes = ["incr", "ar_seq", "ar_tree", "diff_tree"]
        do_ablation = True

    rows: List[Dict] = []

    # Main BS sweep with A+B (diff_tree adaptive) vs baselines.
    for bs in batch_sizes:
        base_pt = None
        for mode in modes:
            adaptive = mode == "diff_tree"
            row = run_mode(
                mode,
                prompts,
                target=args.target,
                draft=args.draft,
                batch_size=bs,
                adaptive=adaptive,
                max_new_tokens=max_new,
                top_k=5,
                nfe=4,
                max_width=3,
            )
            if mode == "incr":
                base_pt = row["per_token_latency_ms"]
                row["speedup_vs_incr"] = 1.0
            else:
                row["speedup_vs_incr"] = (
                    speedup(base_pt, row["per_token_latency_ms"]) if base_pt else 0.0
                )
            rows.append(row)

    write_csv(rows, out_dir / "main_bs_sweep.csv")
    with (out_dir / "main_bs_sweep.json").open("w") as f:
        json.dump(rows, f, indent=2)

    if do_ablation:
        abl_rows: List[Dict] = []
        # Ablation: top-k / width
        for k in [1, 2, 3, 5]:
            row = run_mode(
                "diff_tree",
                prompts[:2],
                target=args.target,
                draft=args.draft,
                batch_size=1,
                adaptive=False,
                max_new_tokens=max_new,
                top_k=k,
                nfe=4,
                max_width=k,
            )
            row["ablation"] = f"topk_{k}"
            abl_rows.append(row)
        # Ablation: fixed NFE vs adaptive B
        for adaptive, tag in [(False, "fixed_nfe"), (True, "adaptive_B")]:
            row = run_mode(
                "diff_tree",
                prompts[:2],
                target=args.target,
                draft=args.draft,
                batch_size=4,
                adaptive=adaptive,
                max_new_tokens=max_new,
                top_k=5,
                nfe=4,
                max_width=3,
            )
            row["ablation"] = tag
            abl_rows.append(row)
        # Ablation: prune margin via max_width extremes (proxy for prune strength)
        for width, tag in [(1, "prune_narrow"), (5, "prune_wide")]:
            row = run_mode(
                "diff_tree",
                prompts[:2],
                target=args.target,
                draft=args.draft,
                batch_size=1,
                adaptive=False,
                max_new_tokens=max_new,
                top_k=5,
                nfe=4,
                max_width=width,
            )
            row["ablation"] = tag
            abl_rows.append(row)

        write_csv(abl_rows, out_dir / "ablation.csv")
        with (out_dir / "ablation.json").open("w") as f:
            json.dump(abl_rows, f, indent=2)

    print("\nAll experiments finished.")
    print(f"Results under: {out_dir}")


if __name__ == "__main__":
    main()
