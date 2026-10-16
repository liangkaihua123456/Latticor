#!/usr/bin/env python3
"""Convert FlexFlow OPT half-precision raw weights to HuggingFace OPT format."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file


def load_f16(path: Path, shape: tuple[int, ...]) -> torch.Tensor:
    arr = np.fromfile(path, dtype=np.float16)
    if arr.size != int(np.prod(shape)):
        raise ValueError(f"{path.name}: got {arr.size}, expect {np.prod(shape)}")
    return torch.from_numpy(arr.reshape(shape).copy())


def convert_opt(ff_dir: Path, cfg_path: Path, tok_dir: Path, out_dir: Path) -> None:
    cfg = json.loads(cfg_path.read_text())
    hidden = int(cfg["hidden_size"])
    ffn = int(cfg["ffn_dim"])
    vocab = int(cfg["vocab_size"])
    nlayers = int(cfg["num_hidden_layers"])
    max_pos = int(cfg["max_position_embeddings"])
    # OPT stores positions as max_pos + 2
    pos_rows = max_pos + 2

    half = ff_dir / "half-precision"
    out_dir.mkdir(parents=True, exist_ok=True)
    state = {}

    print("embeddings...", flush=True)
    state["model.decoder.embed_tokens.weight"] = load_f16(
        half / "embed_tokens_weight", (vocab, hidden)
    )
    state["model.decoder.embed_positions.weight"] = load_f16(
        half / "embed_positions_weight", (pos_rows, hidden)
    )
    state["lm_head.weight"] = load_f16(
        half / "embed_tokens_weight_lm_head", (vocab, hidden)
    )
    state["model.decoder.final_layer_norm.weight"] = load_f16(
        half / "final_layer_norm_weight", (hidden,)
    )
    state["model.decoder.final_layer_norm.bias"] = load_f16(
        half / "final_layer_norm_bias", (hidden,)
    )

    for i in range(nlayers):
        print(f"layer {i}/{nlayers-1}", flush=True)
        p = f"model.decoder.layers.{i}"
        state[f"{p}.self_attn_layer_norm.weight"] = load_f16(
            half / f"layers_{i}_attention_layer_norm_weight", (hidden,)
        )
        state[f"{p}.self_attn_layer_norm.bias"] = load_f16(
            half / f"layers_{i}_attention_layer_norm_bias", (hidden,)
        )
        state[f"{p}.self_attn.q_proj.weight"] = load_f16(
            half / f"layers_{i}_attention_wq_weight", (hidden, hidden)
        )
        state[f"{p}.self_attn.q_proj.bias"] = load_f16(
            half / f"layers_{i}_attention_wq_bias", (hidden,)
        )
        state[f"{p}.self_attn.k_proj.weight"] = load_f16(
            half / f"layers_{i}_attention_wk_weight", (hidden, hidden)
        )
        state[f"{p}.self_attn.k_proj.bias"] = load_f16(
            half / f"layers_{i}_attention_wk_bias", (hidden,)
        )
        state[f"{p}.self_attn.v_proj.weight"] = load_f16(
            half / f"layers_{i}_attention_wv_weight", (hidden, hidden)
        )
        state[f"{p}.self_attn.v_proj.bias"] = load_f16(
            half / f"layers_{i}_attention_wv_bias", (hidden,)
        )
        state[f"{p}.self_attn.out_proj.weight"] = load_f16(
            half / f"layers_{i}_attention_wo_weight", (hidden, hidden)
        )
        # FlexFlow dumps attention residual bias separately (no wo_bias file).
        attn_bias = half / f"layers_{i}_add_bias_residual_layer_norm_attn_bias"
        if attn_bias.exists():
            state[f"{p}.self_attn.out_proj.bias"] = load_f16(attn_bias, (hidden,))
        else:
            state[f"{p}.self_attn.out_proj.bias"] = torch.zeros(hidden, dtype=torch.float16)

        state[f"{p}.final_layer_norm.weight"] = load_f16(
            half / f"layers_{i}_add_bias_residual_layer_norm_weight", (hidden,)
        )
        state[f"{p}.final_layer_norm.bias"] = load_f16(
            half / f"layers_{i}_add_bias_residual_layer_norm_bias", (hidden,)
        )
        state[f"{p}.fc1.weight"] = load_f16(
            half / f"layers_{i}_fc1_weight", (ffn, hidden)
        )
        state[f"{p}.fc1.bias"] = load_f16(half / f"layers_{i}_fc1_bias", (ffn,))
        state[f"{p}.fc2.weight"] = load_f16(
            half / f"layers_{i}_fc2_weight", (hidden, ffn)
        )
        state[f"{p}.fc2.bias"] = load_f16(half / f"layers_{i}_fc2_bias", (hidden,))

    print("saving...", flush=True)
    # Split into shards if large
    total = sum(v.numel() * 2 for v in state.values())
    if total > 8 * (1024**3):
        # two shards roughly half keys
        keys = list(state.keys())
        mid = len(keys) // 2
        s1 = {k: state[k] for k in keys[:mid]}
        s2 = {k: state[k] for k in keys[mid:]}
        save_file(s1, str(out_dir / "model-00001-of-00002.safetensors"))
        save_file(s2, str(out_dir / "model-00002-of-00002.safetensors"))
        index = {
            "metadata": {"total_size": total},
            "weight_map": {
                **{k: "model-00001-of-00002.safetensors" for k in s1},
                **{k: "model-00002-of-00002.safetensors" for k in s2},
            },
        }
        (out_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    else:
        save_file(state, str(out_dir / "model.safetensors"))

    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    for fn in [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    ]:
        src = tok_dir / fn
        if src.exists():
            shutil.copy2(src, out_dir / fn)
    gen = {
        "bos_token_id": cfg.get("bos_token_id", 2),
        "eos_token_id": cfg.get("eos_token_id", 2),
        "pad_token_id": cfg.get("pad_token_id", 1),
    }
    (out_dir / "generation_config.json").write_text(json.dumps(gen, indent=2))
    print("DONE", out_dir, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["opt-13b", "opt-125m"], required=True)
    args = ap.parse_args()
    root = Path("/root/autodl-tmp/flexflow_cache")
    if args.model == "opt-13b":
        convert_opt(
            root / "weights/facebook/opt-13b",
            root / "configs/facebook/opt-13b/config.json",
            root / "tokenizers/facebook/opt-13b",
            Path("/root/autodl-tmp/hf_home/models/opt-13b-from-flexflow"),
        )
    else:
        convert_opt(
            root / "weights/facebook/opt-125m",
            root / "configs/facebook/opt-125m/config.json",
            root / "tokenizers/facebook/opt-125m",
            Path("/root/autodl-tmp/hf_home/models/opt-125m-from-flexflow"),
        )


if __name__ == "__main__":
    main()
