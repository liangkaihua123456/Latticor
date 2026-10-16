#!/usr/bin/env python3
"""Convert FlexFlow LLaMA-7B half-precision raw weights to HuggingFace format."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

FF_DIR = Path("/root/autodl-tmp/flexflow_cache/weights/huggyllama/llama-7b/half-precision")
CFG_SRC = Path("/root/autodl-tmp/flexflow_cache/configs/huggyllama/llama-7b/config.json")
TOK_SRC = Path("/root/autodl-tmp/flexflow_cache/tokenizers/huggyllama/llama-7b")
OUT_DIR = Path("/root/autodl-tmp/hf_home/models/llama-7b-from-flexflow")

HIDDEN = 4096
INTER = 11008
VOCAB = 32000
NLAYERS = 32


def load_f16(name: str, shape: tuple[int, ...]) -> torch.Tensor:
    arr = np.fromfile(FF_DIR / name, dtype=np.float16)
    if arr.size != int(np.prod(shape)):
        raise ValueError(f"{name}: got {arr.size}, expect {np.prod(shape)}")
    return torch.from_numpy(arr.reshape(shape).copy())


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    state = {}

    print("loading embeddings...", flush=True)
    state["model.embed_tokens.weight"] = load_f16("tok_embeddings_weight", (VOCAB, HIDDEN))
    state["model.norm.weight"] = load_f16("norm_weight", (HIDDEN,))
    state["lm_head.weight"] = load_f16("output_weight", (VOCAB, HIDDEN))

    for i in range(NLAYERS):
        print(f"layer {i}/{NLAYERS-1}", flush=True)
        p = f"model.layers.{i}"
        state[f"{p}.input_layernorm.weight"] = load_f16(
            f"layers_{i}_attention_norm_weight", (HIDDEN,)
        )
        # HF Linear: (out, in). FlexFlow dumps match HF shapes for these tensors.
        state[f"{p}.self_attn.q_proj.weight"] = load_f16(
            f"layers_{i}_attention_wq_weight", (HIDDEN, HIDDEN)
        )
        state[f"{p}.self_attn.k_proj.weight"] = load_f16(
            f"layers_{i}_attention_wk_weight", (HIDDEN, HIDDEN)
        )
        state[f"{p}.self_attn.v_proj.weight"] = load_f16(
            f"layers_{i}_attention_wv_weight", (HIDDEN, HIDDEN)
        )
        state[f"{p}.self_attn.o_proj.weight"] = load_f16(
            f"layers_{i}_attention_wo_weight", (HIDDEN, HIDDEN)
        )
        state[f"{p}.post_attention_layernorm.weight"] = load_f16(
            f"layers_{i}_ffn_norm_weight", (HIDDEN,)
        )
        state[f"{p}.mlp.gate_proj.weight"] = load_f16(
            f"layers_{i}_feed_forward_w1_weight", (INTER, HIDDEN)
        )
        state[f"{p}.mlp.down_proj.weight"] = load_f16(
            f"layers_{i}_feed_forward_w2_weight", (HIDDEN, INTER)
        )
        state[f"{p}.mlp.up_proj.weight"] = load_f16(
            f"layers_{i}_feed_forward_w3_weight", (INTER, HIDDEN)
        )

    print("saving safetensors...", flush=True)
    save_file(state, str(OUT_DIR / "model.safetensors"))

    cfg = json.loads(CFG_SRC.read_text())
    (OUT_DIR / "config.json").write_text(json.dumps(cfg, indent=2))

    for fn in ["tokenizer.model", "tokenizer_config.json", "special_tokens_map.json"]:
        src = TOK_SRC / fn
        if src.exists():
            shutil.copy2(src, OUT_DIR / fn)

    # Minimal generation config
    gen = {
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
    }
    (OUT_DIR / "generation_config.json").write_text(json.dumps(gen, indent=2))
    print("DONE", OUT_DIR, flush=True)


if __name__ == "__main__":
    main()
