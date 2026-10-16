#!/usr/bin/env python3
"""Download the paper models into a FlexFlow half-precision cache.

Weights are not stored in this repository. This script pulls the four
Hugging Face repos used by the diffusion-lattice experiments and writes
the same on-disk layout as FlexFlow Serve:

    <cache>/configs/<org>/<name>/config.json
    <cache>/tokenizers/<org>/<name>/
    <cache>/weights/<org>/<name>/half-precision/<tensor files>

Tensor filenames follow FlexFlow's convert_hf_model (Apache-2.0).
Revisions are pinned to the commits recorded in the original cache.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "models.json"


def load_manifest(selected: list[str] | None) -> list[dict]:
    models = json.loads(MANIFEST.read_text())
    if not selected:
        return models
    wanted = {name.lower() for name in selected}
    chosen = [m for m in models if m["repo_id"].lower() in wanted]
    missing = wanted - {m["repo_id"].lower() for m in chosen}
    if missing:
        known = ", ".join(m["repo_id"] for m in models)
        raise SystemExit(f"Unknown model(s): {', '.join(sorted(missing))}. Known: {known}")
    return chosen


def cache_key(repo_id: str) -> str:
    return repo_id.lower()


def read_sha(path: Path) -> str | None:
    if not path.is_file():
        return None
    return "".join(path.read_text().split())


def save_f16(param: torch.Tensor, path: Path) -> None:
    array = param.detach().to(dtype=torch.float16, device="cpu").contiguous().numpy()
    if array.dtype != np.float16:
        array = array.astype(np.float16)
    array.tofile(path)


def convert_llama(model, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name, params in model.named_parameters():
        name = (
            name.replace(".", "_")
            .replace("self_attn", "attention")
            .replace("q_proj", "wq")
            .replace("k_proj", "wk")
            .replace("v_proj", "wv")
            .replace("o_proj", "wo")
            .replace("mlp", "feed_forward")
            .replace("gate_proj", "w1")
            .replace("down_proj", "w2")
            .replace("up_proj", "w3")
            .replace("input_layernorm", "attention_norm")
            .replace("post_attention_layernorm", "ffn_norm")
            .replace("embed_tokens", "tok_embeddings")
            .replace("lm_head", "output")
            .replace("model_", "")
        )
        save_f16(params, dst / name)


def convert_opt(model, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name, params in model.named_parameters():
        name = (
            name.replace(".", "_")
            .replace("decoder_", "")
            .replace("model_", "")
            .replace("self_attn", "attention")
            .replace("q_proj", "wq")
            .replace("k_proj", "wk")
            .replace("v_proj", "wv")
            .replace("out_proj", "wo")
            .replace("attention_wo_bias", "add_bias_residual_layer_norm_attn_bias")
            .replace("_final_layer_norm", "_add_bias_residual_layer_norm")
        )
        save_f16(params, dst / name)
    shutil.copy(dst / "embed_tokens_weight", dst / "embed_tokens_weight_lm_head")


def convert(model, arch: str, dst: Path) -> None:
    if arch == "LlamaForCausalLM":
        convert_llama(model, dst)
    elif arch == "OPTForCausalLM":
        convert_opt(model, dst)
    else:
        raise SystemExit(f"Unsupported architecture: {arch}")


def download_one(spec: dict, cache: Path, refresh: bool) -> None:
    repo_id = spec["repo_id"]
    revision = spec["revision"]
    key = cache_key(repo_id)
    weights = cache / "weights" / key / "half-precision"
    tokenizer_dir = cache / "tokenizers" / key
    config_dir = cache / "configs" / key
    sha_file = weights / "rev_sha.txt"

    if not refresh and read_sha(sha_file) == revision and any(weights.iterdir()):
        print(f"[{repo_id}] cache hit at {weights}")
        return

    print(f"[{repo_id}] revision {revision}")
    config = AutoConfig.from_pretrained(repo_id, revision=revision, trust_remote_code=True)
    arch = (config.architectures or [None])[0]
    if arch is None:
        raise SystemExit(f"{repo_id} has no architectures field")

    config_dir.mkdir(parents=True, exist_ok=True)
    config._name_or_path = repo_id
    config.to_json_file(str(config_dir / "config.json"))

    print(f"[{repo_id}] downloading weights...")
    load_kwargs = {
        "revision": revision,
        "torch_dtype": torch.float16,
        "trust_remote_code": True,
    }
    try:
        import accelerate  # noqa: F401
    except ImportError:
        pass
    else:
        load_kwargs["low_cpu_mem_usage"] = True
    model = AutoModelForCausalLM.from_pretrained(repo_id, **load_kwargs)
    if refresh and weights.exists():
        shutil.rmtree(weights)
    print(f"[{repo_id}] writing FlexFlow tensors to {weights}")
    convert(model, arch, weights)
    sha_file.write_text(revision)
    del model

    print(f"[{repo_id}] downloading tokenizer...")
    if refresh and tokenizer_dir.exists():
        shutil.rmtree(tokenizer_dir)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(repo_id, revision=revision, use_fast=True)
    tokenizer.save_pretrained(tokenizer_dir)
    (tokenizer_dir / "rev_sha.txt").write_text(revision)
    print(f"[{repo_id}] done")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-folder",
        type=Path,
        default=Path(os.environ.get("FF_CACHE_PATH", ROOT / "downloaded")),
        help="Destination FlexFlow cache (default: ./downloaded or $FF_CACHE_PATH)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Subset of repo ids. Default: every model in models.json",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Delete and rewrite weights even when the pinned revision is already present",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache = args.cache_folder.expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    print(f"Cache folder: {cache}")
    if os.environ.get("HF_ENDPOINT"):
        print(f"HF_ENDPOINT: {os.environ['HF_ENDPOINT']}")
    for spec in load_manifest(args.models):
        download_one(spec, cache, args.refresh)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
