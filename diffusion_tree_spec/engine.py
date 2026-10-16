"""End-to-end engine: Diffusion Lattice → Token Tree + Timestep-as-Tree."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from .adaptive import TimestepAsTreeController
from .diffusion_drafter import (
    ARSequenceDrafter,
    ARTreeDrafter,
    DiscreteDiffusionDrafter,
)
from .metrics import RunMetrics
from .token_tree import build_tree_from_lattice
from .tree_verify import incremental_decode_step, verify_tree_greedy


@dataclass
class EngineConfig:
    target_model: str = "huggyllama/llama-7b"
    draft_model: str = "JackFram/llama-68m"
    # Optional separate tokenizer source (useful when weight dir lacks tokenizer files).
    tokenizer_name: Optional[str] = None
    mode: str = "diff_tree"  # incr | ar_seq | ar_tree | diff_tree
    max_new_tokens: int = 64
    draft_len: int = 8
    top_k: int = 5
    nfe: int = 4
    max_width: int = 3
    prune_margin: float = 1.0
    adaptive: bool = True  # innovation B
    dtype: str = "float16"
    device: str = "cuda"
    trust_remote_code: bool = True


class DiffusionTreeEngine:
    def __init__(self, cfg: EngineConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        dtype = dict(float16=torch.float16, bfloat16=torch.bfloat16, float32=torch.float32)[
            cfg.dtype
        ]
        if self.device.type == "cpu":
            dtype = torch.float32

        tok_src = cfg.tokenizer_name or cfg.target_model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tok_src, use_fast=True, trust_remote_code=cfg.trust_remote_code
            )
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tok_src, use_fast=False, trust_remote_code=cfg.trust_remote_code
            )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kw = dict(trust_remote_code=cfg.trust_remote_code)
        # transformers>=4.56 prefers `dtype=`; older builds use `torch_dtype=`.
        try:
            self.target = AutoModelForCausalLM.from_pretrained(
                cfg.target_model, dtype=dtype, **load_kw
            )
        except TypeError:
            self.target = AutoModelForCausalLM.from_pretrained(
                cfg.target_model, torch_dtype=dtype, **load_kw
            )
        self.target.to(self.device)
        self.target.eval()

        self.draft_model: Optional[PreTrainedModel] = None
        self.drafter: Any = None
        if cfg.mode != "incr":
            try:
                self.draft_model = AutoModelForCausalLM.from_pretrained(
                    cfg.draft_model, dtype=dtype, **load_kw
                )
            except TypeError:
                self.draft_model = AutoModelForCausalLM.from_pretrained(
                    cfg.draft_model, torch_dtype=dtype, **load_kw
                )
            self.draft_model.to(self.device)
            self.draft_model.eval()
            if cfg.mode == "ar_seq":
                self.drafter = ARSequenceDrafter(
                    self.draft_model, draft_len=cfg.draft_len, device=self.device
                )
            elif cfg.mode == "ar_tree":
                self.drafter = ARTreeDrafter(
                    self.draft_model,
                    draft_len=cfg.draft_len,
                    width=cfg.max_width,
                    device=self.device,
                )
            else:
                self.drafter = DiscreteDiffusionDrafter(
                    self.draft_model,
                    draft_len=cfg.draft_len,
                    top_k=cfg.top_k,
                    nfe=cfg.nfe,
                    device=self.device,
                )

        self.controller = TimestepAsTreeController(
            base_draft_len=cfg.draft_len,
            max_width=cfg.max_width,
            max_nfe=max(cfg.nfe, 2),
            enabled=cfg.adaptive and cfg.mode == "diff_tree",
        )

    def generate(self, prompt: str, *, batch_size: int = 1) -> RunMetrics:
        """Generate for one prompt. ``batch_size`` is a load proxy for controller B."""
        metrics = RunMetrics(mode=self.cfg.mode, batch_size=batch_size)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        generated: List[int] = []
        t_wall0 = time.perf_counter()

        while len(generated) < self.cfg.max_new_tokens:
            if self.cfg.mode == "incr":
                tok, step_ms = incremental_decode_step(self.target, input_ids)
                input_ids = torch.cat(
                    [input_ids, torch.tensor([[tok]], device=self.device)], dim=1
                )
                generated.append(tok)
                metrics.draft_ms_sum += 0.0
                metrics.verify_ms_sum += step_ms
                metrics.steps += 1
                metrics.accepted_lens.append(1.0)
                metrics.tree_nodes_list.append(0)
                if tok == self.tokenizer.eos_token_id:
                    break
                continue

            decision = self.controller.decide(batch_size=batch_size)
            lattice = self.drafter.draft_lattice(
                input_ids,
                nfe=decision.nfe if self.cfg.mode == "diff_tree" else None,
                top_k=decision.top_k,
                draft_len=decision.draft_len,
            )
            tree = build_tree_from_lattice(
                lattice.topk,
                max_width=decision.max_width,
                max_nodes=64,
                prune_margin=self.cfg.prune_margin,
            )
            vr = verify_tree_greedy(self.target, input_ids, tree, device=self.device)

            # Append accepted tokens (verify already includes bonus/correction).
            for tok in vr.accepted_tokens:
                if len(generated) >= self.cfg.max_new_tokens:
                    break
                generated.append(tok)
                input_ids = torch.cat(
                    [input_ids, torch.tensor([[tok]], device=self.device)], dim=1
                )
                if tok == self.tokenizer.eos_token_id:
                    break

            metrics.draft_ms_sum += lattice.draft_ms
            metrics.verify_ms_sum += vr.verify_ms
            metrics.steps += 1
            metrics.accepted_lens.append(float(len(vr.accepted_tokens)))
            metrics.tree_nodes_list.append(int(vr.tree_nodes))
            metrics.nfe_list.append(int(lattice.nfe_used))
            metrics.decisions.append(decision.reason)

            self.controller.observe(len(vr.accepted_tokens))

            if generated and generated[-1] == self.tokenizer.eos_token_id:
                break

        metrics.wall_ms = (time.perf_counter() - t_wall0) * 1000.0
        metrics.num_tokens = len(generated)
        metrics.text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return metrics
