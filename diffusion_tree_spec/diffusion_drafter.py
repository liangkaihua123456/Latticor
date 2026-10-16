"""Innovation A: discrete-diffusion-style parallel drafter → token lattice."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel


@dataclass
class LatticeResult:
    """Per-position top-k candidates with log-probs."""

    topk: List[List[Tuple[int, float]]]
    nfe_used: int
    draft_ms: float
    best_tokens: List[int]


class DiscreteDiffusionDrafter:
    """Parallel draft lattice with a small LM (MaskGIT-like iterative refine).

    Keeps small-model forwards to ``nfe`` (not draft_len). Cheap when the
    target LLM is much larger than the drafter (RTX 6000D co-resident setup).
    """

    def __init__(
        self,
        model: PreTrainedModel,
        *,
        draft_len: int = 8,
        top_k: int = 5,
        nfe: int = 4,
        conf_threshold: float = 0.35,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.draft_len = draft_len
        self.top_k = top_k
        self.nfe = nfe
        self.conf_threshold = conf_threshold
        self.device = device or next(model.parameters()).device
        self.mask_id: Optional[int] = getattr(model.config, "pad_token_id", None)
        if self.mask_id is None:
            self.mask_id = 0

    @torch.inference_mode()
    def draft_lattice(
        self,
        prompt_ids: torch.Tensor,
        *,
        nfe: Optional[int] = None,
        top_k: Optional[int] = None,
        draft_len: Optional[int] = None,
    ) -> LatticeResult:
        nfe = nfe if nfe is not None else self.nfe
        top_k = top_k if top_k is not None else self.top_k
        draft_len = draft_len if draft_len is not None else self.draft_len

        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        prompt_ids = prompt_ids.to(self.device)
        bsz, plen = prompt_ids.shape
        assert bsz == 1, "drafter currently supports batch size 1 per call"

        t0 = time.perf_counter()
        revealed = torch.full((draft_len,), -1, dtype=torch.long, device=self.device)
        vocab = self.model.config.vocab_size
        acc_logits = torch.zeros(draft_len, vocab, device=self.device)
        acc_count = torch.zeros(draft_len, device=self.device)

        for step in range(max(1, nfe)):
            draft_fill = revealed.clone()
            draft_fill[draft_fill < 0] = int(self.mask_id)
            inp = torch.cat([prompt_ids, draft_fill.unsqueeze(0)], dim=1)
            logits = self.model(input_ids=inp).logits[0]
            for i in range(draft_len):
                acc_logits[i] += logits[plen - 1 + i]
                acc_count[i] += 1.0

            still_masked = (revealed < 0).nonzero(as_tuple=False).flatten()
            if len(still_masked) == 0:
                break

            confs, toks = [], []
            for i in still_masked.tolist():
                probs = F.softmax(acc_logits[i] / max(acc_count[i], 1.0), dim=-1)
                conf, tok = torch.max(probs, dim=-1)
                confs.append(conf.item())
                toks.append(tok.item())

            order = sorted(range(len(still_masked)), key=lambda j: confs[j], reverse=True)
            frac = (step + 1) / max(1, nfe)
            already = int((revealed >= 0).sum().item())
            budget = max(1, int(round(draft_len * frac)) - already)
            revealed_this = 0
            for j in order:
                i = int(still_masked[j].item())
                if confs[j] >= self.conf_threshold or revealed_this < budget:
                    revealed[i] = toks[j]
                    revealed_this += 1
                if revealed_this >= budget:
                    break

        for i in range(draft_len):
            if revealed[i] < 0:
                revealed[i] = int(torch.argmax(acc_logits[i]).item())

        lattice: List[List[Tuple[int, float]]] = []
        best_tokens: List[int] = []
        for i in range(draft_len):
            avg = acc_logits[i] / max(acc_count[i].item(), 1.0)
            log_probs = F.log_softmax(avg, dim=-1)
            vals, idxs = torch.topk(log_probs, k=min(top_k, log_probs.numel()))
            cands = [(int(idxs[j]), float(vals[j])) for j in range(vals.numel())]
            lattice.append(cands)
            best_tokens.append(cands[0][0])

        return LatticeResult(
            topk=lattice,
            nfe_used=nfe,
            draft_ms=(time.perf_counter() - t0) * 1000.0,
            best_tokens=best_tokens,
        )


class ARTreeDrafter:
    """SpecInfer-style expansion tree using a small AR model (baseline)."""

    def __init__(
        self,
        model: PreTrainedModel,
        *,
        draft_len: int = 8,
        expand_at: int = 2,
        width: int = 3,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.draft_len = draft_len
        self.expand_at = expand_at
        self.width = width
        self.device = device or next(model.parameters()).device

    @torch.inference_mode()
    def draft_lattice(
        self,
        prompt_ids: torch.Tensor,
        *,
        nfe: Optional[int] = None,
        top_k: Optional[int] = None,
        draft_len: Optional[int] = None,
    ) -> LatticeResult:
        del nfe
        top_k = top_k if top_k is not None else self.width
        draft_len = draft_len if draft_len is not None else self.draft_len
        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        prompt_ids = prompt_ids.to(self.device)

        t0 = time.perf_counter()
        lattice: List[List[Tuple[int, float]]] = []
        seq = prompt_ids
        best_tokens: List[int] = []
        past = None
        for _ in range(draft_len):
            if past is None:
                out = self.model(input_ids=seq, use_cache=True)
            else:
                out = self.model(
                    input_ids=seq[:, -1:], past_key_values=past, use_cache=True
                )
            past = out.past_key_values
            log_probs = F.log_softmax(out.logits[0, -1], dim=-1)
            vals2, idxs2 = torch.topk(log_probs, k=min(top_k, log_probs.numel()))
            cands = [(int(idxs2[j]), float(vals2[j])) for j in range(vals2.numel())]
            lattice.append(cands)
            best = cands[0][0]
            best_tokens.append(best)
            seq = torch.cat(
                [seq, torch.tensor([[best]], device=self.device, dtype=seq.dtype)],
                dim=1,
            )
        return LatticeResult(
            topk=lattice,
            nfe_used=draft_len,
            draft_ms=(time.perf_counter() - t0) * 1000.0,
            best_tokens=best_tokens,
        )


class ARSequenceDrafter:
    """Classic speculative decoding: single sequence draft."""

    def __init__(
        self,
        model: PreTrainedModel,
        *,
        draft_len: int = 8,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.draft_len = draft_len
        self.device = device or next(model.parameters()).device

    @torch.inference_mode()
    def draft_lattice(
        self,
        prompt_ids: torch.Tensor,
        *,
        nfe: Optional[int] = None,
        top_k: Optional[int] = None,
        draft_len: Optional[int] = None,
    ) -> LatticeResult:
        del nfe, top_k
        draft_len = draft_len if draft_len is not None else self.draft_len
        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        prompt_ids = prompt_ids.to(self.device)

        t0 = time.perf_counter()
        lattice: List[List[Tuple[int, float]]] = []
        seq = prompt_ids
        best_tokens: List[int] = []
        past = None
        for _ in range(draft_len):
            if past is None:
                out = self.model(input_ids=seq, use_cache=True)
            else:
                out = self.model(
                    input_ids=seq[:, -1:], past_key_values=past, use_cache=True
                )
            past = out.past_key_values
            log_probs = F.log_softmax(out.logits[0, -1], dim=-1)
            tok = int(torch.argmax(log_probs))
            lattice.append([(tok, float(log_probs[tok]))])
            best_tokens.append(tok)
            seq = torch.cat(
                [seq, torch.tensor([[tok]], device=self.device, dtype=seq.dtype)],
                dim=1,
            )
        return LatticeResult(
            topk=lattice,
            nfe_used=draft_len,
            draft_ms=(time.perf_counter() - t0) * 1000.0,
            best_tokens=best_tokens,
        )
