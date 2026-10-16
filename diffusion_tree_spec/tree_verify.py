"""Tree-based parallel verification against the target LLM (SpecInfer-style)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel

from .token_tree import TokenTree, build_tree_attention_mask


@dataclass
class VerifyResult:
    accepted_tokens: List[int]
    verify_ms: float
    tree_nodes: int
    # Logits at the last accepted position (for sampling next bonus token).
    bonus_token: int


@torch.inference_mode()
def verify_tree_greedy(
    target: PreTrainedModel,
    prompt_ids: torch.Tensor,
    tree: TokenTree,
    *,
    device: Optional[torch.device] = None,
) -> VerifyResult:
    """Greedy SpecInfer-style tree verification with topology-aware mask."""
    device = device or next(target.parameters()).device
    if prompt_ids.dim() == 1:
        prompt_ids = prompt_ids.unsqueeze(0)
    prompt_ids = prompt_ids.to(device)
    plen = prompt_ids.shape[1]

    if tree.size <= 1:
        # Empty draft: one normal decode step.
        t0 = time.perf_counter()
        out = target(input_ids=prompt_ids)
        next_tok = int(torch.argmax(out.logits[0, -1]))
        return VerifyResult(
            accepted_tokens=[next_tok],
            verify_ms=(time.perf_counter() - t0) * 1000.0,
            tree_nodes=0,
            bonus_token=next_tok,
        )

    draft_tok, pos_ids_full, attn_bias = build_tree_attention_mask(
        tree, plen, device, dtype=next(target.parameters()).dtype
    )
    input_ids = torch.cat([prompt_ids, draft_tok.unsqueeze(0)], dim=1)
    # Position ids: prompt then draft.
    pos = torch.arange(plen, device=device)
    pos_ids = pos_ids_full.clone()
    pos_ids[:plen] = pos

    t0 = time.perf_counter()
    # Many HF models accept attention_mask as 2D; for tree we pass 4D via
    # `attention_mask` when supported. Fallback: verify path-by-path.
    try:
        out = target(
            input_ids=input_ids,
            attention_mask=attn_bias,
            position_ids=pos_ids.unsqueeze(0),
        )
        logits = out.logits[0]
        accepted, bonus = _walk_tree_greedy(tree, logits, plen)
    except Exception:
        accepted, bonus, verify_extra = _verify_paths_fallback(
            target, prompt_ids, tree
        )
        return VerifyResult(
            accepted_tokens=accepted,
            verify_ms=(time.perf_counter() - t0) * 1000.0 + verify_extra,
            tree_nodes=tree.size - 1,
            bonus_token=bonus,
        )

    verify_ms = (time.perf_counter() - t0) * 1000.0
    return VerifyResult(
        accepted_tokens=accepted,
        verify_ms=verify_ms,
        tree_nodes=tree.size - 1,
        bonus_token=bonus,
    )


def _walk_tree_greedy(
    tree: TokenTree, logits: torch.Tensor, plen: int
) -> Tuple[List[int], int]:
    """Walk from root following greedy target predictions."""
    # Map node → packed column index.
    draft_node_ids = list(range(1, len(tree.nodes)))
    node_to_col = {nid: plen + j for j, nid in enumerate(draft_node_ids)}

    accepted: List[int] = []
    u = 0  # virtual root
    # Prediction at prompt last token:
    pred = int(torch.argmax(logits[plen - 1]))

    while True:
        children = tree.nodes[u].children
        matched = None
        for c in children:
            if tree.nodes[c].token_id == pred:
                matched = c
                break
        if matched is None:
            # Reject; emit target prediction as bonus/correction.
            accepted.append(pred)
            return accepted, pred
        accepted.append(tree.nodes[matched].token_id)
        u = matched
        # Next prediction from this node's column.
        col = node_to_col[u]
        pred = int(torch.argmax(logits[col]))
        if not tree.nodes[u].children:
            accepted.append(pred)
            return accepted, pred


def _verify_paths_fallback(
    target: PreTrainedModel,
    prompt_ids: torch.Tensor,
    tree: TokenTree,
) -> Tuple[List[int], int, float]:
    """Fallback: verify the best leaf path with standard speculative decoding."""
    paths = tree.leaf_paths()
    if not paths:
        t0 = time.perf_counter()
        out = target(input_ids=prompt_ids)
        tok = int(torch.argmax(out.logits[0, -1]))
        return [tok], tok, (time.perf_counter() - t0) * 1000.0

    # Prefer highest logprob leaf (stored on node).
    def path_score(p: Sequence[int]) -> float:
        # Find node with this path.
        for i, tp in enumerate(tree.paths):
            if tp == list(p):
                return tree.nodes[i].logprob
        return -1e9

    path = max(paths, key=path_score)
    draft = torch.tensor([path], device=prompt_ids.device, dtype=prompt_ids.dtype)
    inp = torch.cat([prompt_ids, draft], dim=1)
    t0 = time.perf_counter()
    out = target(input_ids=inp)
    logits = out.logits[0]
    plen = prompt_ids.shape[1]
    accepted: List[int] = []
    for i, tok in enumerate(path):
        pred = int(torch.argmax(logits[plen - 1 + i]))
        if pred != tok:
            accepted.append(pred)
            return accepted, pred, (time.perf_counter() - t0) * 1000.0
        accepted.append(tok)
    bonus = int(torch.argmax(logits[plen - 1 + len(path)]))
    accepted.append(bonus)
    return accepted, bonus, (time.perf_counter() - t0) * 1000.0


@torch.inference_mode()
def incremental_decode_step(
    target: PreTrainedModel, prompt_ids: torch.Tensor
) -> Tuple[int, float]:
    if prompt_ids.dim() == 1:
        prompt_ids = prompt_ids.unsqueeze(0)
    t0 = time.perf_counter()
    out = target(input_ids=prompt_ids.to(next(target.parameters()).device))
    tok = int(torch.argmax(out.logits[0, -1]))
    return tok, (time.perf_counter() - t0) * 1000.0
