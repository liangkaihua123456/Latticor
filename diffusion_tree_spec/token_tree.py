"""Token tree utilities: lattice → SpecInfer-style tree + causal prune."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import torch


@dataclass
class TreeNode:
    token_id: int
    parent: Optional[int]
    depth: int
    logprob: float = 0.0
    children: List[int] = field(default_factory=list)


@dataclass
class TokenTree:
    """Flat representation of a SpecInfer-style token tree."""

    nodes: List[TreeNode]
    # For each node index, the full token path from root (inclusive).
    paths: List[List[int]]

    @property
    def size(self) -> int:
        return len(self.nodes)

    @property
    def depth(self) -> int:
        if not self.nodes:
            return 0
        return 1 + max(n.depth for n in self.nodes)

    def leaf_paths(self) -> List[List[int]]:
        child_parents = {c for n in self.nodes for c in n.children}
        return [self.paths[i] for i in range(len(self.nodes)) if i not in child_parents]


def build_tree_from_lattice(
    lattice_topk: Sequence[Sequence[Tuple[int, float]]],
    *,
    max_width: int = 3,
    max_nodes: int = 64,
    prune_margin: float = 0.0,
) -> TokenTree:
    """Build a prefix-sharing token tree from a per-position top-k lattice.

    ``lattice_topk[pos]`` is a list of ``(token_id, logprob)`` candidates for
    draft position ``pos`` (0-based relative to the speculative block).

    Causal-consistency pruning keeps, at each depth, only the top-``max_width``
    extensions of surviving prefixes (SpecInfer expansion + DiffuSpec-style
    consistency, but retaining a full tree for parallel verification).
    """
    if not lattice_topk:
        return TokenTree(nodes=[], paths=[])

    # Virtual root (no token); children are depth-0 tokens.
    nodes: List[TreeNode] = [TreeNode(token_id=-1, parent=None, depth=-1, logprob=0.0)]
    paths: List[List[int]] = [[]]
    frontier = [0]

    for depth, candidates in enumerate(lattice_topk):
        if not candidates or not frontier:
            break
        # Sort candidates by logprob descending.
        cands = sorted(candidates, key=lambda x: x[1], reverse=True)
        if prune_margin > 0 and cands:
            best = cands[0][1]
            cands = [c for c in cands if c[1] >= best - prune_margin]
        cands = cands[: max(1, max_width)]

        new_frontier: List[int] = []
        scored: List[Tuple[float, int, int, float]] = []  # score, parent, tok, lp

        for parent_idx in frontier:
            parent_score = nodes[parent_idx].logprob
            for tok, lp in cands:
                scored.append((parent_score + lp, parent_idx, tok, lp))

        scored.sort(key=lambda x: x[0], reverse=True)
        # Limit branching at this depth.
        width_budget = max_width * max(1, len(frontier))
        scored = scored[: min(width_budget, max_width * 4)]

        for score, parent_idx, tok, lp in scored:
            if len(nodes) >= max_nodes:
                break
            idx = len(nodes)
            nodes.append(
                TreeNode(token_id=tok, parent=parent_idx, depth=depth, logprob=score)
            )
            nodes[parent_idx].children.append(idx)
            paths.append(paths[parent_idx] + [tok])
            new_frontier.append(idx)

        frontier = new_frontier
        if len(nodes) >= max_nodes:
            break

    # Drop virtual root from exported paths used for verification sequences,
    # but keep it for topology. Verification uses nodes[1:].
    return TokenTree(nodes=nodes, paths=paths)


def tree_to_packed_tensors(
    tree: TokenTree,
    prompt_len: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
    """Pack tree nodes (excluding virtual root) for tree-attention verify.

    Returns
    -------
    input_ids : (1, prompt_len + n_draft)
    position_ids : (1, prompt_len + n_draft)
    attn_mask : (1, 1, q_len, q_len) causal + tree topology for draft region
    node_indices : mapping from packed draft column → tree node index
    """
    # Skip virtual root.
    draft_nodes = list(range(1, len(tree.nodes)))
    n = len(draft_nodes)
    if n == 0:
        raise ValueError("empty tree")

    draft_tokens = torch.tensor(
        [tree.nodes[i].token_id for i in draft_nodes], dtype=torch.long, device=device
    )
    # Placeholder prompt filled by caller; we only build draft piece + mask helpers.
    return draft_tokens, draft_nodes


def build_tree_attention_mask(
    tree: TokenTree,
    prompt_len: int,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build packed draft ids, position ids and additive attention bias.

    Layout: [prompt tokens | draft node tokens in node-index order (skip root)].
    A draft node may attend to the full prompt and to its ancestors' draft tokens.
    """
    draft_node_ids = list(range(1, len(tree.nodes)))
    n_draft = len(draft_node_ids)
    q_len = prompt_len + n_draft

    draft_token_ids = torch.tensor(
        [tree.nodes[i].token_id for i in draft_node_ids], dtype=torch.long, device=device
    )

    # Map tree node index → packed position.
    node_to_pos = {nid: prompt_len + j for j, nid in enumerate(draft_node_ids)}

    pos_ids = torch.zeros(q_len, dtype=torch.long, device=device)
    # Prompt positions 0..prompt_len-1 set by caller; draft positions use depth+prompt_len.
    for j, nid in enumerate(draft_node_ids):
        pos_ids[prompt_len + j] = prompt_len + tree.nodes[nid].depth

    # Additive mask: 0 = allow, -inf = block.
    mask = torch.zeros(q_len, q_len, dtype=dtype, device=device)
    neg = torch.finfo(dtype).min

    # Prompt region: standard causal.
    for i in range(prompt_len):
        if i + 1 < prompt_len:
            mask[i, i + 1 : prompt_len] = neg
        # Prompt cannot see draft.
        if n_draft:
            mask[i, prompt_len:] = neg

    # Draft region.
    for j, nid in enumerate(draft_node_ids):
        row = prompt_len + j
        # Cannot see other non-ancestor draft nodes.
        allowed = set()
        # All prompt positions.
        allowed.update(range(prompt_len))
        # Ancestors (and self).
        cur: Optional[int] = nid
        while cur is not None and cur != 0:
            allowed.add(node_to_pos[cur])
            cur = tree.nodes[cur].parent
        for col in range(q_len):
            if col not in allowed:
                mask[row, col] = neg

    return draft_token_ids, pos_ids, mask.unsqueeze(0).unsqueeze(0)
