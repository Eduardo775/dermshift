"""Losses, always evaluated in fp32 outside autocast.

:class:`HybridLoss` = lambda_d * soft Dice + lambda_r * top-K margin ranking loss, where the ranking term is
mean over (i in S+, j in S-) of max(0, p_j - p_i + m); S+ are the K lesion pixels with the lowest probability,
S- the K background pixels with the highest, K = min(floor(f * H * W), n_pos, n_neg), each set subsampled to
at most K' elements. Computed per image and averaged over the batch. For the cascade,
L = alpha * L(coarse) + (1 - alpha) * L(final).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_dice_loss(p: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-image soft Dice loss. p, y: (B, N)."""
    inter = (p * y).sum(1)
    return 1.0 - (2.0 * inter + eps) / (p.sum(1) + y.sum(1) + eps)


def topk_rank_loss(p: torch.Tensor, y: torch.Tensor, fraction: float, margin: float, max_pairs: int,
                   generator: torch.Generator | None = None, chunk: int = 8) -> torch.Tensor:
    """Per-image top-K margin rank loss, vectorised over the batch (no host synchronisation).
    p, y: (B, N) -> (B,).

    For image b: K_b = min(floor(f*N), n_pos_b, n_neg_b); S+ = the K_b lesion pixels with the
    lowest probability, S- = the K_b background pixels with the highest probability; if
    K_b > K' each set is replaced by a uniform random subset of size K'; the loss is
    mean_{i in S+, j in S-} max(0, P_j - P_i + m). Images without positives or negatives give 0.
    Identical to the per-image loop when no subsampling occurs (tested); with subsampling it
    draws the same distribution of subsets."""
    B, N = p.shape
    k_base = max(1, int(fraction * N))
    pos = y > 0.5
    n_pos = pos.sum(1)
    K = torch.clamp(torch.minimum(n_pos, N - n_pos), max=k_base)                  # (B,)
    big = torch.finfo(p.dtype).max
    hard_pos = torch.topk(torch.where(pos, p, p.new_full((), big)), k_base, dim=1, largest=False).values
    hard_neg = torch.topk(torch.where(~pos, p, p.new_full((), -big)), k_base, dim=1, largest=True).values
    valid = torch.arange(k_base, device=p.device)[None, :] < K[:, None]            # (B, k_base)
    if k_base > max_pairs:
        def pick(vals):
            keys = torch.rand(B, k_base, device=p.device, generator=generator).masked_fill(~valid, 2.0)
            idx = keys.topk(max_pairs, dim=1, largest=False).indices                # valid entries first, uniformly
            return vals.gather(1, idx), valid.gather(1, idx)
        hard_pos, v_pos = pick(hard_pos)
        hard_neg, v_neg = pick(hard_neg)
    else:
        v_pos = v_neg = valid
    hard_pos = torch.where(v_pos, hard_pos, torch.zeros_like(hard_pos))
    hard_neg = torch.where(v_neg, hard_neg, torch.zeros_like(hard_neg))
    out = []
    for s in range(0, B, chunk):
        m = v_pos[s:s + chunk, :, None] & v_neg[s:s + chunk, None, :]
        d = F.relu(hard_neg[s:s + chunk, None, :] - hard_pos[s:s + chunk, :, None] + margin) * m
        out.append(d.sum((1, 2)) / m.sum((1, 2)).clamp(min=1))
    return torch.cat(out)


class HybridLoss(nn.Module):
    def __init__(self, lambda_dice=0.8, lambda_rank=0.3, alpha=0.4, topk_fraction=0.1, margin=0.3,
                 max_pairs_per_side=2048, eps=1e-6, use_rank=True):
        super().__init__()
        self.ld, self.lr, self.alpha = lambda_dice, lambda_rank, alpha
        self.f, self.m, self.kmax, self.eps = topk_fraction, margin, max_pairs_per_side, eps
        self.use_rank = use_rank

    def single(self, logits: torch.Tensor, y: torch.Tensor) -> dict:
        p = torch.sigmoid(logits.float()).flatten(1)
        y = y.float().flatten(1)
        dice = soft_dice_loss(p, y, self.eps).mean()
        if self.use_rank:
            rank = topk_rank_loss(p, y, self.f, self.m, self.kmax).mean()
            total = self.ld * dice + self.lr * rank
        else:
            rank = torch.zeros((), device=p.device)
            total = dice
        return {"total": total, "dice": dice.detach(), "rank": rank.detach()}

    def forward(self, out: dict, y: torch.Tensor) -> dict:
        fin = self.single(out["logits"], y)
        if out.get("coarse_logits") is None:
            return {"loss": fin["total"], "dice": fin["dice"], "rank": fin["rank"]}
        co = self.single(out["coarse_logits"], y)
        loss = self.alpha * co["total"] + (1 - self.alpha) * fin["total"]
        return {"loss": loss, "loss_coarse": co["total"].detach(), "loss_ref": fin["total"].detach(),
                "dice": fin["dice"], "rank": fin["rank"]}


def build_loss(cfg: dict) -> nn.Module:
    lc = cfg["loss"]
    if lc["name"] in {"hybrid", "dice"}:
        return HybridLoss(lc["lambda_dice"], lc["lambda_rank"], lc["alpha"], lc["topk_fraction"], lc["margin"],
                          lc["max_pairs_per_side"], lc["eps"], use_rank=(lc["name"] == "hybrid"))
    raise ValueError(f"unknown loss '{lc['name']}'")
