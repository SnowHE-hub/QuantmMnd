"""排序 / IC 损失：一个 batch 由若干完整交易日组成，所有统计只在同一日期内计算."""
from __future__ import annotations

import torch


def _segments(day: torch.Tensor) -> tuple[torch.Tensor, int]:
    """day 为任意整数日期编号 → 0..G−1 的连续组号."""
    _, inv = torch.unique(day, return_inverse=True)
    return inv, int(inv.max()) + 1


def daily_ic_loss(pred: torch.Tensor, y: torch.Tensor, day: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """−（各交易日内 pred 与 y 的 Pearson 相关的均值）。y 通常已是当日 gauss-rank，故等价于对秩的 IC。

    pred 可为 (N,) 或 (N, K)（K 个集成成员）；后者返回 K 个成员、各交易日相关的总均值。
    """
    g, G = _segments(day)
    pred, y = pred.float(), y.float()
    if pred.ndim == 2:
        y = y[:, None].expand_as(pred)
    shape = (G, *pred.shape[1:])
    z = lambda: torch.zeros(shape, device=pred.device)  # noqa: E731
    n = z().index_add_(0, g, torch.ones_like(pred))
    mp = z().index_add_(0, g, pred) / n
    my = z().index_add_(0, g, y) / n
    dp, dy = pred - mp[g], y - my[g]
    cov = z().index_add_(0, g, dp * dy)
    vp = z().index_add_(0, g, dp * dp)
    vy = z().index_add_(0, g, dy * dy)
    corr = cov / torch.sqrt(vp * vy + eps)
    return -corr.mean()


def listnet_loss(pred: torch.Tensor, y: torch.Tensor, day: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """ListNet top-1：各交易日内 softmax(y/τ) 与 softmax(pred) 的交叉熵，日均."""
    g, G = _segments(day)
    pred, y = pred.float(), y.float() / temperature

    def seg_logsumexp(v):
        m = torch.full((G,), -torch.inf, device=v.device).scatter_reduce(0, g, v, reduce="amax")
        s = torch.zeros(G, device=v.device).index_add_(0, g, torch.exp(v - m[g]))
        return m + torch.log(s)

    p_true = torch.exp(y - seg_logsumexp(y)[g])
    logp = pred - seg_logsumexp(pred)[g]
    ce = torch.zeros(G, device=pred.device).index_add_(0, g, -p_true * logp)
    return ce.mean()


def pinball_loss(pred_q: torch.Tensor, y: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
    """pred_q: (N, Q)；y: (N,)."""
    d = y[:, None] - pred_q
    return torch.maximum(quantiles * d, (quantiles - 1) * d).mean()
