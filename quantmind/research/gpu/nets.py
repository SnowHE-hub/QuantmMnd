"""深度模型结构。所有模型输出 [n] 或 [n, k]（TabM 的 k 个成员）；训练时每个成员各自算损失再平均，预测取成员均值.

输入约定（由 dl.py 组装）：
  tab  [n, F_tab]   股票级 gauss-rank（NaN→0）+ 组缺失标记 + 标准化后的市场特征
  seq  [n, L, C]    F6 过去 L 天（含当天），按该折训练统计量标准化，NaN→0
  snap [n, F_snap]  序列融合模型的快照分支输入
  mkt  [n, F_mkt]   标准化后的市场特征（MASTER 门控用）
  day  [n]          批内交易日编号（MASTER 的股间注意力只在同一天内）
"""
from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, d_in: int, d: int = 256, n_blocks: int = 3, dropout: float = 0.1):
        super().__init__()
        layers, w = [], d_in
        for _ in range(n_blocks):
            layers += [nn.Linear(w, d), nn.GELU(), nn.Dropout(dropout)]
            w = d
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(w, 1)

    def forward(self, tab, **_):
        return self.head(self.body(tab)).squeeze(-1)


class TabMNet(nn.Module):
    def __init__(self, d_in: int, k: int = 32, d: int = 256, n_blocks: int = 2, dropout: float = 0.1):
        super().__init__()
        from tabm import TabM
        self.m = TabM.make(n_num_features=d_in, cat_cardinalities=[], d_out=1, k=k, d_block=d, n_blocks=n_blocks,
                           dropout=dropout)

    def forward(self, tab, **_):
        return self.m(tab).squeeze(-1)                     # [n, k]


class _TCN(nn.Module):
    """因果膨胀卷积编码器；取最后一个时间步的表示。"""

    def __init__(self, c_in: int, d: int, n_layers: int, kernel: int = 3, dropout: float = 0.1):
        super().__init__()
        self.inp = nn.Conv1d(c_in, d, 1)
        self.blocks = nn.ModuleList()
        self.pads = []
        for i in range(n_layers):
            dil = 2 ** i
            self.pads.append((kernel - 1) * dil)
            self.blocks.append(nn.Sequential(nn.Conv1d(d, d, kernel, dilation=dil), nn.GELU(), nn.Dropout(dropout)))

    def forward(self, x):                                   # x [n, L, C]
        h = self.inp(x.transpose(1, 2))
        for pad, blk in zip(self.pads, self.blocks, strict=True):
            h = h + blk(nn.functional.pad(h, (pad, 0)))
        return h[:, :, -1]


class SeqFusion(nn.Module):
    """F6 序列 → TCN；快照（F2+F4+F7）→ MLP；拼接 → 头。"""

    def __init__(self, c_in: int, d_snap_in: int, d: int = 64, n_layers: int = 5, d_snap: int = 64, dropout: float = 0.1):
        super().__init__()
        self.enc = _TCN(c_in, d, n_layers, dropout=dropout)
        self.snap = nn.Sequential(nn.Linear(d_snap_in, d_snap), nn.GELU(), nn.Dropout(dropout))
        self.head = nn.Sequential(nn.Linear(d + d_snap, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, 1))

    def forward(self, seq, snap, **_):
        return self.head(torch.cat([self.enc(seq), self.snap(snap)], dim=-1)).squeeze(-1)


class GRUNet(nn.Module):
    def __init__(self, c_in: int, d: int = 64, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.rnn = nn.GRU(c_in, d, n_layers, batch_first=True, dropout=dropout)
        self.head = nn.Linear(d, 1)

    def forward(self, seq, **_):
        out, _ = self.rnn(seq)
        return self.head(out[:, -1]).squeeze(-1)


class ALSTM(nn.Module):
    """Qlib ALSTM：LSTM 输出的注意力池化 + 最后隐状态。"""

    def __init__(self, c_in: int, d: int = 64, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(c_in, d), nn.Tanh())
        self.rnn = nn.LSTM(d, d, n_layers, batch_first=True, dropout=dropout)
        self.att = nn.Sequential(nn.Linear(d, d // 2), nn.Dropout(dropout), nn.Tanh(), nn.Linear(d // 2, 1, bias=False))
        self.head = nn.Linear(2 * d, 1)

    def forward(self, seq, **_):
        out, _ = self.rnn(self.inp(seq))
        w = torch.softmax(self.att(out), dim=1)
        return self.head(torch.cat([out[:, -1], (w * out).sum(1)], dim=-1)).squeeze(-1)


class MASTERNet(nn.Module):
    """MASTER 式：市场引导的特征门控 → 股内时序注意力 → 股间注意力（同一交易日内全部股票）→ 时间聚合 → 头。"""

    def __init__(self, c_in: int, d_mkt: int, d: int = 64, heads: int = 4, dropout: float = 0.1, beta: float = 5.0):
        super().__init__()
        self.beta = beta
        self.gate = nn.Linear(d_mkt, c_in)
        self.inp = nn.Linear(c_in, d)
        self.pos = nn.Parameter(torch.zeros(1, 256, d))
        self.intra = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout, batch_first=True, norm_first=True)
        self.inter = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.tq = nn.Linear(d, d, bias=False)
        self.head = nn.Linear(d, 1)

    def forward(self, seq, mkt, day, **_):
        n, L, _ = seq.shape
        g = seq.shape[-1] * torch.softmax(self.gate(mkt) / self.beta, dim=-1)       # [n, C]
        h = self.inp(seq * g[:, None, :]) + self.pos[:, -L:]
        h = self.intra(h)                                                           # [n, L, d]
        z = h[:, -1]                                                                # 当天表示
        out = torch.empty_like(z)
        for dd in torch.unique(day):                                                # 股间注意力只在同一天内
            idx = torch.nonzero(day == dd, as_tuple=True)[0]
            zi = z[idx][None]
            a, _ = self.inter(zi, zi, zi, need_weights=False)
            out[idx] = self.norm(zi + a)[0]
        hz = h + out[:, None, :]
        lam = torch.softmax(torch.einsum("nld,nd->nl", self.tq(hz), hz[:, -1]), dim=1)
        return self.head((lam[..., None] * hz).sum(1)).squeeze(-1)


ARCHS = {"mlp": MLP, "tabm": TabMNet, "seqfusion": SeqFusion, "gru": GRUNet, "alstm": ALSTM, "master": MASTERNet}
INPUTS = {"mlp": "tab", "tabm": "tab", "seqfusion": "seq+snap", "gru": "seq", "alstm": "seq", "master": "seq+mkt"}
