"""quantmind/models/factor_cnn.py

Inception 风格多分支因子降维网络（FactorCNN）。

背景
----
LGBM v6 的 71 个特征存在大量组内相关性（pe_ttm / earnings_yield 近似共线，
momentum 组内高度相关）。FactorCNN 把特征按经济语义分成 4 组，每组走独立
bottleneck 分支压缩到 8 维，concat 后用两层 MLP 输出截面 alpha score。

架构
----
    价值(9)   → Linear(9,16) → BN → ReLU → Linear(16,8)  ┐
    质量(17)  → Linear(17,16)→ BN → ReLU → Linear(16,8)  │ concat(32)
    动量(9)   → Linear(9,16) → BN → ReLU → Linear(16,8)  │ BN → Linear(32,16)
    技术(36)  → Linear(36,32)→ BN → ReLU → Linear(32,8)  ┘ ReLU → Dropout(0.2)
                                                             Linear(16,1) → score

训练细节
--------
* 损失函数：IC Loss = −cosine_similarity(pred_dm, label_dm)  等价最大化截面 IC
* 优化器：Adam  lr=1e-3  weight_decay=1e-4
* 截面预处理：winsorize(±3) → cross-section zscore → 组内中位数填 NaN
* 滚动训练：扩展窗口，前 n_train_quarters 训练，后 n_val_quarters 验证
* Early stopping：val IC 连续 5 epoch 不改善则停止
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

log = logging.getLogger(__name__)

# ─── 特征分组定义 ──────────────────────────────────────────────────────────────

# 价值 / 规模（9 维）
VALUE_FEATURES: list[str] = [
    "pe_ttm",
    "pb",
    "ps_ttm",
    "book_to_market",
    "earnings_yield",
    "dividend_yield_ttm",
    "log_market_cap",
    "log_circ_market_cap",
    "fcf_yield",
]

# 质量 / 成长 / 现金流（17 维）
QUALITY_FEATURES: list[str] = [
    "roe_ttm",
    "roa_ttm",
    "gross_margin",
    "net_margin",
    "debt_to_assets",
    "current_ratio",
    "asset_turnover",
    "equity_multiplier",
    "revenue_yoy",
    "operating_profit_yoy",
    "net_profit_yoy",
    "quarterly_revenue_yoy",
    "accruals",
    "ocf_to_revenue_ttm",
    "size_rank",
    "earnings_accel_q",
    "revenue_accel_q",
]

# 动量 / 波动 / 反转（9 维）
MOMENTUM_FEATURES: list[str] = [
    "momentum_1m",
    "momentum_3m",
    "momentum_6m",
    "momentum_12m_skip_1m",
    "reversal_1w",
    "volatility_3m",
    "volatility_1y",
    "downside_volatility_3m",
    "max_drawdown_3m",
]

# 技术形态 / 流动性 / 北向 / 融资融券 / 市场级（36 维）
TECHNICAL_FEATURES: list[str] = [
    "amihud_illiquidity",
    "turnover_3m_avg",
    "volume_spike_5_30",
    "rsi_14",
    "bollinger_position",
    "distance_to_52w_high",
    "price_to_52w_low",
    "turnover_acceleration",
    "turnover_rate_quantile",
    "amplitude_quantile",
    "free_float_ratio",
    "north_bound_30d_net_inflow",
    "list_age_years",
    "is_recent_ipo",
    "north_hold_ratio",
    "north_hold_amount",
    "north_hold_ratio_change_20d",
    "north_hold_ratio_change_60d",
    "north_hold_amount_change_20d",
    "north_hold_trend_60d",
    "margin_balance",
    "margin_balance_change_20d",
    "margin_buy_amount_20d",
    "margin_buy_intensity",
    "short_balance_change_20d",
    "short_sell_pressure",
    "margin_short_ratio",
    "beta_252d",
    "beta_60d",
    "relative_strength_vs_csi300_60d",
    "relative_strength_vs_csi300_120d",
    "market_momentum_60d",
    "market_volatility_60d",
    "market_drawdown_60d",
    "relative_strength_vs_csi500_60d",
    "volume_price_corr_20d",
]

# 分组映射（供外部调用）
FACTOR_GROUPS: dict[str, list[str]] = {
    "value":    VALUE_FEATURES,
    "quality":  QUALITY_FEATURES,
    "momentum": MOMENTUM_FEATURES,
    "technical": TECHNICAL_FEATURES,
}

ALL_CNN_FEATURES: list[str] = (
    VALUE_FEATURES + QUALITY_FEATURES + MOMENTUM_FEATURES + TECHNICAL_FEATURES
)

# 维度常量
_N_VALUE     = len(VALUE_FEATURES)      # 9
_N_QUALITY   = len(QUALITY_FEATURES)    # 17
_N_MOMENTUM  = len(MOMENTUM_FEATURES)   # 9
_N_TECHNICAL = len(TECHNICAL_FEATURES)  # 36
_BRANCH_OUT  = 8                        # 每分支输出维度
_CONCAT_DIM  = _BRANCH_OUT * 4          # 32


# ─── 数据预处理 ────────────────────────────────────────────────────────────────

def _cross_section_winsorize(s: pd.Series, clip: float = 3.0) -> pd.Series:
    """组内 MAD Winsorize（先 MAD 缩尾，再 zscore）."""
    med = s.median()
    mad = (s - med).abs().median()
    if mad < 1e-12:
        return s.clip(lower=med - clip, upper=med + clip)
    lower = med - clip * mad
    upper = med + clip * mad
    return s.clip(lower=lower, upper=upper)


def cross_section_zscore(s: pd.Series, winsorize: bool = True,
                         clip: float = 3.0) -> pd.Series:
    """截面 zscore：先 MAD winsorize(±3)，再标准化均值0方差1。

    缺失值保持 NaN（由后续 median-fill 处理）。
    """
    if winsorize:
        s = _cross_section_winsorize(s, clip)
    mu  = s.mean()
    std = s.std(ddof=0)
    if std < 1e-12:
        return pd.Series(0.0, index=s.index, name=s.name)
    return (s - mu) / std


def preprocess_cross_section(df: pd.DataFrame,
                              feature_cols: list[str]) -> pd.DataFrame:
    """对单个截面做：winsorize → zscore → 组内中位数填 NaN。

    Parameters
    ----------
    df          : 单季截面 DataFrame（index=ticker）
    feature_cols: 要处理的特征列
    """
    out = df[feature_cols].copy().astype(float)

    # 按特征分组填 NaN，避免跨组污染
    group_map: dict[str, list[str]] = {}
    for col in feature_cols:
        for g, cols in FACTOR_GROUPS.items():
            if col in cols:
                group_map.setdefault(g, []).append(col)
                break
        else:
            group_map.setdefault("other", []).append(col)

    for g_cols in group_map.values():
        # 组内中位数填 NaN（先 dropna 取中位数，避免 nanmean empty slice 告警）
        for col in g_cols:
            valid_vals = out[col].dropna()
            med = float(valid_vals.median()) if len(valid_vals) > 0 else 0.0
            out[col] = out[col].fillna(med)

    # 截面 winsorize + zscore
    for col in feature_cols:
        out[col] = cross_section_zscore(out[col], winsorize=True)

    return out


def preprocess_panel(
    panel: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = "forward_return_63d",
) -> pd.DataFrame:
    """对整个 panel 逐季做截面预处理，返回 (features + label) DataFrame。

    Panel index：MultiIndex(as_of, ts_code)
    """
    dates = panel.index.get_level_values(0).unique().sort_values()
    chunks: list[pd.DataFrame] = []

    for dt in dates:
        slice_df = panel.loc[dt].copy()
        if label_col not in slice_df.columns:
            continue
        label = slice_df[label_col].astype(float)

        # 过滤 label NaN
        valid = label.notna()
        slice_df = slice_df.loc[valid]
        label    = label.loc[valid]

        # 特征预处理
        avail = [c for c in feature_cols if c in slice_df.columns]
        feat  = preprocess_cross_section(slice_df, avail)

        feat[label_col] = label
        feat.index.name = "ts_code"
        feat["as_of"]   = dt
        chunks.append(feat.reset_index())

    if not chunks:
        return pd.DataFrame()

    return pd.concat(chunks, ignore_index=True)


# ─── 模型定义 ──────────────────────────────────────────────────────────────────

class _Branch(nn.Module):
    """单组因子的 bottleneck 分支：Linear → BN → ReLU → Linear。"""

    def __init__(self, in_dim: int, mid_dim: int = 16,
                 out_dim: int = _BRANCH_OUT) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mid_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FactorCNN(nn.Module):
    """Inception 风格多分支因子网络。

    Parameters
    ----------
    n_value     : 价值因子数量，默认 9
    n_quality   : 质量因子数量，默认 17
    n_momentum  : 动量因子数量，默认 9
    n_technical : 技术因子数量，默认 36
    branch_dim  : 每分支输出维度，默认 8
    dropout     : MLP head dropout 率，默认 0.2

    Forward input order
    -------------------
    x_value, x_quality, x_momentum, x_technical (各组特征张量)
    OR 单个合并张量 x (按 value+quality+momentum+technical 列序拼接)

    Returns
    -------
    Tensor shape (batch,)  截面 alpha score（unbounded real）
    """

    def __init__(
        self,
        n_value:     int = _N_VALUE,
        n_quality:   int = _N_QUALITY,
        n_momentum:  int = _N_MOMENTUM,
        n_technical: int = _N_TECHNICAL,
        branch_dim:  int = _BRANCH_OUT,
        dropout:     float = 0.2,
    ) -> None:
        super().__init__()
        self.n_value     = n_value
        self.n_quality   = n_quality
        self.n_momentum  = n_momentum
        self.n_technical = n_technical
        self.branch_dim  = branch_dim
        concat_dim       = branch_dim * 4

        # 技术因子更宽的 bottleneck（36 → 32 → 8）
        self.branch_value    = _Branch(n_value,     mid_dim=16, out_dim=branch_dim)
        self.branch_quality  = _Branch(n_quality,   mid_dim=16, out_dim=branch_dim)
        self.branch_momentum = _Branch(n_momentum,  mid_dim=16, out_dim=branch_dim)
        self.branch_technical = _Branch(n_technical, mid_dim=32, out_dim=branch_dim)

        # MLP head
        self.head = nn.Sequential(
            nn.BatchNorm1d(concat_dim),
            nn.Linear(concat_dim, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(16, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, n_value + n_quality + n_momentum + n_technical)
            所有因子按 value→quality→momentum→technical 顺序拼接的矩阵
        """
        idx_v = self.n_value
        idx_q = idx_v + self.n_quality
        idx_m = idx_q + self.n_momentum

        x_val  = x[:, :idx_v]
        x_qua  = x[:, idx_v:idx_q]
        x_mom  = x[:, idx_q:idx_m]
        x_tec  = x[:, idx_m:]

        v = self.branch_value(x_val)
        q = self.branch_quality(x_qua)
        m = self.branch_momentum(x_mom)
        t = self.branch_technical(x_tec)

        concat = torch.cat([v, q, m, t], dim=1)   # (batch, 32)
        out    = self.head(concat).squeeze(-1)     # (batch,)
        return out


# ─── 损失函数 ──────────────────────────────────────────────────────────────────

def ic_loss(pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """IC Loss：最大化截面 Pearson 相关（等价最大化 IC）。

    ic_loss = −cosine_similarity(pred − mean(pred), label − mean(label))

    值域：[−1, +1]，最优值 −1（pred 与 label 完全正相关）。
    """
    if pred.shape[0] < 2:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    pred_dm  = pred  - pred.mean()
    label_dm = label - label.mean()
    return -F.cosine_similarity(pred_dm.unsqueeze(0),
                                label_dm.unsqueeze(0), dim=1).squeeze()


# ─── 训练与评估 ───────────────────────────────────────────────────────────────

@dataclass
class FoldMetrics:
    """单折训练结果."""
    train_dates: list
    val_dates:   list
    val_ic:      float          = float("nan")
    best_epoch:  int            = 0
    train_ic:    float          = float("nan")

    def __repr__(self) -> str:
        return (
            f"FoldMetrics(val_ic={self.val_ic:.4f}  train_ic={self.train_ic:.4f}"
            f"  best_epoch={self.best_epoch})"
        )


@dataclass
class CNNTrainResult:
    """train_factor_cnn 的完整返回结果."""
    folds:          list[FoldMetrics] = field(default_factory=list)
    val_ic_mean:    float = float("nan")
    val_ic_std:     float = float("nan")
    val_icir:       float = float("nan")
    model:          Optional[FactorCNN] = None
    feature_cols:   list[str] = field(default_factory=list)


def _compute_ic(pred: np.ndarray, label: np.ndarray) -> float:
    """Spearman IC（截面）。"""
    from scipy.stats import spearmanr
    if len(pred) < 5:
        return float("nan")
    ic, _ = spearmanr(pred, label)
    return float(ic) if not np.isnan(ic) else 0.0


def _panel_to_tensors(
    processed: pd.DataFrame,
    dates: list,
    feature_cols: list[str],
    label_col: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """将处理好的 panel 子集转换为 (X, y) 张量。"""
    sub = processed[processed["as_of"].isin(dates)]
    X = sub[feature_cols].values.astype(np.float32)
    y = sub[label_col].values.astype(np.float32)
    # 去掉 label NaN（已在 preprocess 时过滤，但以防万一）
    valid = ~np.isnan(y) & ~np.any(np.isnan(X), axis=1)
    X, y = X[valid], y[valid]
    return (torch.tensor(X, device=device),
            torch.tensor(y, device=device))


def train_factor_cnn(
    panel: pd.DataFrame,
    label_col: str = "forward_return_63d",
    n_train_quarters: int = 16,
    n_val_quarters:   int = 4,
    epochs:           int = 100,
    batch_size:       int = 512,
    lr:               float = 1e-3,
    weight_decay:     float = 1e-4,
    patience:         int = 5,
    device: str = "cpu",
) -> CNNTrainResult:
    """滚动窗口训练 FactorCNN，返回验证集 IC 统计。

    Parameters
    ----------
    panel            : alpha_panel_v4（MultiIndex as_of × ts_code）
    label_col        : 标签列名，默认 forward_return_63d
    n_train_quarters : 每折训练季度数
    n_val_quarters   : 每折验证季度数
    epochs           : 每折最大 epoch 数
    batch_size       : DataLoader batch size
    lr               : Adam 学习率
    weight_decay     : Adam L2 正则
    patience         : Early stopping patience（val IC 无改善 epoch 数）
    device           : 'cuda' 或 'cpu'

    Returns
    -------
    CNNTrainResult
    """
    dev = torch.device(device)
    log.info("device=%s", dev)

    # 确认可用特征（面板里可能没有所有 71 个特征）
    avail = [c for c in ALL_CNN_FEATURES if c in panel.columns]
    log.info("可用特征: %d / %d", len(avail), len(ALL_CNN_FEATURES))

    # 重建特征分组（只含 panel 里实际存在的）
    grp_value    = [c for c in VALUE_FEATURES    if c in avail]
    grp_quality  = [c for c in QUALITY_FEATURES  if c in avail]
    grp_momentum = [c for c in MOMENTUM_FEATURES if c in avail]
    grp_technical = [c for c in TECHNICAL_FEATURES if c in avail]
    feature_cols = grp_value + grp_quality + grp_momentum + grp_technical

    log.info("分支维度  value=%d  quality=%d  momentum=%d  technical=%d",
             len(grp_value), len(grp_quality), len(grp_momentum), len(grp_technical))

    # 整体预处理（截面 zscore 在每个季度内独立完成）
    log.info("预处理 panel...")
    processed = preprocess_panel(panel, feature_cols, label_col=label_col)
    if processed.empty:
        log.error("processed panel 为空，检查 label_col=%s", label_col)
        return CNNTrainResult()

    # 时序顺序
    dates = sorted(processed["as_of"].unique())
    n_dates = len(dates)
    log.info("可用季度数: %d", n_dates)

    if n_dates < n_train_quarters + n_val_quarters:
        log.warning("季度数 %d < n_train+n_val=%d",
                    n_dates, n_train_quarters + n_val_quarters)
        return CNNTrainResult()

    # ─── 滚动训练 ────────────────────────────────────────────────────────────
    folds: list[FoldMetrics] = []
    best_global_model: Optional[FactorCNN] = None
    best_global_ic: float = -float("inf")

    for fold_start in range(0, n_dates - n_train_quarters - n_val_quarters + 1,
                            n_val_quarters):
        train_dates = dates[fold_start: fold_start + n_train_quarters]
        val_dates   = dates[fold_start + n_train_quarters:
                            fold_start + n_train_quarters + n_val_quarters]

        if len(val_dates) < n_val_quarters:
            break

        log.info("Fold  train=%s→%s  val=%s→%s",
                 train_dates[0].strftime("%Y-%m-%d"),
                 train_dates[-1].strftime("%Y-%m-%d"),
                 val_dates[0].strftime("%Y-%m-%d"),
                 val_dates[-1].strftime("%Y-%m-%d"))

        X_tr, y_tr = _panel_to_tensors(processed, train_dates,
                                        feature_cols, label_col, dev)
        X_va, y_va = _panel_to_tensors(processed, val_dates,
                                        feature_cols, label_col, dev)

        if X_tr.shape[0] == 0 or X_va.shape[0] == 0:
            log.warning("  空数据，跳过本折")
            continue

        # 重新实例化（每折独立）
        model = FactorCNN(
            n_value=len(grp_value),
            n_quality=len(grp_quality),
            n_momentum=len(grp_momentum),
            n_technical=len(grp_technical),
        ).to(dev)

        optim    = torch.optim.Adam(model.parameters(),
                                    lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=epochs, eta_min=lr * 0.01
        )

        train_ds = TensorDataset(X_tr, y_tr)
        loader   = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        best_val_ic    = -float("inf")
        no_improve_cnt = 0
        best_state     = None
        best_epoch     = 0

        for epoch in range(1, epochs + 1):
            # ── train ──
            model.train()
            for xb, yb in loader:
                optim.zero_grad()
                pred = model(xb)
                loss = ic_loss(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
            scheduler.step()

            # ── val IC ──
            model.eval()
            with torch.no_grad():
                val_pred = model(X_va).cpu().numpy()
                val_lbl  = y_va.cpu().numpy()

            # Per-quarter Spearman IC
            val_ics: list[float] = []
            val_sub = processed[processed["as_of"].isin(val_dates)]
            valid_mask = (~np.isnan(val_lbl) &
                          ~np.any(np.isnan(X_va.cpu().numpy()), axis=1))
            val_pred_f = val_pred[valid_mask]
            val_lbl_f  = val_lbl[valid_mask]
            # 简单聚合 IC（全 val 截面）
            val_ic = _compute_ic(val_pred_f, val_lbl_f)

            if val_ic > best_val_ic:
                best_val_ic    = val_ic
                no_improve_cnt = 0
                best_epoch     = epoch
                best_state     = {k: v.clone()
                                  for k, v in model.state_dict().items()}
            else:
                no_improve_cnt += 1

            if no_improve_cnt >= patience:
                log.info("  Early stop @ epoch %d  best_val_ic=%.4f",
                         epoch, best_val_ic)
                break

        # 恢复最佳权重计算 train IC
        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        with torch.no_grad():
            tr_pred = model(X_tr).cpu().numpy()
        tr_ic = _compute_ic(tr_pred, y_tr.cpu().numpy())

        fm = FoldMetrics(
            train_dates=train_dates,
            val_dates=val_dates,
            val_ic=best_val_ic,
            best_epoch=best_epoch,
            train_ic=tr_ic,
        )
        folds.append(fm)
        log.info("  %s  val_ic=%.4f  train_ic=%.4f",
                 fm, fm.val_ic, fm.train_ic)

        if best_val_ic > best_global_ic:
            best_global_ic    = best_val_ic
            best_global_model = model

    # ─── 汇总 ────────────────────────────────────────────────────────────────
    valid_ics = [f.val_ic for f in folds if not np.isnan(f.val_ic)]
    if valid_ics:
        ic_mean = float(np.mean(valid_ics))
        ic_std  = float(np.std(valid_ics, ddof=1)) if len(valid_ics) > 1 else 0.0
        icir    = ic_mean / ic_std if ic_std > 1e-9 else float("nan")
    else:
        ic_mean = ic_std = icir = float("nan")

    result = CNNTrainResult(
        folds=folds,
        val_ic_mean=ic_mean,
        val_ic_std=ic_std,
        val_icir=icir,
        model=best_global_model,
        feature_cols=feature_cols,
    )
    log.info("─── 训练完毕 ───────────────────────────────────────")
    log.info("val_ic_mean=%.4f  val_ic_std=%.4f  ICIR=%.3f",
             ic_mean, ic_std, icir)
    return result


# ─── Ensemble ──────────────────────────────────────────────────────────────────

def ensemble_scores(
    lgbm_score: pd.Series,
    cnn_score:  pd.Series,
    lgbm_weight: float = 0.7,
    cnn_weight:  float = 0.3,
) -> pd.Series:
    """Rank-based ensemble（规避量纲差异）。

    Parameters
    ----------
    lgbm_score, cnn_score : 同截面、同 index 的评分 Series
    lgbm_weight, cnn_weight : 权重（不要求归一化，内部自动归一化）

    Returns
    -------
    pd.Series  ensemble rank score ∈ [0, 1]
    """
    total = lgbm_weight + cnn_weight
    w_lgbm = lgbm_weight / total
    w_cnn  = cnn_weight  / total

    # 对齐 index
    common = lgbm_score.index.intersection(cnn_score.index)
    lgbm_r = lgbm_score.loc[common].rank(pct=True, na_option="bottom")
    cnn_r  = cnn_score.loc[common].rank(pct=True, na_option="bottom")

    return (w_lgbm * lgbm_r + w_cnn * cnn_r).rename("ensemble_score")


# ─── 推理 API ──────────────────────────────────────────────────────────────────

def predict_cnn(
    model: FactorCNN,
    panel_slice: pd.DataFrame,
    feature_cols: list[str],
    device: str = "cpu",
) -> pd.Series:
    """对单个截面做推理，返回每只股票的 alpha score Series。

    Parameters
    ----------
    model         : 训练好的 FactorCNN
    panel_slice   : 单季截面 DataFrame（index=ticker，columns⊇feature_cols）
    feature_cols  : 模型输入特征列表（顺序必须与训练时一致）

    注：若某行所有特征均为 NaN（无任何有效数据），则对应输出保持 NaN。
    """
    dev = torch.device(device)
    scores = pd.Series(np.nan, index=panel_slice.index, name="cnn_score")

    # 原始 NaN 检测（预处理会填充中位数，先记录原始全-NaN 行）
    raw = panel_slice[feature_cols].astype(float)
    all_nan_rows = raw.isnull().all(axis=1)   # 所有特征均 NaN 的行

    # 至少有一个特征有值的行才参与推理
    candidate_mask = ~all_nan_rows
    if candidate_mask.sum() == 0:
        return scores  # 全部为全-NaN 行

    feat  = preprocess_cross_section(panel_slice.loc[candidate_mask], feature_cols)
    X     = feat[feature_cols].values.astype(np.float32)
    valid = ~np.any(np.isnan(X), axis=1)

    if valid.sum() == 0:
        return scores

    model.eval()
    with torch.no_grad():
        xt  = torch.tensor(X[valid], device=dev)
        out = model(xt).cpu().numpy()

    # 把推理结果写回 candidate_mask 对应的位置
    cand_idx = panel_slice.index[candidate_mask]
    sub_scores = pd.Series(np.nan, index=cand_idx)
    sub_scores.iloc[np.where(valid)[0]] = out
    scores.loc[cand_idx] = sub_scores.values
    return scores


# ─── Regime-aware 训练 & 推理 ──────────────────────────────────────────────────

def train_regime_aware_cnn(
    panel: pd.DataFrame,
    regime_map: Dict[pd.Timestamp, str],
    *,
    label_col: str = "forward_return_63d",
    n_train_quarters: int = 8,
    n_val_quarters: int = 2,
    epochs: int = 60,
    batch_size: int = 256,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    patience: int = 10,
    device: str = "cpu",
    min_regime_quarters: int = 4,
) -> Dict[str, CNNTrainResult]:
    """按 HMM regime 分组，分别训练专用 FactorCNN 模型。

    Parameters
    ----------
    panel : pd.DataFrame
        MultiIndex (date, ticker) 面板，与 ``train_factor_cnn`` 格式相同。
    regime_map : dict[pd.Timestamp, str]
        季度截面日期 → regime 标签（'bull' | 'neutral' | 'bear'）。
        由 ``RegimeHMM.predict_regime()`` 批量调用生成。
    min_regime_quarters : int
        某 regime 可用季度数低于此阈值时跳过训练（样本太少）。
        默认 4（需至少 n_val_quarters + 2 个有效季度）。
    其余参数与 ``train_factor_cnn`` 相同。

    Returns
    -------
    dict[str, CNNTrainResult]
        Keys 为 'bull' / 'neutral' / 'bear'（仅包含成功训练的 regime）。
        值为 CNNTrainResult，可从 ``.model`` 取出 FactorCNN 用于推理。

    Examples
    --------
    >>> regime_map = {pd.Timestamp('2022-03-31'): 'bull', ...}
    >>> results = train_regime_aware_cnn(panel, regime_map)
    >>> for r, res in results.items():
    ...     print(f"{r}: val_IC={res.val_ic_mean:.4f}")
    """
    # 面板中所有季度日期（排序）
    panel_dates = (
        panel.index.get_level_values(0).unique().sort_values()
        if isinstance(panel.index, pd.MultiIndex)
        else panel.index.unique().sort_values()
    )

    # 统一 key 类型为 pd.Timestamp，label 小写
    regime_map_ts: Dict[pd.Timestamp, str] = {
        pd.Timestamp(k): str(v).lower() for k, v in regime_map.items()
    }

    results: Dict[str, CNNTrainResult] = {}

    for regime in ["bull", "neutral", "bear"]:
        # 筛选属于该 regime 的季度日期
        regime_dates: List[pd.Timestamp] = [
            d for d in panel_dates if regime_map_ts.get(d, "") == regime
        ]

        if len(regime_dates) < min_regime_quarters:
            log.warning(
                "train_regime_aware_cnn: '%s' regime 仅 %d 季度 (< %d)，跳过",
                regime, len(regime_dates), min_regime_quarters,
            )
            continue

        log.info(
            "train_regime_aware_cnn: 训练 '%s' 模型，%d 季度截面",
            regime, len(regime_dates),
        )

        # 过滤面板至当前 regime
        regime_panel = panel.loc[
            panel.index.get_level_values(0).isin(regime_dates)
        ]

        # 动态调整窗口大小，避免超出样本
        n_total = len(regime_dates)
        n_val   = min(n_val_quarters,   max(1, n_total // 5))
        n_train = min(n_train_quarters, n_total - n_val)

        try:
            result = train_factor_cnn(
                panel=regime_panel,
                label_col=label_col,
                n_train_quarters=n_train,
                n_val_quarters=n_val,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                weight_decay=weight_decay,
                patience=patience,
                device=device,
            )
            results[regime] = result
            log.info(
                "train_regime_aware_cnn: '%s' 完成 — val_IC=%.4f  ICIR=%.3f",
                regime, result.val_ic_mean, result.val_icir,
            )
        except Exception as exc:  # noqa: BLE001
            log.error(
                "train_regime_aware_cnn: '%s' 训练失败 — %s", regime, exc
            )

    if not results:
        log.error("train_regime_aware_cnn: 所有 regime 均训练失败，返回空字典")
    else:
        log.info(
            "train_regime_aware_cnn: 完成 %d/3 regime 模型 — %s",
            len(results), list(results.keys()),
        )

    return results


def predict_cnn_regime_aware(
    models: Dict[str, FactorCNN],
    regime: str,
    panel_slice: pd.DataFrame,
    feature_cols: List[str],
    device: str = "cpu",
    fallback_regime: str = "neutral",
) -> pd.Series:
    """使用 regime 对应的专用 CNN 模型做截面推理。

    Parameters
    ----------
    models : dict[str, FactorCNN]
        从 ``train_regime_aware_cnn()`` 结果提取的模型字典，例如：
        ``{r: res.model for r, res in results.items()}``
    regime : str
        当前 HMM regime（'bull' | 'neutral' | 'bear'，大小写不敏感）。
    panel_slice : pd.DataFrame
        单季截面（index=ticker，columns ⊇ feature_cols）。
    feature_cols : list[str]
        与训练时完全一致的特征列顺序。
    fallback_regime : str
        当 ``regime`` 无对应模型时的回退策略。默认 'neutral'。

    Returns
    -------
    pd.Series  index=ticker，values=cnn_score（alpha rank score）
    """
    regime_lower = regime.lower()

    if regime_lower not in models:
        log.warning(
            "predict_cnn_regime_aware: '%s' 无对应模型，回退到 '%s'",
            regime, fallback_regime,
        )
        regime_lower = fallback_regime.lower()

    if regime_lower not in models:
        # 最后保底：取可用模型中的第一个
        regime_lower = next(iter(models))
        log.warning(
            "predict_cnn_regime_aware: 回退 regime 也无模型，使用 '%s'",
            regime_lower,
        )

    return predict_cnn(
        model=models[regime_lower],
        panel_slice=panel_slice,
        feature_cols=feature_cols,
        device=device,
    )


__all__ = [
    "ALL_CNN_FEATURES",
    "CNNTrainResult",
    "FACTOR_GROUPS",
    "FactorCNN",
    "FoldMetrics",
    "MOMENTUM_FEATURES",
    "QUALITY_FEATURES",
    "TECHNICAL_FEATURES",
    "VALUE_FEATURES",
    "cross_section_zscore",
    "ensemble_scores",
    "ic_loss",
    "predict_cnn",
    "predict_cnn_regime_aware",
    "preprocess_cross_section",
    "preprocess_panel",
    "train_factor_cnn",
    "train_regime_aware_cnn",
]
