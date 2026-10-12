"""按风险维度分层选股的 executable NAV 引擎（T-027，研究用；不修改 nav_engine.py / nav_turnover.py）.

分层选股：决策日在通过选股池与近期成交过滤后的候选里，按指定维度把候选分组，
每组名额 = top_n × 该组候选数 / 候选总数（最大余数法取整，名额合计恰为 top_n），组内按分数从高到低取；
组合仍等权。效果：组合在这些维度上的构成与候选池一致，去掉模型在行业 / 市值 / beta 上的整体偏离。

维度：
- 类别型（如申万一级 ``sw_l1``）：直接用作组别；
- 数值型（如 ``log_circ_market_cap``、``beta_252d``）：在当期候选中按排名等分成 ``n_quantiles`` 组；
- 多个维度时取各维度组别的组合（如 行业 × 市值五分位）；缺失值单独成组（"NA"）。
``dims`` 为空时与 TurnoverControlledNavEngine（不设换手控制时即 ExecutableNavEngine）逐笔相同。
成交、成本、涨跌停、停牌、现金、退市处理、汇总全部沿用父类。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quantmind.execution.nav_engine import NavConfig
from quantmind.execution.nav_turnover import TurnoverControlledNavEngine


def stratum_labels(x: pd.DataFrame, dims: tuple[str, ...], n_quantiles: int) -> pd.Series:
    """x：index = 候选（按分数从高到低），列 = 维度。返回每只票的组别标签。"""
    parts = []
    for d in dims:
        col = x[d]
        if pd.api.types.is_numeric_dtype(col):
            r = col.rank(method="first")
            n_ok = int(col.notna().sum())
            q = np.floor((r - 1) * n_quantiles / max(n_ok, 1))
            lab = pd.Series(np.where(q.isna(), "NA", "q" + q.fillna(-1).astype(int).astype(str)), index=x.index)
        else:
            lab = col.astype(object).where(col.notna(), "NA").astype(str)
        parts.append(lab)
    out = parts[0]
    for p in parts[1:]:
        out = out + "|" + p
    return out


def stratum_quotas(labels: pd.Series, n: int) -> pd.Series:
    """名额 = n × 组内候选数 / 候选总数，最大余数法取整（余数相同按组名排序），合计恰为 n。"""
    counts = labels.value_counts().sort_index()
    raw = counts * n / counts.sum()
    quota = np.floor(raw).astype(int)
    rem = int(n - quota.sum())
    frac = (raw - quota).sort_values(ascending=False, kind="mergesort")
    quota.loc[frac.index[:rem]] += 1
    return quota


class StratifiedNavEngine(TurnoverControlledNavEngine):
    def __init__(self, preds: pd.DataFrame, prices: pd.DataFrame, config: NavConfig,
                 amihud: pd.DataFrame | None = None, *,
                 exposures: pd.DataFrame | None = None, dims: tuple[str, ...] = (), n_quantiles: int = 5) -> None:
        dims = tuple(dims)
        if dims:
            if exposures is None:
                raise ValueError("dims given but exposures is None")
            missing = [d for d in dims if d not in exposures.columns]
            if missing:
                raise ValueError(f"exposures lacks columns {missing}")
        if n_quantiles < 2:
            raise ValueError("n_quantiles must be >= 2")
        super().__init__(preds, prices, config, amihud=amihud)
        self.dims = dims
        self.n_quantiles = n_quantiles
        self.exposures = None
        if dims:
            e = exposures[list(dims)].copy()
            lv = e.index.names.index("as_of")
            e.index = e.index.set_levels(pd.to_datetime(e.index.levels[lv]).normalize(), level=lv)
            self.exposures = e.sort_index()
        self.strata_log: list[dict] = []

    def _exposures_at(self, as_of: pd.Timestamp, names: list[str]) -> pd.DataFrame:
        try:
            x = self.exposures.xs(as_of, level="as_of")
        except KeyError:
            x = self.exposures.iloc[0:0].droplevel("as_of")
        return x.reindex(names)

    def _select(self, ranked: list[str], as_of: pd.Timestamp) -> list[str]:
        n = self.cfg.top_n
        if not self.dims or len(ranked) <= n:
            return super()._select(ranked, as_of)
        lab = stratum_labels(self._exposures_at(as_of, ranked), self.dims, self.n_quantiles)
        quota = stratum_quotas(lab, n)
        taken = dict.fromkeys(quota.index, 0)
        chosen = []
        for t in ranked:
            g = lab[t]
            if taken[g] < quota[g]:
                chosen.append(t)
                taken[g] += 1
        top = set(ranked[:n])
        self.strata_log.append({"as_of": str(as_of.date()), "n_candidates": len(ranked), "n_groups": int(len(quota)),
                                "n_na_candidates": int(lab.str.contains("NA").sum()),
                                "n_swapped_vs_plain_top": len(top - set(chosen))})
        return chosen


__all__ = ["StratifiedNavEngine", "stratum_labels", "stratum_quotas"]
