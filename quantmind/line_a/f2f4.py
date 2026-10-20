"""quantmind.line_a.f2f4 — F2 支持域划分与 F4 研究候选召回（03_final_plan_v2 第 5 节）.

F2：原研究池 = 模型输入 meta 的 top1500（PIT 入池 ∩ 按时点 adv20 前 1500，与研究期同一函数）；
    扩展研究域 = PIT 入池但不在前 1500；数据不足域 = 股票特征缺失比例超过阈值。
F4：宽候选 = 同期限 Ridge 前 K 与 ENS 前 K 的并集（只在原研究池）+ 全部持仓 + 事件队列 + 分层抽查样本。
    事件队列与抽查样本只提高研究优先级，不进入任何组合；每只候选都记录进入原因。
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def support_domains(model_root: Path, as_of, *, max_nan_share: float = 0.5) -> pd.DataFrame:
    d = pd.Timestamp(as_of)
    mdir = Path(model_root) / "data/gpu_v1/model/v1"
    m = d.strftime("%Y-%m")
    meta = pd.read_parquet(mdir / f"{m}.meta.parquet")
    meta["as_of"] = pd.to_datetime(meta["as_of"])
    sel = (meta["as_of"] == d).to_numpy()
    if not sel.any():
        raise ValueError(f"no model rows for {d.date()}")
    X = np.load(mdir / f"{m}.X.npy", mmap_mode="r")[np.flatnonzero(sel)]
    nan_share = np.isnan(X.astype(np.float32)).mean(axis=1)
    out = meta.loc[sel, ["ticker", "top1500"]].set_index("ticker")
    out["nan_share"] = nan_share
    out["domain"] = np.where(out["top1500"], "research_pool", "extended")
    out.loc[out["nan_share"] > max_nan_share, "domain"] = "data_insufficient"
    return out


def _sample_seed(as_of) -> int:
    return int(hashlib.sha256(str(pd.Timestamp(as_of).date()).encode()).hexdigest()[:8], 16)


def research_candidates(scores: pd.DataFrame, domains: pd.DataFrame, as_of, *, holdings=frozenset(),
                        event_queue=frozenset(), top_k: int = 150, n_samples: int = 10,
                        strata_col: str | None = None) -> pd.DataFrame:
    """scores：索引 ticker，列为同一期限的各模型分数（例如 ens20、ridge20）；domains：support_domains 的输出。"""
    pool = domains.index[domains["domain"] == "research_pool"]
    sc = scores.reindex(pool)
    reasons: dict[str, list[str]] = {}
    for col in sc.columns:
        top = sc[col].dropna().sort_values(ascending=False).head(top_k).index
        for t in top:
            reasons.setdefault(t, []).append(f"top{top_k}_{col}")
    for t in holdings:
        reasons.setdefault(t, []).append("holding")
    for t in event_queue:
        reasons.setdefault(t, []).append("event_queue")
    rest = [t for t in pool if t not in reasons]
    rng = np.random.default_rng(_sample_seed(as_of))
    if rest and n_samples:
        if strata_col and strata_col in domains.columns:
            strata = domains.loc[rest, strata_col]
            per = max(1, n_samples // max(1, strata.nunique()))
            picks = [t for _, g in strata.groupby(strata) for t in rng.choice(g.index, size=min(per, len(g)),
                                                                                replace=False)]
        else:
            picks = list(rng.choice(rest, size=min(n_samples, len(rest)), replace=False))
        for t in picks:
            reasons.setdefault(t, []).append("stratified_sample")
    out = pd.DataFrame({"reasons": pd.Series(reasons)})
    out["domain"] = domains["domain"].reindex(out.index).fillna("outside_support")
    prio = {"holding": 0, "event_queue": 1}
    out["priority"] = [min([prio.get(r, 2) for r in rs]) for rs in out["reasons"]]
    for col in sc.columns:
        out[col] = scores[col].reindex(out.index)
    return out.sort_values(["priority"] + list(sc.columns[:1]), ascending=[True] + [False])
