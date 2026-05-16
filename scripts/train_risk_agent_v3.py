#!/usr/bin/env python3
"""训练 RiskAgent v3：HMM 市场状态 + 状态条件 GARCH + CVaR + Beta.

升级要点（相比 garch_v2）：
  1. HMM 状态检测    — 3 状态（低波动 / 正常 / 危机），在线识别当前状态
  2. 状态条件波动率  — 每个状态拟合独立的 EWMA 参数，防止 regime-mix 低估风险
  3. CVaR(95%)        — 基于 Student-t 尾部分布，比正态 VaR 更准确
  4. Beta             — 个股相对 CSI300 / CSI500 的滚动 Beta（252日）
  5. 市场级 HMM      — 基于 CSI300 的系统性风险状态，全局感知

运行：
  python scripts/train_risk_agent_v3.py \\
    --prices data/raw/alpha_prices_panel.parquet \\
    --out    models/agents/risk_hmm_v3.pkl
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
_SNAPSHOT_DIR = ROOT / "data" / "snapshots"


# ── HMM 工具 ──────────────────────────────────────────────────────────────────

def _fit_hmm(obs_series: np.ndarray, n_states: int = 3):
    """用观测序列训练高斯 HMM，返回 (model, viterbi_states)."""
    obs = obs_series.reshape(-1, 1)
    try:
        from hmmlearn.hmm import GaussianHMM  # type: ignore
        model = GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=150,
            tol=1e-2,
            random_state=42,
            verbose=False,
        )
        model.fit(obs)
        states = model.predict(obs)
        return model, states
    except ImportError:
        logger.warning("hmmlearn 未安装，使用分位数分级替代")
        q33 = np.percentile(obs_series, 33)
        q67 = np.percentile(obs_series, 67)
        states = np.where(obs_series < q33, 0, np.where(obs_series < q67, 1, 2))
        return None, states


def _ewma_vol(returns: np.ndarray, lam: float = 0.94) -> float:
    """RiskMetrics EWMA 当日年化波动率（最后一步）."""
    var = returns[0] ** 2
    for r in returns[1:]:
        var = lam * var + (1 - lam) * r ** 2
    return float(np.sqrt(var * 252))


def _ewma_vol_by_lambda(returns: np.ndarray, lam: float) -> np.ndarray:
    """返回整条序列的 EWMA 波动率（年化）."""
    n = len(returns)
    var_arr = np.zeros(n)
    var_arr[0] = returns[0] ** 2
    for i in range(1, n):
        var_arr[i] = lam * var_arr[i - 1] + (1 - lam) * returns[i] ** 2
    return np.sqrt(var_arr * 252)


def _cvar_t(returns: np.ndarray, alpha: float = 0.05) -> float:
    """Student-t CVaR（参数法）."""
    from scipy import stats
    try:
        df_fit, loc_fit, scale_fit = stats.t.fit(returns, floc=0)
        var = stats.t.ppf(alpha, df=df_fit, loc=loc_fit, scale=scale_fit)
        cvar = -stats.t.expect(
            lambda x: x,
            args=(df_fit,),
            loc=loc_fit,
            scale=scale_fit,
            lb=-np.inf, ub=var,
        ) / alpha
        return float(max(cvar, 0.0))
    except Exception:
        sorted_ret = np.sort(returns)
        k = max(1, int(len(returns) * alpha))
        return float(-sorted_ret[:k].mean())


def _ols_beta(stock_ret: np.ndarray, mkt_ret: np.ndarray) -> float:
    """OLS Beta = Cov(stock, mkt) / Var(mkt)."""
    common = min(len(stock_ret), len(mkt_ret))
    if common < 30:
        return float("nan")
    s, m = stock_ret[-common:], mkt_ret[-common:]
    mask = ~(np.isnan(s) | np.isnan(m))
    s, m = s[mask], m[mask]
    if len(s) < 30:
        return float("nan")
    var_m = np.var(m)
    if var_m < 1e-12:
        return float("nan")
    return float(np.cov(s, m)[0, 1] / var_m)


# ── 加载指数数据 ───────────────────────────────────────────────────────────────

def _load_index_returns() -> dict[str, pd.Series]:
    """从所有快照的 index_daily.parquet 拼合 CSI300/CSI500 收益序列."""
    dfs = []
    for snap in sorted(_SNAPSHOT_DIR.glob("*/index_daily.parquet")):
        try:
            dfs.append(pd.read_parquet(snap))
        except Exception:
            pass
    if not dfs:
        logger.warning("未找到 index_daily 快照，Beta 计算将跳过")
        return {}
    all_idx = pd.concat(dfs)
    all_idx["trade_date"] = pd.to_datetime(all_idx["trade_date"])
    result: dict[str, pd.Series] = {}
    for code in ["000300.SH", "000905.SH"]:
        sub = (
            all_idx[all_idx["ts_code"] == code]
            .drop_duplicates("trade_date")
            .set_index("trade_date")
            .sort_index()["close"]
            .astype("float64")
        )
        if len(sub) > 60:
            result[code] = sub.pct_change(fill_method=None).dropna()
    logger.info(
        f"指数收益: CSI300={len(result.get('000300.SH', pd.Series()))} 天, "
        f"CSI500={len(result.get('000905.SH', pd.Series()))} 天"
    )
    return result


# ── 市场级 HMM ─────────────────────────────────────────────────────────────────

def _build_market_hmm(csi300_ret: pd.Series) -> dict:
    """基于 CSI300 日收益率训练市场级 HMM，识别三种市场状态.

    使用三维特征：21d 滚动方差、63d 累积收益、21d 均绝对收益（波动代理）。
    返回包含当前市场状态和模型的字典。
    """
    ret = csi300_ret.values.astype(np.float64)
    if len(ret) < 252:
        return {}

    # 特征：用 log(1 + 21d rolling var) 作为 HMM 观测（波动状态信号）
    rolling_var = pd.Series(ret).rolling(21, min_periods=10).var().bfill().values
    rolling_var = np.maximum(rolling_var, 1e-10)
    obs = np.log(rolling_var)

    hmm_model, states = _fit_hmm(obs, n_states=3)

    # 按均值波动率大小重新排列状态：0=低波动/牛市, 1=正常, 2=高波动/危机
    state_means: dict[int, float] = {}
    for s in range(3):
        sel = rolling_var[states == s]
        state_means[s] = float(np.mean(sel)) if sel.size > 0 else float("nan")

    finite = sorted([(s, m) for s, m in state_means.items() if np.isfinite(m)], key=lambda x: x[1])
    missing = [s for s in range(3) if not np.isfinite(state_means[s])]
    sorted_states = [s for s, _ in finite] + sorted(missing)
    state_label = {s: i for i, s in enumerate(sorted_states)}
    labeled = np.array([state_label[s] for s in states])

    current = int(labeled[-1])
    recent_30d = labeled[-30:]
    regime_probs = {i: round(float((recent_30d == i).mean()), 3) for i in range(3)}

    # 各状态平均收益（年化）
    state_ann_returns = {}
    for lbl in range(3):
        mask = labeled == lbl
        if mask.sum() > 10:
            state_ann_returns[lbl] = round(float(ret[mask].mean() * 252), 4)

    # 最近 63d 指数表现
    recent_ret_63d = float(np.prod(1 + ret[-63:]) - 1) if len(ret) >= 63 else float("nan")
    recent_vol_21d = float(np.std(ret[-21:]) * np.sqrt(252)) if len(ret) >= 21 else float("nan")

    return {
        "current_regime":      current,
        "regime_labels":       {0: "bull_low_vol", 1: "normal", 2: "bear_crisis"},
        "recent_30d_probs":    regime_probs,
        "state_ann_returns":   state_ann_returns,
        "csi300_ret_63d":      round(recent_ret_63d, 4) if np.isfinite(recent_ret_63d) else None,
        "csi300_vol_21d":      round(recent_vol_21d, 4) if np.isfinite(recent_vol_21d) else None,
        "n_observations":      len(obs),
    }


# ── 单股处理 ───────────────────────────────────────────────────────────────────

def process_ticker(
    ticker: str,
    prices: pd.Series,
    idx_rets: dict[str, pd.Series],
) -> dict:
    """为单只股票拟合全套风险参数."""
    prices = pd.to_numeric(prices, errors="coerce").dropna()
    ret = prices.pct_change(fill_method=None).dropna().values.astype(np.float64)
    if len(ret) < 120:
        return {}

    # ── 最优 EWMA lambda ──────────────────────────────────────────────────
    best_lam, best_ll = 0.94, -np.inf
    for lam in [0.90, 0.92, 0.94, 0.96, 0.98]:
        vols = _ewma_vol_by_lambda(ret, lam)
        sigma2 = (vols / np.sqrt(252)) ** 2
        ll = -0.5 * np.sum(np.log(sigma2[1:]) + ret[1:] ** 2 / sigma2[1:])
        if np.isfinite(ll) and ll > best_ll:
            best_ll, best_lam = ll, lam

    # ── 波动率 ────────────────────────────────────────────────────────────
    vol_21d = _ewma_vol(ret[-21:], lam=best_lam)
    vol_63d = float(np.std(ret[-63:]) * np.sqrt(252)) if len(ret) >= 63 else vol_21d
    long_run_vol = float(np.std(ret) * np.sqrt(252))

    # ── 个股 HMM 状态 ─────────────────────────────────────────────────────
    rolling_var = pd.Series(ret).rolling(21, min_periods=10).var().dropna().values
    _, states = _fit_hmm(rolling_var, n_states=3)

    state_means: dict[int, float] = {}
    for s in range(3):
        sel = rolling_var[states == s]
        state_means[s] = float(np.mean(sel)) if sel.size > 0 else float("nan")

    finite = sorted([(s, m) for s, m in state_means.items() if np.isfinite(m)], key=lambda x: x[1])
    missing = [s for s in range(3) if not np.isfinite(state_means[s])]
    sorted_states = [s for s, _ in finite] + sorted(missing)
    state_label = {s: i for i, s in enumerate(sorted_states)}
    labeled_states = np.array([state_label[s] for s in states])
    current_state = int(labeled_states[-1])

    state_vols = {}
    for lbl in range(3):
        mask = labeled_states == lbl
        if mask.sum() > 5:
            state_ret_indices = np.where(mask)[0] + 21
            state_ret_indices = state_ret_indices[state_ret_indices < len(ret)]
            if len(state_ret_indices) > 10:
                sv = _ewma_vol(ret[state_ret_indices[-21:]], lam=best_lam)
            else:
                sv = float(rolling_var[mask].mean() ** 0.5 * np.sqrt(252))
            state_vols[lbl] = round(sv, 6)

    # ── CVaR ──────────────────────────────────────────────────────────────
    cvar_window = ret[-252:] if len(ret) >= 252 else ret
    cvar_95 = _cvar_t(cvar_window, alpha=0.05)
    cvar_99 = _cvar_t(cvar_window, alpha=0.01)

    # ── Beta vs CSI300 / CSI500 ───────────────────────────────────────────
    # 基于价格日期对齐，取最近 252 日重叠区间
    beta_csi300 = float("nan")
    beta_csi500 = float("nan")

    # 将个股日收益转为带日期 Series（用价格 index）
    stock_ret_series = prices.pct_change(fill_method=None).dropna()
    window_days = 252

    for code, attr in [("000300.SH", "beta_vs_csi300"), ("000905.SH", "beta_vs_csi500")]:
        idx_r = idx_rets.get(code)
        if idx_r is None or idx_r.empty:
            continue
        # 对齐日期
        common = stock_ret_series.index.intersection(idx_r.index)
        common = common.sort_values()[-window_days:]
        if len(common) < 60:
            continue
        s_arr = stock_ret_series.loc[common].values.astype(np.float64)
        m_arr = idx_r.loc[common].values.astype(np.float64)
        b = _ols_beta(s_arr, m_arr)
        if attr == "beta_vs_csi300":
            beta_csi300 = b
        else:
            beta_csi500 = b

    return {
        "ann_vol_forecast_21d": round(vol_21d, 6),
        "vol_63d_annualized":   round(vol_63d, 6),
        "long_run_vol":         round(long_run_vol, 6),
        "ewma_lambda":          round(best_lam, 2),
        "current_regime":       current_state,
        "regime_vols":          state_vols,
        "cvar_95_daily":        round(cvar_95, 6),
        "cvar_99_daily":        round(cvar_99, 6),
        "beta_vs_csi300":       round(beta_csi300, 4) if np.isfinite(beta_csi300) else None,
        "beta_vs_csi500":       round(beta_csi500, 4) if np.isfinite(beta_csi500) else None,
        "regime_counts":        {lbl: int((labeled_states == lbl).sum()) for lbl in range(3)},
    }


# ── 主流程 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prices",
        default="data/raw/alpha_prices_panel.parquet",
        help="复权价格面板（长表或宽表）",
    )
    parser.add_argument(
        "--out",
        default="models/agents/risk_hmm_v3.pkl",
        help="输出 pkl 路径",
    )
    parser.add_argument("--min-days", type=int, default=250)
    args = parser.parse_args()

    price_path = ROOT / args.prices
    out_path   = ROOT / args.out

    if not price_path.exists():
        # 回退路径
        alt = ROOT / "data" / "raw" / "alpha_prices_panel.parquet"
        if alt.exists():
            price_path = alt
        else:
            raise FileNotFoundError(f"价格面板不存在: {price_path}")

    logger.info(f"加载价格: {price_path}")
    raw = pd.read_parquet(price_path)

    # 自动检测长表 vs 宽表
    if "ts_code" in raw.columns and "trade_date" in raw.columns:
        price_col = next((c for c in ["adj_close", "close"] if c in raw.columns), None)
        if price_col is None:
            raise ValueError(f"未找到价格列，现有列: {list(raw.columns)}")
        raw["trade_date"] = pd.to_datetime(raw["trade_date"])
        panel = (
            raw[["ts_code", "trade_date", price_col]]
            .drop_duplicates(subset=["ts_code", "trade_date"])
            .pivot(index="trade_date", columns="ts_code", values=price_col)
            .sort_index()
        )
        panel.columns.name = None
        logger.info(f"长表 pivot 完成: {panel.shape[0]} 天 × {panel.shape[1]} 只")
    else:
        panel = raw
        panel.index = pd.to_datetime(panel.index)
        panel = panel.sort_index()

    # ── 加载指数数据 ──────────────────────────────────────────────────────
    idx_rets = _load_index_returns()

    # ── 市场级 HMM ────────────────────────────────────────────────────────
    market_hmm: dict = {}
    if "000300.SH" in idx_rets:
        logger.info("训练市场级 HMM（CSI300）…")
        market_hmm = _build_market_hmm(idx_rets["000300.SH"])
        r = market_hmm.get("current_regime", -1)
        lbl = market_hmm.get("regime_labels", {}).get(r, "?")
        logger.info(
            f"  当前市场状态: [{r}] {lbl}  "
            f"| CSI300 近63d={market_hmm.get('csi300_ret_63d', '?')*100:.1f}%"
            if market_hmm.get("csi300_ret_63d") else
            f"  当前市场状态: [{r}] {lbl}"
        )

    # ── 逐股处理 ──────────────────────────────────────────────────────────
    tickers = [str(c) for c in panel.columns]
    logger.info(f"处理 {len(tickers)} 只股票 …")

    vol_by_ticker: dict[str, dict] = {}
    skipped = 0

    for i, tkr in enumerate(tickers):
        s = pd.to_numeric(panel[tkr], errors="coerce").dropna()
        if len(s) < args.min_days:
            skipped += 1
            continue
        result = process_ticker(tkr, s, idx_rets)
        if result:
            vol_by_ticker[tkr] = result

        if (i + 1) % 200 == 0:
            done = len(vol_by_ticker)
            logger.info(f"  进度 {i+1}/{len(tickers)}，已完成 {done}")

    logger.info(f"完成 {len(vol_by_ticker)} 只（跳过 {skipped} 只）")

    # 当前状态分布统计
    regime_dist = {0: 0, 1: 0, 2: 0}
    for info in vol_by_ticker.values():
        r = info.get("current_regime", 1)
        regime_dist[r] = regime_dist.get(r, 0) + 1
    logger.info(
        f"个股状态分布：低波动={regime_dist[0]}, 正常={regime_dist[1]}, 危机={regime_dist[2]}"
    )

    # Beta 统计
    betas = [v["beta_vs_csi300"] for v in vol_by_ticker.values() if v.get("beta_vs_csi300") is not None]
    if betas:
        logger.info(
            f"Beta vs CSI300：均值={np.mean(betas):.2f}, 中位数={np.median(betas):.2f}, "
            f"覆盖率={len(betas)}/{len(vol_by_ticker)}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "kind":              "risk_hmm_v3",
        "vol_by_ticker":     vol_by_ticker,
        "n_tickers":         len(vol_by_ticker),
        "regime_labels":     {0: "low_vol", 1: "normal", 2: "crisis"},
        "market_hmm":        market_hmm,
    }
    with open(out_path, "wb") as f:
        pickle.dump(bundle, f)
    logger.info(f"保存: {out_path}  ({out_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
