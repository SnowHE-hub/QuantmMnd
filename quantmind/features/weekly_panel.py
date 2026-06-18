"""quantmind.features.weekly_panel — 周频训练面板 v5 构建器（免回补因子子集）.

依据 ``docs/plans/weekly_panel_build_plan.md``（已评审批准）。

设计要点
--------
- **采样**：每 5 个交易日一个 ``as_of``，相位锚定到 fwd_12d 边界，使最后一个带
  ``forward_return_12d`` 标签的 ``as_of`` 恰为 2026-04-20（审计 §5）。
- **标签**：复用 :func:`quantmind.features.panel.compute_forward_returns`，用 **adj_close**，
  horizons=(12, 21, 63)。
- **因子**：35 个免回补子集（纯价 17 + 静态 4 + 指数 8 + 市场北向 1 + margin 纯量 5），
  存 **原始值**（``do_standardize=False``）。因子 ``close`` 用 **raw close**（与 v4 口径一致）。
- **外部日频**：经 :func:`quantmind.data.lake.read_lake_window` 严格 ``trade_date <= as_of``
  读 index_daily / north_bound / margin（不依赖 daily_basic / hk_hold 回补）。
- **universe**：快照并集（``data/lake/stock_basic.parquet``）∩ 有行情的票，按
  ``list_date <= as_of`` PIT 入池（修新股前视；**无法消除源头幸存者偏差**，见 meta）。
- **输出**：``data/panel/alpha_panel_weekly_v5.parquet``（新文件，**绝不覆盖 v4**）。

硬条件（计划 §10）
------------------
① 长窗 lookback：prices / index 窗口 >= 280 交易日，保证 beta_252d 在数据充足时用满
   252 窗口；数据不足的早期 ``as_of`` 由因子层自然产出 NaN（非垃圾值）。
② split-agnostic：面板只存真实 ``as_of`` + 三档标签，**不做任何 train/test 切分**。
③ 63d 复核以 v5 自身手算为基准（见 ``scripts/verify_weekly_panel.py``）。
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from quantmind.data.lake import load_trading_calendar, read_lake_window
from quantmind.features.panel import compute_forward_returns
from quantmind.features.pipeline import FeaturePipeline

# ============================================================================
# 路径与常量
# ============================================================================

_REPO_ROOT = Path(__file__).resolve().parents[2]
PRICE_PANEL_PATH = _REPO_ROOT / "data" / "raw" / "alpha_prices_panel.parquet"
STOCK_BASIC_PATH = _REPO_ROOT / "data" / "lake" / "stock_basic.parquet"
PANEL_DIR = _REPO_ROOT / "data" / "panel"
OUT_PATH = PANEL_DIR / "alpha_panel_weekly_v5.parquet"
META_PATH = PANEL_DIR / "alpha_panel_weekly_v5.meta.json"
GRID_PATH = PANEL_DIR / "weekly_asof_grid.parquet"
#: 绝不可被本构建器写入的既有生产面板
V4_PATH = PANEL_DIR / "alpha_panel_v4.parquet"

SAMPLE_STEP = 5
FWD_HORIZONS = (12, 21, 63)
#: 用于锚定网格相位的前瞻（fwd_12d 末截面落 2026-04-20）
ANCHOR_HORIZON = 12

# lookback（交易日）。prices/index >= 280 以支持 beta_252d / volatility_1y 用满 252 窗口
PRICES_LOOKBACK_TD = 280
INDEX_LOOKBACK_TD = 280
NORTH_LOOKBACK_TD = 45
MARGIN_LOOKBACK_TD = 60

#: 启用的因子组（只含产出白名单 35 列的组；fundamental 全部排除）
ENABLED_GROUPS = ["technical", "sentiment", "expansion"]

#: 硬条件①（全市场粒度）：整个窗口的交易日数 < 窗口时整列置 NaN。
#: 仅对"数据集很早期"（全市场窗口本身不足）兜底；逐 ticker 短历史由 _PER_TICKER_MIN_BARS 处理。
_MIN_TRADING_DAYS: dict[str, int] = {
    "momentum_1m": 21, "momentum_3m": 63, "momentum_6m": 126,
    "momentum_12m_skip_1m": 252, "volatility_3m": 63, "volatility_1y": 252,
    "downside_volatility_3m": 63, "max_drawdown_3m": 63, "amihud_illiquidity": 63,
    "distance_to_52w_high": 252, "price_to_52w_low": 252,
    "beta_60d": 60, "beta_252d": 252,
    "relative_strength_vs_csi300_60d": 60, "relative_strength_vs_csi300_120d": 120,
    "relative_strength_vs_csi500_60d": 60, "market_momentum_60d": 60,
    "market_volatility_60d": 60, "market_drawdown_60d": 60,
}

#: M2 修复（逐 ticker 粒度）：对"取末 N 行做 NaN-skipping 聚合"的滚动因子，
#: 当某 ticker 在窗口内的可用 bar 数 < 其窗口时，把【该 ticker 该列】置 NaN，
#: 防止短历史新股产出"薄估计"（违反"数据不足→NaN"契约）。
#: 仅含 iloc[-N:].{std,mean,cummax,max,min} 一族；momentum_*/beta_*/quantile 类已自带守卫（自然 NaN）。
_PER_TICKER_MIN_BARS: dict[str, int] = {
    "volatility_3m": 63,
    "volatility_1y": 252,
    "downside_volatility_3m": 63,
    "max_drawdown_3m": 63,
    "amihud_illiquidity": 63,
    "distance_to_52w_high": 252,
    "price_to_52w_low": 252,
}

# ----------------------------------------------------------------------------
# 35 因子白名单（计划 §3.1）
# ----------------------------------------------------------------------------
PURE_PRICE_17 = [
    "momentum_1m", "momentum_3m", "momentum_6m", "momentum_12m_skip_1m",
    "reversal_1w", "volatility_3m", "volatility_1y", "downside_volatility_3m",
    "max_drawdown_3m", "amihud_illiquidity", "volume_spike_5_30", "rsi_14",
    "bollinger_position", "distance_to_52w_high", "price_to_52w_low",
    "amplitude_quantile", "volume_price_corr_20d",
]
STATIC_4 = ["exposure_industry", "exposure_area", "list_age_years", "is_recent_ipo"]
INDEX_8 = [
    "beta_252d", "beta_60d", "relative_strength_vs_csi300_60d",
    "relative_strength_vs_csi300_120d", "relative_strength_vs_csi500_60d",
    "market_momentum_60d", "market_volatility_60d", "market_drawdown_60d",
]
NORTH_1 = ["north_bound_30d_net_inflow"]
MARGIN_PURE_5 = [
    "margin_balance", "margin_balance_change_20d", "margin_buy_amount_20d",
    "short_balance_change_20d", "margin_short_ratio",
]
WHITELIST_35: list[str] = PURE_PRICE_17 + STATIC_4 + INDEX_8 + NORTH_1 + MARGIN_PURE_5
assert len(WHITELIST_35) == 35, f"whitelist must be 35, got {len(WHITELIST_35)}"

LABEL_COLS = [f"forward_return_{h}d" for h in FWD_HORIZONS]

# ============================================================================
# 行情加载 + 字段映射（计划 §6）
# ============================================================================

#: alpha_prices_panel -> 因子函数期望列名
_PRICE_RENAME = {"ts_code": "ticker", "vol": "volume", "pct_chg": "pct_change"}
#: 因子 prices 子表需要的列（close 用 raw close，与 v4 口径一致）
_FACTOR_PRICE_COLS = [
    "trade_date", "ticker", "open", "high", "low", "close",
    "pre_close", "volume", "amount",
]


def load_prices_indexed() -> pd.DataFrame:
    """读 alpha_prices_panel，做列名映射，按 trade_date 建索引（供窗口切片）.

    返回长格式 DataFrame，``trade_date`` 为升序 DatetimeIndex（非唯一），含
    raw ``close`` 与 ``adj_close``（标签用）。
    """
    df = pd.read_parquet(PRICE_PANEL_PATH)
    df = df.rename(columns=_PRICE_RENAME)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["ticker"] = df["ticker"].astype(str)
    df = df.sort_values("trade_date").set_index("trade_date", drop=False)
    return df


def build_adj_close_pivot(prices: pd.DataFrame) -> pd.DataFrame:
    """标签用复权收盘透视：index=trade_date, columns=ticker, values=adj_close."""
    sub = prices[["trade_date", "ticker", "adj_close"]].dropna(subset=["adj_close"])
    return (
        sub.drop_duplicates(["trade_date", "ticker"], keep="last")
        .pivot(index="trade_date", columns="ticker", values="adj_close")
        .sort_index()
    )


# ============================================================================
# 采样网格（计划 §1）
# ============================================================================


def build_asof_grid(
    cal: pd.DatetimeIndex | None = None,
    *,
    step: int = SAMPLE_STEP,
    anchor_horizon: int = ANCHOR_HORIZON,
) -> list[pd.Timestamp]:
    """每 ``step`` 交易日取一个 ``as_of``，相位锚定到 fwd_{anchor_horizon} 边界.

    锚点 = ``cal[len-1-anchor_horizon]``，使最后一个带 fwd_{anchor_horizon} 标签的
    ``as_of`` 恰落在该日（受行情末日限制）。向前每 ``step`` 日回退取点，再升序。
    """
    if cal is None:
        cal = load_trading_calendar()
    n = len(cal)
    anchor_idx = n - 1 - anchor_horizon
    if anchor_idx < 0:
        raise ValueError("trading calendar too short for anchor horizon")
    idxs = sorted(range(anchor_idx, -1, -step))
    return [pd.Timestamp(cal[i]) for i in idxs]


# ============================================================================
# PIT universe（计划 §0.5 / §7）
# ============================================================================


def load_stock_basic() -> pd.DataFrame:
    """读数据湖 stock_basic（快照并集），规整 list_date / delist_date 为 datetime."""
    sb = pd.read_parquet(STOCK_BASIC_PATH).copy()
    sb["ticker"] = sb["ticker"].astype(str)
    for c in ("list_date", "delist_date"):
        if c in sb.columns:
            sb[c] = pd.to_datetime(sb[c], format="%Y%m%d", errors="coerce")
    return sb


def pit_universe(
    as_of: pd.Timestamp,
    stock_basic: pd.DataFrame,
    priced_tickers: set[str],
) -> pd.DataFrame:
    """快照并集 ∩ 有行情票，按 ``list_date <= as_of`` PIT 入池.

    退市票（``delist_date <= as_of``）若存在则剔除——本数据集 delist_date 全空，
    故该过滤实际不生效（源头幸存者偏差，见 meta 标注）。
    """
    as_of = pd.Timestamp(as_of).normalize()
    sb = stock_basic
    listed = sb["list_date"].notna() & (sb["list_date"] <= as_of)
    if "delist_date" in sb.columns:
        not_delisted = sb["delist_date"].isna() | (sb["delist_date"] > as_of)
    else:
        not_delisted = True
    pool = sb.loc[listed & not_delisted, "ticker"].astype(str)
    tickers = sorted(set(pool) & priced_tickers)
    return pd.DataFrame({"ticker": tickers})


# ============================================================================
# 单截面：组装 snapshot dict + 计算 35 因子（原始值）
# ============================================================================


def assemble_snapshot(
    as_of: pd.Timestamp,
    prices_indexed: pd.DataFrame,
    stock_basic: pd.DataFrame,
    universe: pd.DataFrame,
    cal: pd.DatetimeIndex,
) -> dict:
    """为 ``as_of`` 组装 ``FeaturePipeline`` 所需 snapshot dict（不调 Tushare）.

    缺 daily_basic / hk_hold / 财报键 → 相关因子函数走 NaN 兜底（已在白名单外）。
    """
    as_of = pd.Timestamp(as_of).normalize()
    # prices 窗口：[as_of 回看 PRICES_LOOKBACK_TD 交易日, as_of]，严格 <= as_of
    cal_le = cal[cal <= as_of]
    start = cal_le[-PRICES_LOOKBACK_TD] if len(cal_le) >= PRICES_LOOKBACK_TD else cal_le[0]
    px_win = prices_indexed.loc[start:as_of]
    # 严格 PIT 上界（.loc 已含 as_of；再防御性裁剪）
    px_win = px_win[px_win["trade_date"] <= as_of]
    prices = px_win[_FACTOR_PRICE_COLS].reset_index(drop=True)

    snap = {
        "prices": prices,
        "universe": universe,
        "stock_basic": stock_basic,
        # 结构性 shim（非数据）：north_bound_30d_net_inflow 从 daily_basic 取 ticker 索引
        # 来广播市场级值。这里只提供 universe 的 ticker 列表，不注入任何未回补的估值数据。
        "daily_basic": pd.DataFrame({"ticker": universe["ticker"].astype(str)}),
        "index_daily": read_lake_window("index_daily", as_of, INDEX_LOOKBACK_TD),
        "north_bound": read_lake_window("north_bound", as_of, NORTH_LOOKBACK_TD),
        "margin": read_lake_window("margin", as_of, MARGIN_LOOKBACK_TD),
    }
    return snap


def compute_factors_for_asof(
    as_of: pd.Timestamp,
    prices_indexed: pd.DataFrame,
    stock_basic: pd.DataFrame,
    cal: pd.DatetimeIndex,
    priced_tickers: set[str],
    *,
    pipe: FeaturePipeline | None = None,
) -> pd.DataFrame:
    """返回单截面 35 因子原始值 DataFrame(index=ticker, columns=WHITELIST_35)."""
    if pipe is None:
        pipe = FeaturePipeline(groups=ENABLED_GROUPS, do_standardize=False)
    universe = pit_universe(as_of, stock_basic, priced_tickers)
    if universe.empty:
        return pd.DataFrame(columns=WHITELIST_35)
    snap = assemble_snapshot(as_of, prices_indexed, stock_basic, universe, cal)
    raw = pipe.compute_raw_from_snapshot(snap, as_of.date())
    # 选白名单 35 列（缺列补 NaN 并断言齐全）
    missing = [c for c in WHITELIST_35 if c not in raw.columns]
    if missing:
        raise RuntimeError(f"factor compute missing whitelist cols: {missing}")
    out = raw[WHITELIST_35].copy()
    px = snap["prices"]
    px_days = pd.to_datetime(px["trade_date"]).dt.normalize()

    # 硬条件①（全市场）：整个窗口交易日数 < 窗口时整列置 NaN（数据集早期兜底）
    available_td = int(px_days.nunique())
    for col, req in _MIN_TRADING_DAYS.items():
        if available_td < req and col in out.columns:
            out[col] = np.nan

    # M2（逐 ticker）：每只票在窗口内的可用 close bar 数 < 因子窗口 → 该票该列置 NaN
    # （防晚期 as_of 下短历史新股在 iloc[-N:] 聚合上产出薄估计）
    bars = (
        px.assign(_d=px_days)
        .dropna(subset=["close"])
        .drop_duplicates(["ticker", "_d"])
        .groupby("ticker")["_d"]
        .size()
    )
    bars = bars.reindex(out.index).fillna(0).astype(int)
    for col, req in _PER_TICKER_MIN_BARS.items():
        if col in out.columns:
            out.loc[bars < req, col] = np.nan
    return out


# ============================================================================
# 标签（计划 §2，复用 compute_forward_returns）
# ============================================================================


def attach_labels(
    feat_panel: pd.DataFrame,
    adj_pivot: pd.DataFrame,
    as_of_list: list[pd.Timestamp],
) -> pd.DataFrame:
    """对每个 as_of 用 adj_close 复用 compute_forward_returns，merge 三档标签."""
    rows = []
    for as_of in as_of_list:
        fr = compute_forward_returns(adj_pivot, as_of.date(), FWD_HORIZONS)
        if fr.empty:
            continue
        fr = fr.reset_index().rename(columns={"index": "ticker"})
        fr["ticker"] = fr["ticker"].astype(str)
        fr["as_of"] = pd.Timestamp(as_of)
        rows.append(fr)
    if not rows:
        out = feat_panel.copy()
        for c in LABEL_COLS:
            out[c] = np.nan
        return out
    fr_panel = pd.concat(rows, ignore_index=True)
    out = feat_panel.merge(fr_panel, on=["as_of", "ticker"], how="left")
    return out


# ============================================================================
# 主入口
# ============================================================================


#: 每个 worker 进程加载一次的全局缓存（fork 后各进程独立）
_WORKER: dict = {}


def _worker_init() -> None:
    _WORKER["cal"] = load_trading_calendar()
    _WORKER["prices"] = load_prices_indexed()
    _WORKER["priced"] = set(_WORKER["prices"]["ticker"].astype(str).unique())
    _WORKER["sb"] = load_stock_basic()
    _WORKER["pipe"] = FeaturePipeline(groups=ENABLED_GROUPS, do_standardize=False)


def _worker_one(as_of_iso: str) -> pd.DataFrame | None:
    import time as _time

    as_of = pd.Timestamp(as_of_iso)
    # 对 UNC/网络路径下偶发的 parquet 读失败做有限重试（瞬时错误，非逻辑错误）
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            df = compute_factors_for_asof(
                as_of, _WORKER["prices"], _WORKER["sb"], _WORKER["cal"],
                _WORKER["priced"], pipe=_WORKER["pipe"],
            )
            break
        except Exception as e:  # noqa: BLE001
            last_exc = e
            _time.sleep(0.5 * (attempt + 1))
    else:
        raise RuntimeError(f"as_of {as_of_iso} failed after retries: {last_exc}")
    if df.empty:
        return None
    df = df.reset_index()
    if "ticker" not in df.columns:
        df = df.rename(columns={df.columns[0]: "ticker"})
    df["ticker"] = df["ticker"].astype(str)
    df["as_of"] = pd.Timestamp(as_of)
    return df


def _one_cross_section(
    as_of: pd.Timestamp, prices, stock_basic, cal, priced_tickers, pipe
) -> pd.DataFrame | None:
    df = compute_factors_for_asof(as_of, prices, stock_basic, cal, priced_tickers, pipe=pipe)
    if df.empty:
        return None
    df = df.reset_index()
    if "ticker" not in df.columns:
        df = df.rename(columns={df.columns[0]: "ticker"})
    df["ticker"] = df["ticker"].astype(str)
    df["as_of"] = pd.Timestamp(as_of)
    return df


def build_weekly_panel(
    *,
    save: bool = True,
    limit_asof: int | None = None,
    progress: bool = True,
    n_workers: int | None = None,
) -> pd.DataFrame:
    """端到端构建周频面板 v5。

    Args:
        save: 是否落盘到 OUT_PATH（+ meta + grid）。
        limit_asof: 仅取前 N 个 as_of（小样本调试用；None=全量）。
        progress: 是否显示进度条。
        n_workers: 并行进程数（None=自动取 min(16, cpu)；1=单进程）。

    Returns:
        MultiIndex(as_of, ticker) 面板：35 因子原始值 + 三档 forward_return。
    """
    cal = load_trading_calendar()
    prices = load_prices_indexed()
    priced_tickers = set(prices["ticker"].astype(str).unique())
    stock_basic = load_stock_basic()
    adj_pivot = build_adj_close_pivot(prices)

    as_of_list = build_asof_grid(cal)
    if limit_asof is not None:
        as_of_list = as_of_list[:limit_asof]

    if n_workers is None:
        n_workers = min(16, os.cpu_count() or 1)

    feats: list[pd.DataFrame] = []
    if n_workers and n_workers > 1:
        isos = [a.isoformat() for a in as_of_list]
        with ProcessPoolExecutor(max_workers=n_workers, initializer=_worker_init) as ex:
            it = ex.map(_worker_one, isos, chunksize=1)
            it = tqdm(it, total=len(isos), desc="weekly_panel", ncols=80) if progress else it
            for df in it:
                if df is not None:
                    feats.append(df)
    else:
        pipe = FeaturePipeline(groups=ENABLED_GROUPS, do_standardize=False)
        it = tqdm(as_of_list, desc="weekly_panel", ncols=80) if progress else as_of_list
        for as_of in it:
            df = _one_cross_section(as_of, prices, stock_basic, cal, priced_tickers, pipe)
            if df is not None:
                feats.append(df)

    if not feats:
        raise RuntimeError("no factor cross-sections built")

    feat_panel = pd.concat(feats, ignore_index=True)
    panel = attach_labels(feat_panel, adj_pivot, as_of_list)
    panel = panel.set_index(["as_of", "ticker"]).sort_index()
    # 列顺序固定：35 因子 + 3 标签
    panel = panel[WHITELIST_35 + LABEL_COLS]

    if save:
        _save_panel(panel, as_of_list)
    return panel


def _save_panel(panel: pd.DataFrame, as_of_list: list[pd.Timestamp]) -> None:
    if OUT_PATH.resolve() == V4_PATH.resolve():
        raise RuntimeError("refuse to overwrite alpha_panel_v4.parquet")
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUT_PATH)
    # as_of 网格副本（供增量 join 复用，保证同键）
    pd.DataFrame({"as_of": pd.to_datetime(as_of_list)}).to_parquet(GRID_PATH, index=False)

    asof_idx = panel.index.get_level_values("as_of")
    meta = {
        "panel": "alpha_panel_weekly_v5",
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "sampling": f"every {SAMPLE_STEP} trading days, anchored to fwd_{ANCHOR_HORIZON}d boundary",
        "n_asof": int(asof_idx.nunique()),
        "n_rows": int(len(panel)),
        "first_asof": str(asof_idx.min().date()),
        "last_asof": str(asof_idx.max().date()),
        "factor_cols": WHITELIST_35,
        "label_cols": LABEL_COLS,
        "n_factors": len(WHITELIST_35),
        "standardized": False,
        "factor_close": "raw close (v4 口径)",
        "label_price": "adj_close (qfq)",
        "groups_enabled": ENABLED_GROUPS,
        "universe": "lake stock_basic union ∩ priced tickers, PIT by list_date<=as_of",
        "lookback_trading_days": {
            "prices": PRICES_LOOKBACK_TD, "index_daily": INDEX_LOOKBACK_TD,
            "north_bound": NORTH_LOOKBACK_TD, "margin": MARGIN_LOOKBACK_TD,
        },
        "survivorship_caveat": (
            "源头幸存者偏差：alpha_prices_panel / stock_basic 仅含构建时仍挂牌(L)的票，"
            "0 退市记录（delist_date 全空）。非静态 universe 仅修新股前视，无法消除幸存者偏差。"
            "基于本面板的 alpha/P1 结论须当作【乐观上界 optimistic upper bound】读；"
            "真正修复需回补退市票行情（后续数据湖回补项）。"
        ),
        "margin_note": (
            "margin 因子在 data/lake/margin 覆盖起点之前的 as_of 为 NaN（不填充）。"
        ),
        "deferred_increments": {
            "daily_basic_16": "估值/换手/自由流通 + margin_buy_intensity/short_sell_pressure",
            "hk_hold_6": "north_hold_* 6 因子",
            "fundamental_pit_16": "财报指标，按 f_ann_date<=as_of PIT",
            "join_contract": "merge_increment() 按 (as_of,ticker) 左对齐，不重建整表",
        },
    }
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


# ============================================================================
# 增量 join 接口（计划 §4，不重建整表）
# ============================================================================


def merge_increment(
    base_panel: pd.DataFrame,
    increment: pd.DataFrame,
) -> pd.DataFrame:
    """把增量因子表按 ``(as_of, ticker)`` 左对齐 join 进 base 面板.

    - 两者 index 均须为 MultiIndex(as_of, ticker)。
    - 不新增/删除 base 行；只增列。列名冲突即报错（避免静默覆盖）。
    - 用于日后 daily_basic-16 / hk_hold-6 / 财报-16 增量补入（§4），不重建整表。
    """
    if not isinstance(base_panel.index, pd.MultiIndex):
        raise ValueError("base_panel must have MultiIndex(as_of, ticker)")
    if not isinstance(increment.index, pd.MultiIndex):
        raise ValueError("increment must have MultiIndex(as_of, ticker)")
    if list(base_panel.index.names) != list(increment.index.names):
        raise ValueError(
            f"index names mismatch: base={list(base_panel.index.names)} "
            f"vs increment={list(increment.index.names)}"
        )
    if not increment.index.is_unique:
        # 增量含重复键会令 join 先笛卡尔放大——显式拒绝，避免静默错对齐/放大
        n_dup = int(increment.index.duplicated().sum())
        raise ValueError(f"increment index has {n_dup} duplicate (as_of,ticker) keys")
    dup = set(base_panel.columns) & set(increment.columns)
    if dup:
        raise ValueError(f"increment columns collide with base: {sorted(dup)}")
    out = base_panel.join(increment, how="left")
    if len(out) != len(base_panel):
        # 显式 raise（assert 在 python -O 下失效）
        raise RuntimeError(
            f"row count changed by join: base={len(base_panel)} -> out={len(out)}"
        )
    return out


__all__ = [
    "WHITELIST_35",
    "LABEL_COLS",
    "FWD_HORIZONS",
    "OUT_PATH",
    "GRID_PATH",
    "build_asof_grid",
    "load_prices_indexed",
    "build_adj_close_pivot",
    "load_stock_basic",
    "pit_universe",
    "assemble_snapshot",
    "compute_factors_for_asof",
    "attach_labels",
    "build_weekly_panel",
    "merge_increment",
]
