"""scripts/build_full_panel.py — 多时点季末快照 → 全特征面板（含扩展因子 + 前视标签）.

从 snapshot-dir 扫描 meta.json，按季度去重后逐期调用 FeaturePipeline；
8 文件快照扩展因子列为 NaN，12 文件快照计算扩展因子。输出 MultiIndex(as_of, ticker)。
"""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from quantmind.core.config import PROJECT_ROOT
from quantmind.core.logger import get_logger, setup_logger
from quantmind.features.expansion import (
    CATEGORICAL_EXPOSURE_FIELDS,
    EXPANSION_FACTORS,
)
from quantmind.features.panel import (
    build_return_index_pivot,
    compute_forward_returns,
    fetch_forward_price_pivot,
    splice_return_index,
)
from quantmind.features.pipeline import ALL_FACTOR_GROUPS, FeaturePipeline

log = get_logger(__name__)

EXPANSION_FILE_THRESHOLD = 12
BASE_GROUPS = ["fundamental", "technical", "sentiment"]


def _parse_date(s: str) -> date:
    return date.fromisoformat(s)


def _quarter_key(d: date) -> tuple[int, int]:
    return (d.year, (d.month - 1) // 3 + 1)


def _resolve_under_root(p: Path) -> Path:
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()


def _load_snapshot_from_dir(out_dir: Path) -> dict:
    meta_path = out_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(out_dir)
    meta = json.loads(meta_path.read_text())
    out: dict = {"meta": meta}
    for f in meta.get("files", []):
        key = str(f).removesuffix(".parquet")
        path = out_dir / f
        if path.exists():
            out[key] = pd.read_parquet(path)
    return out


def _price_pivot_from_snapshots(
    snapshot_root: Path, dates_kept: list[date]
) -> pd.DataFrame:
    """合并各 snapshot 的 prices.parquet，生成用于前视标签的 close 宽表."""
    chunks: list[pd.DataFrame] = []
    for d in sorted(dates_kept):
        p = snapshot_root / d.isoformat() / "prices.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p, columns=["trade_date", "ticker", "close"])
        chunks.append(df)
    if not chunks:
        return pd.DataFrame()
    long = pd.concat(chunks, ignore_index=True)
    long["trade_date"] = pd.to_datetime(long["trade_date"])
    long = long.sort_values(["trade_date", "ticker"])
    long = long.drop_duplicates(subset=["trade_date", "ticker"], keep="last")
    return long.pivot(index="trade_date", columns="ticker", values="close").sort_index()


# 每个数据源：(它实际有数据的交易日——不按股票过滤, 它出现过的股票)
Coverage = list[tuple[pd.DatetimeIndex, frozenset]]
DEFAULT_TRADE_CALENDAR = Path("data/lake/trade_cal_sse.parquet")


def _return_index_from_snapshots(
    snapshot_root: Path,
    tickers: list[str] | None = None,
    *,
    through: pd.Timestamp | None = None,
    extra_long: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """合并 ``snapshot_root`` 下全部快照的 pct_change，连乘成单一基准的收益指数宽表（T-003）.

    取代 ``_price_pivot_from_snapshots``：后者拼接的是各快照以自身日期为基准的前复权
    close，跨快照基准不一致。读全部快照（而不只是入选截面的快照）以覆盖每个交易日；
    pct_change 跨快照一致，重复日期保留排序靠后的快照的行。

    ``extra_long``（列同上）补充快照里没有的 (trade_date, ticker) 行，例如日线长表；
    快照已有的行以快照为准。只有与 pct_change 同定义的日收益才能放进来。

    注意：缺行在这里按停牌处理（不贡献收益）。只有数据源确实覆盖该票时才成立，
    计算标签时必须配合 ``_observed_mask`` + ``_drop_unobserved`` 使用。
    """
    long, _ = _load_return_sources(snapshot_root, tickers, through=through, extra_long=extra_long)
    return build_return_index_pivot(long) if not long.empty else pd.DataFrame()


def _load_return_sources(
    snapshot_root: Path,
    tickers: list[str] | None = None,
    *,
    through: pd.Timestamp | None = None,
    extra_long: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, Coverage]:
    """读取全部日收益来源（按 ``tickers`` 过滤后的长表），并返回每个来源的覆盖 (日期集合, 股票集合).

    一个来源"覆盖" (d, t) 指该来源在 d 这天有数据（任意股票有行）且 t 出现在该来源里：此时 t 当天无行才是
    真正的停牌。覆盖在**按股票过滤之前**计算——面板里的票全体停牌、其他票照常交易的日子仍算被覆盖（T-003 重验修复）。
    快照只含其自身票池；某票不在某快照里时，那段日期对它而言是**缺数据**，不是停牌；
    来源内部整日缺失的日期不算被该来源覆盖。
    """
    tick_set = set(tickers) if tickers else None
    chunks: list[pd.DataFrame] = []
    coverage: Coverage = []

    def take(df: pd.DataFrame) -> pd.DataFrame:
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        if through is not None:
            df = df[df["trade_date"] <= through]
        if not df.empty:
            coverage.append((pd.DatetimeIndex(df["trade_date"].unique()).sort_values(),
                             frozenset(df["ticker"].astype(str).unique())))
        return df[df["ticker"].isin(tick_set)] if tick_set is not None else df

    if extra_long is not None and not extra_long.empty:
        # 放在最前，同日同票时被快照行覆盖
        chunks.append(take(extra_long[["trade_date", "ticker", "close", "pct_change"]].copy()))
    for d_dir in sorted(snapshot_root.iterdir()):
        if not d_dir.is_dir():
            continue
        try:
            date.fromisoformat(d_dir.name)
        except ValueError:
            continue
        p = d_dir / "prices.parquet"
        if not p.exists():
            continue
        chunks.append(take(pd.read_parquet(p, columns=["trade_date", "ticker", "close", "pct_change"])))
    chunks = [c for c in chunks if not c.empty]
    if not chunks:
        return pd.DataFrame(), coverage
    long = pd.concat(chunks, ignore_index=True)
    long = long.sort_values(["trade_date", "ticker"], kind="stable")
    return long, coverage


def _load_trade_calendar(path: Path | None) -> pd.DatetimeIndex | None:
    """独立交易日历（Tushare trade_cal，列 cal_date）；不存在时返回 None。"""
    if path is None or not path.is_file():
        return None
    return pd.DatetimeIndex(pd.to_datetime(pd.read_parquet(path)["cal_date"])).sort_values().unique()


def _label_axis(coverage: Coverage, calendar: pd.DatetimeIndex | None) -> pd.DatetimeIndex:
    """标签的交易日轴：各来源实际有数据的日期（不按股票过滤）∪ 来源日期区间内的独立交易日历.

    与面板票池无关，因此面板里的票全体停牌的日子不会从轴上消失、horizon 不会错位。
    日历里有、但没有任何来源有数据的日子留在轴上，由 ``_observed_mask`` 标为未覆盖（标签置 NaN）。
    来源里出现日历范围内的非交易日 → ValueError。
    """
    if not coverage:
        return pd.DatetimeIndex([])
    src = coverage[0][0]
    for dates, _ in coverage[1:]:
        src = src.union(dates)
    if calendar is None or len(calendar) == 0:
        return src
    inside = src[(src >= calendar.min()) & (src <= calendar.max())]
    bad = inside.difference(calendar)
    if len(bad):
        raise ValueError(f"{len(bad)} source dates are not trading days in the calendar, e.g. {list(bad[:5])}")
    return src.union(calendar[(calendar >= src.min()) & (calendar <= src.max())])


def _observed_mask(index: pd.DatetimeIndex, columns: pd.Index, coverage: Coverage) -> pd.DataFrame:
    """(交易日 × 股票) 布尔矩阵：该票在该日是否被至少一个数据源覆盖（该源当天有数据且含该票）."""
    arr = np.zeros((len(index), len(columns)), dtype=bool)
    colpos = {str(c): i for i, c in enumerate(columns)}
    for dates, names in coverage:
        pos = index.get_indexer(dates)
        rows = pos[pos >= 0]
        cols = [colpos[t] for t in names if t in colpos]
        if len(rows) and cols:
            arr[np.ix_(rows, cols)] = True
    return pd.DataFrame(arr, index=index, columns=columns)


def _align_to_axis(
    pivot: pd.DataFrame, coverage: Coverage, calendar: pd.DatetimeIndex | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """收益指数宽表对齐到 ``_label_axis``（补出的行全为 NaN；指数值不变：没有行的日子本就不贡献收益），
    并给出同轴的覆盖矩阵。"""
    axis = _label_axis(coverage, calendar)
    missing = pivot.index.difference(axis)
    if len(missing):
        raise ValueError(f"return index has {len(missing)} dates outside the label axis, e.g. {list(missing[:5])}")
    out = pivot.reindex(axis)
    observed = _observed_mask(out.index, out.columns, coverage)
    uncovered = int((~observed.any(axis=1)).sum())
    log.info("label axis: {} days ({} added beyond the panel tickers' own rows); days no source covers: {}",
             len(axis), len(axis) - len(pivot.index), uncovered)
    return out, observed


def _drop_unobserved(
    fr: pd.DataFrame, observed: pd.DataFrame, as_of: date, horizons: tuple[int, ...],
) -> tuple[pd.DataFrame, int]:
    """标签窗口 (基准日, 目标日] 内有任何一天未被覆盖的票，标签置 NaN；返回 (结果, 置空个数).

    基准日与目标日的确定方式与 ``compute_forward_returns`` 相同。
    """
    idx = observed.index
    target = pd.Timestamp(as_of)
    on_or_before = idx[idx <= target]
    after = idx[idx >= target]
    if len(after) == 0:
        return fr, 0
    base = on_or_before[-1] if len(on_or_before) else after[0]
    future = idx[idx > base]
    out, n = fr.copy(), 0
    for h in horizons:
        col = f"forward_return_{h}d"
        if col not in out.columns or len(future) < h:
            continue
        window = observed.loc[(idx > base) & (idx <= future[h - 1])]
        gap = ~window.all(axis=0)
        hit = out.index.intersection(gap.index[gap.to_numpy()])
        n += int(out.loc[hit, col].notna().sum())
        out.loc[hit, col] = np.nan
    return out, n


def _daily_panel_long(raw_parquet: Path) -> pd.DataFrame | None:
    """日线长表 → (trade_date, ticker, close, pct_change)；无 pct_chg 列时返回 None.

    [实测 2026-10-10] daily_prices_panel.parquet 的 pct_chg 与快照 pct_change 在
    670,254 个重叠 (trade_date, ticker) 上逐条相同。
    """
    if not raw_parquet.is_file():
        return None
    import pyarrow.parquet as pq

    names = set(pq.read_schema(raw_parquet).names)
    if not {"trade_date", "ts_code", "close", "pct_chg"}.issubset(names):
        return None
    df = pd.read_parquet(raw_parquet, columns=["trade_date", "ts_code", "close", "pct_chg"])
    return df.rename(columns={"ts_code": "ticker", "pct_chg": "pct_change"})


def _extend_pivot_with_daily_parquet(
    pivot: pd.DataFrame,
    tickers: list[str],
    *,
    raw_parquet: Path,
    extend_through: pd.Timestamp,
) -> pd.DataFrame:
    """快照行情往往止于截面日，用日线长表补足后续交易日以便计算末期的 forward 标签.

    ``pivot`` 是收益指数（见 ``_return_index_from_snapshots``），补足部分按日收益接续。
    """
    if pivot.empty or not raw_parquet.is_file():
        return pivot
    if pivot.index.max() >= extend_through:
        return pivot
    df = pd.read_parquet(raw_parquet)
    req = {"trade_date", "ts_code"}
    if not req.issubset(df.columns):
        log.warning(
            "extend pivot: {} missing columns {}, have {}",
            raw_parquet,
            sorted(req - set(df.columns)),
            list(df.columns)[:20],
        )
        return pivot
    price_col = "adj_close" if "adj_close" in df.columns else "close"
    if price_col not in df.columns:
        log.warning("extend pivot: no adj_close/close in {}", raw_parquet)
        return pivot
    df = df[["trade_date", "ts_code", price_col]].copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    tick_set = set(tickers)
    df = df[df["ts_code"].isin(tick_set)]
    if df.empty:
        log.warning("extend pivot: no rows in {} after ticker filter", raw_parquet)
        return pivot
    wide = df.pivot(index="trade_date", columns="ts_code", values=price_col).sort_index()
    cols = [c for c in pivot.columns if c in wide.columns]
    if not cols:
        log.warning("extend pivot: no overlapping ticker columns with {}", raw_parquet)
        return pivot
    # T-003：按日收益衔接，不按价格水平拼接（两段复权基准不同）
    out = splice_return_index(pivot, wide)
    log.info(
        "extended pivot: {}→{} sessions (via {})",
        len(pivot.index),
        len(out.index),
        raw_parquet.name,
    )
    return out


def _dedup_quarter(
    dates: list[date],
    snapshot_root: Path,
    strategy: Literal["trading_day", "earlier"],
) -> list[date]:
    by_q: dict[tuple[int, int], list[date]] = {}
    for d in sorted(dates):
        by_q.setdefault(_quarter_key(d), []).append(d)

    kept: list[date] = []
    for q in sorted(by_q.keys()):
        cands = sorted(by_q[q])
        if len(cands) == 1:
            kept.append(cands[0])
            continue

        if strategy == "earlier":
            kept.append(cands[0])
            continue

        # trading_day: keep folder date matching universe.trade_date; else earliest date
        scores: list[tuple[int, date]] = []
        for d in cands:
            u_path = snapshot_root / d.isoformat() / "universe.parquet"
            if not u_path.exists():
                scores.append((2, d))
                continue
            try:
                u = pd.read_parquet(u_path, columns=["trade_date"])
                td = pd.to_datetime(u["trade_date"].iloc[0], errors="coerce")
            except Exception:
                scores.append((2, d))
                continue
            if pd.isna(td):
                scores.append((2, d))
                continue
            td_d = td.date()
            priority = 0 if d == td_d else 1
            scores.append((priority, d))
        min_pri = min(s[0] for s in scores)
        best = [d for pri, d in scores if pri == min_pri]
        kept.append(min(best))

    return sorted(kept)


def _empty_expansion_df(index: pd.Index) -> pd.DataFrame:
    data: dict[str, pd.Series] = {}
    for name, _ in EXPANSION_FACTORS:
        if name in CATEGORICAL_EXPOSURE_FIELDS:
            data[name] = pd.Series(pd.NA, index=index, dtype="string")
        else:
            data[name] = pd.Series(np.nan, index=index, dtype="float64")
    return pd.DataFrame(data)


def _base_column_order() -> list[str]:
    out: list[str] = []
    for g in BASE_GROUPS:
        out.extend([name for name, _ in ALL_FACTOR_GROUPS[g]])
    seen: set[str] = set()
    result: list[str] = []
    for c in out:
        if c not in seen:
            seen.add(c)
            result.append(c)
    return result


def _expansion_column_order() -> list[str]:
    return [name for name, _ in EXPANSION_FACTORS]


LABEL_COLS = ["forward_return_21d", "forward_return_63d"]


def _relabel_panel(
    src: Path,
    out: Path,
    *,
    snapshot_root: Path,
    raw_daily: Path,
    through: pd.Timestamp | None,
    keep_label_mask: bool = False,
    trade_calendar: Path | None = None,
) -> int:
    """只重算已有面板的两列标签（T-003），因子列原样保留；输出写新文件，不覆盖任何已有文件.

    ``keep_label_mask``：新标签只写在源面板该列非空的位置，使新旧面板的样本集合完全一致，
    差别只在标签数值（用于隔离"价格基准"这一个变量）。
    ``trade_calendar``：独立交易日历 parquet（列 cal_date）；None 时日期轴只由各来源的日期（不按股票过滤）构成。
    """
    if out.resolve() == src.resolve():
        log.error("--output must differ from --relabel-from ({})", src)
        return 1
    if out.exists():
        log.error("output exists, refusing to overwrite: {}", out)
        return 1
    panel = pd.read_parquet(src)
    missing = [c for c in LABEL_COLS if c not in panel.columns]
    if missing:
        log.error("panel {} lacks label columns {}", src, missing)
        return 1
    as_ofs = sorted(pd.Timestamp(x) for x in panel.index.get_level_values("as_of").unique())
    tickers = sorted(panel.index.get_level_values("ticker").astype(str).unique())

    daily_long = _daily_panel_long(raw_daily)
    long, coverage = _load_return_sources(
        snapshot_root, tickers, through=through, extra_long=daily_long,
    )
    if long.empty:
        log.error("no snapshot prices under {}", snapshot_root)
        return 1
    pivot = build_return_index_pivot(long)
    need_through = as_ofs[-1] + timedelta(days=65)
    if daily_long is None and raw_daily.is_file() and pivot.index.max() < need_through:
        last = pivot.index.max()
        pivot = _extend_pivot_with_daily_parquet(
            pivot, tickers, raw_parquet=raw_daily, extend_through=need_through,
        )
        if pivot.index.max() > last:  # 接续段：无价即停牌的语义由 splice 保证
            coverage.append((pivot.index[pivot.index > last], frozenset(map(str, pivot.columns))))
    if through is not None:
        pivot = pivot[pivot.index <= through]
        coverage = [(d[d <= through], n) for d, n in coverage]
    pivot, observed = _align_to_axis(pivot, coverage, _load_trade_calendar(trade_calendar))
    log.info("relabel: return index {} sessions × {} tickers, last {}",
             pivot.shape[0], pivot.shape[1], pivot.index.max().date())

    parts: list[pd.DataFrame] = []
    n_unobserved = 0
    for as_of in as_ofs:
        fr = compute_forward_returns(pivot, as_of.date(), (21, 63))
        if fr.empty:
            continue
        fr, k = _drop_unobserved(fr, observed, as_of.date(), (21, 63))
        n_unobserved += k
        fr.index.name = "ticker"
        fr["as_of"] = as_of
        parts.append(fr.reset_index())
    labels = (
        pd.concat(parts, ignore_index=True).set_index(["as_of", "ticker"])
        if parts else pd.DataFrame(columns=LABEL_COLS)
    )

    new = panel.copy()
    for lc in LABEL_COLS:
        new[lc] = labels[lc].reindex(new.index).astype(float) if lc in labels else np.nan
        if keep_label_mask:
            new[lc] = new[lc].where(panel[lc].notna())
    pd.testing.assert_frame_equal(
        new.drop(columns=LABEL_COLS), panel.drop(columns=LABEL_COLS), check_exact=True,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    new.to_parquet(out)

    for lc in LABEL_COLS:
        old_v, new_v = panel[lc], new[lc]
        both = old_v.notna() & new_v.notna()
        diff = (old_v[both] - new_v[both]).abs()
        print(
            f"{lc}: old non-NaN {int(old_v.notna().sum())} · new non-NaN {int(new_v.notna().sum())} · "
            f"both {int(both.sum())} · |diff|>0.5% {int((diff > 0.005).sum())} · |diff|>10% {int((diff > 0.10).sum())}"
        )
    print(f"labels set to NaN because the window has days no source covers: {n_unobserved}")
    print(f"RELABELED shape={new.shape} path={out}")
    return 0


def main() -> int:
    setup_logger()
    parser = argparse.ArgumentParser(description="Build full cross-sectional feature panel")
    parser.add_argument("--snapshot-dir", type=Path, default=Path("data/snapshots"))
    parser.add_argument(
        "--snapshot-dirs",
        type=Path,
        nargs="*",
        default=None,
        help=(
            "仅处理列出的快照目录（完整路径或相对项目根）。须位于同一父目录下（通常为 data/snapshots）。"
            "指定后忽略按日期范围的目录扫描。"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/features/csi300_full_panel.parquet"),
    )
    parser.add_argument("--start-date", type=_parse_date, default=date(2019, 3, 31))
    parser.add_argument("--end-date", type=_parse_date, default=date(2024, 12, 31))
    parser.add_argument(
        "--dedup-strategy",
        choices=["trading_day", "earlier"],
        default="trading_day",
    )
    parser.add_argument("--log-dir", type=Path, default=Path("reports/panel_build"))
    parser.add_argument(
        "--extra-industry-neutralize",
        action="store_true",
        help="在写入前对数值因子再做一轮行业内减均值（quantmind.features.neutralize）",
    )
    parser.add_argument(
        "--add-sentiment-parquet",
        type=Path,
        default=None,
        help="可选：sentiment_scores_YYYY-MM-DD.parquet，按文件名日期合并 sentiment_score 列",
    )
    parser.add_argument(
        "--price-panel",
        type=Path,
        default=None,
        help=(
            "可选：用于补足快照行情、计算 forward 标签的长表 parquet。"
            "须含 trade_date、ts_code，以及 adj_close 或 close。"
            "默认使用项目内 data/raw/daily_prices_panel.parquet（若存在）。"
        ),
    )
    parser.add_argument(
        "--universe-file",
        "--universe",
        type=Path,
        default=None,
        dest="universe_file",
        help=(
            "可选：一行一只 ts_code（如 000001.SZ），构建完成后仅保留这些 ticker。"
            "别名：--universe。因子仍由各快照 universe.parquet 驱动。"
        ),
    )
    parser.add_argument(
        "--relabel-from",
        type=Path,
        default=None,
        help=(
            "T-003：只重算该面板的 forward_return_21d/63d（单一价格基准），因子列原样保留，"
            "写入 --output（不得与输入相同，且不得已存在）。指定后忽略其余构建参数。"
        ),
    )
    parser.add_argument(
        "--price-through",
        type=_parse_date,
        default=None,
        help="仅与 --relabel-from 配合：只使用该日（含）之前的行情，用于复现历史面板的标签可得范围。",
    )
    parser.add_argument(
        "--keep-label-mask",
        action="store_true",
        help="仅与 --relabel-from 配合：只在源面板标签非空的位置写新标签（样本集合不变，只改数值）。",
    )
    parser.add_argument(
        "--trade-calendar",
        type=Path,
        default=DEFAULT_TRADE_CALENDAR,
        help=(
            "独立交易日历 parquet（列 cal_date，Tushare trade_cal）。标签的交易日轴 = 各价格来源的日期（不按股票过滤）"
            "∪ 该日历；日历里有、没有任何来源覆盖的日子使相关标签为 NaN。文件不存在时只用来源日期，并给出警告。"
        ),
    )
    args = parser.parse_args()

    snapshot_root = _resolve_under_root(args.snapshot_dir)
    trade_cal_path = _resolve_under_root(args.trade_calendar)
    if not trade_cal_path.is_file():
        log.warning("trade calendar {} not found; label axis uses source dates only", trade_cal_path)
    out_path = _resolve_under_root(args.output)
    if args.relabel_from is not None:
        return _relabel_panel(
            _resolve_under_root(args.relabel_from),
            out_path,
            snapshot_root=snapshot_root,
            raw_daily=(
                _resolve_under_root(args.price_panel)
                if args.price_panel is not None
                else PROJECT_ROOT / "data/raw/daily_prices_panel.parquet"
            ),
            through=pd.Timestamp(args.price_through) if args.price_through else None,
            keep_label_mask=args.keep_label_mask,
            trade_calendar=trade_cal_path,
        )
    log_dir = _resolve_under_root(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    raw_dates: list[date] = []
    if args.snapshot_dirs:
        parents: set[Path] = set()
        for rel in args.snapshot_dirs:
            ddir = _resolve_under_root(rel)
            if not ddir.is_dir():
                log.error("snapshot folder missing: {}", ddir)
                return 1
            if not (ddir / "meta.json").is_file():
                log.error("snapshot folder has no meta.json: {}", ddir)
                return 1
            try:
                d = date.fromisoformat(ddir.name)
            except ValueError:
                log.error("snapshot folder name must be YYYY-MM-DD: {}", ddir)
                return 1
            raw_dates.append(d)
            parents.add(ddir.resolve().parent)
        if len(parents) != 1:
            log.error(
                "--snapshot-dirs must share one parent (got {})",
                [str(x) for x in sorted(parents)],
            )
            return 1
        snapshot_root = parents.pop()
        dates_kept = _dedup_quarter(raw_dates, snapshot_root, args.dedup_strategy)
        log.info("explicit snapshot dirs → {} periods under {}", len(dates_kept), snapshot_root)
    else:
        # discover dates
        if not snapshot_root.is_dir():
            log.error("snapshot dir missing: {}", snapshot_root)
            return 1
        for p in snapshot_root.iterdir():
            if not p.is_dir():
                continue
            try:
                d = date.fromisoformat(p.name)
            except ValueError:
                continue
            if not (args.start_date <= d <= args.end_date):
                continue
            if not (p / "meta.json").exists():
                continue
            raw_dates.append(d)

        dates_kept = _dedup_quarter(raw_dates, snapshot_root, args.dedup_strategy)
    log.info("dates after filter+dedup: {} ({} periods)", dates_kept, len(dates_kept))

    base_order = _base_column_order()
    exp_order = _expansion_column_order()
    label_cols = list(LABEL_COLS)

    pipe_base = FeaturePipeline(
        groups=BASE_GROUPS,
        do_standardize=True,
        do_neutralize=True,
    )
    pipe_full = FeaturePipeline(
        groups=list(ALL_FACTOR_GROUPS.keys()),
        do_standardize=True,
        do_neutralize=True,
    )

    per_date_frames: list[pd.DataFrame] = []
    files_per_date: dict[date, int] = {}
    for as_of in dates_kept:
        snap_dir = snapshot_root / as_of.isoformat()
        meta = json.loads((snap_dir / "meta.json").read_text())
        files = meta.get("files", [])
        n_files = len(files)
        files_per_date[as_of] = n_files
        use_expansion = n_files >= EXPANSION_FILE_THRESHOLD

        snap = _load_snapshot_from_dir(snap_dir)
        # 市场级时序 z-score 依赖 data/features/*；与单点 build 行为一致
        if use_expansion:
            feat_df = pipe_full.run_single_from_snapshot(snap, as_of, universe="csi300")
        else:
            feat_df = pipe_base.run_single_from_snapshot(snap, as_of, universe="csi300")
            exp_empty = _empty_expansion_df(feat_df.index)
            feat_df = pd.concat([feat_df, exp_empty], axis=1)

        # 列顺序：基础 → 扩展（pipeline 可能打乱，统一重排）
        for c in base_order + exp_order:
            if c not in feat_df.columns:
                if c in CATEGORICAL_EXPOSURE_FIELDS:
                    feat_df[c] = pd.Series(pd.NA, index=feat_df.index, dtype="string")
                else:
                    feat_df[c] = np.nan
        feat_df = feat_df[base_order + exp_order]

        feat_df = feat_df.reset_index()
        if "ticker" not in feat_df.columns:
            raise RuntimeError("expected ticker column from features")
        feat_df["as_of"] = pd.Timestamp(as_of)
        per_date_frames.append(feat_df)

    panel_long = pd.concat(per_date_frames, ignore_index=True)
    if args.universe_file:
        uf = _resolve_under_root(args.universe_file)
        if uf.is_file():
            allowed = {ln.strip() for ln in uf.read_text(encoding="utf-8").splitlines() if ln.strip()}
            before = len(panel_long)
            panel_long = panel_long[panel_long["ticker"].astype(str).isin(allowed)]
            log.info("universe-file filter: {} → {} rows ({})", before, len(panel_long), uf)
        else:
            log.warning("universe-file not found: {}", uf)

    all_tickers = sorted(panel_long["ticker"].astype(str).unique())
    earliest = min(dates_kept)
    latest = max(dates_kept)
    raw_daily = (
        _resolve_under_root(args.price_panel)
        if args.price_panel is not None
        else PROJECT_ROOT / "data/raw/daily_prices_panel.parquet"
    )
    daily_long = _daily_panel_long(raw_daily)
    long, coverage = _load_return_sources(snapshot_root, all_tickers, extra_long=daily_long)
    pivot = build_return_index_pivot(long) if not long.empty else pd.DataFrame()
    if daily_long is None and raw_daily.is_file() and not pivot.empty:
        need_through = pd.Timestamp(latest) + timedelta(days=65)
        if pivot.index.max() < need_through:
            last = pivot.index.max()
            pivot = _extend_pivot_with_daily_parquet(
                pivot,
                all_tickers,
                raw_parquet=raw_daily,
                extend_through=need_through,
            )
            if pivot.index.max() > last:
                coverage.append((pivot.index[pivot.index > last], frozenset(map(str, pivot.columns))))
    observed = None
    if not pivot.empty:
        pivot, observed = _align_to_axis(pivot, coverage, _load_trade_calendar(trade_cal_path))
    if pivot.empty:
        log.warning("merged snapshot prices empty; falling back to Tushare fetch")
        fwd_end = latest + timedelta(days=int(63 * 1.6) + 10)
        pivot = fetch_forward_price_pivot(all_tickers, earliest, fwd_end)
    else:
        log.info(
            "forward prices from merged snapshots: {} sessions × {} tickers",
            pivot.shape[0],
            pivot.shape[1],
        )
    if pivot.empty:
        log.warning("forward price pivot empty; labels all NaN")
        for lc in label_cols:
            panel_long[lc] = np.nan
    else:
        fr_parts: list[pd.DataFrame] = []
        for as_of in dates_kept:
            fr = compute_forward_returns(pivot, as_of, (21, 63))
            if fr.empty:
                continue
            if observed is not None:  # Tushare 回退路径是单次一致抓取，不需要覆盖检查
                fr, _ = _drop_unobserved(fr, observed, as_of, (21, 63))
            fr = fr.reset_index().rename(columns={"index": "ticker"})
            fr["as_of"] = pd.Timestamp(as_of)
            fr_parts.append(fr)
        if fr_parts:
            fr_all = pd.concat(fr_parts, ignore_index=True)
            panel_long = panel_long.merge(fr_all, on=["as_of", "ticker"], how="left")
        else:
            for lc in label_cols:
                panel_long[lc] = np.nan

    for lc in label_cols:
        if lc not in panel_long.columns:
            panel_long[lc] = np.nan

    if args.extra_industry_neutralize:
        from quantmind.features.neutralize import neutralize_cross_section

        feats_only = [c for c in base_order + exp_order if c in panel_long.columns]
        rebuilt: list[pd.DataFrame] = []
        for dt in sorted(panel_long["as_of"].unique()):
            g = panel_long.loc[panel_long["as_of"] == dt].copy()
            sub = g.set_index("ticker")
            if "exposure_industry" not in sub.columns:
                rebuilt.append(g)
                continue
            nums = [c for c in feats_only if pd.api.types.is_numeric_dtype(sub[c])]
            pack = sub[nums + ["exposure_industry"]].rename(columns={"exposure_industry": "industry"})
            neu = neutralize_cross_section(pack, industry_col="industry", factors=nums)
            neu = neu.rename(columns={"industry": "exposure_industry"})
            for c in nums:
                g[c] = neu[c].reindex(g["ticker"].astype(str)).values
            rebuilt.append(g)
        panel_long = pd.concat(rebuilt, ignore_index=True)

    if args.add_sentiment_parquet:
        import re

        sp_path = _resolve_under_root(args.add_sentiment_parquet)
        if sp_path.is_file():
            sp = pd.read_parquet(sp_path)
            m = re.search(r"sentiment_scores_(\d{4}-\d{2}-\d{2})", sp_path.name)
            sad = pd.Timestamp(m.group(1)) if m else pd.Timestamp(panel_long["as_of"].max())
            sp = sp.copy()
            sp["as_of"] = sad
            cols = [c for c in ("ticker", "ts_code") if c in sp.columns]
            if not cols:
                log.warning("sentiment parquet missing ticker/ts_code")
            else:
                tcol = cols[0]
                sp = sp.rename(columns={tcol: "ticker"})
                keep = ["as_of", "ticker", "sentiment_score"]
                sp = sp[[c for c in keep if c in sp.columns]]
                panel_long = panel_long.merge(sp, on=["as_of", "ticker"], how="left")

    suffix_cols: list[str] = []
    if "sentiment_score" in panel_long.columns:
        suffix_cols.append("sentiment_score")

    panel = panel_long.set_index(["as_of", "ticker"]).sort_index()
    ordered_cols = base_order + exp_order + suffix_cols + label_cols
    for c in ordered_cols:
        if c not in panel.columns:
            panel[c] = np.nan
    panel = panel[ordered_cols]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out_path)
    log.info("wrote {} shape={}", out_path, panel.shape)

    last_two = set(sorted(dates_kept)[-2:])
    for as_of in dates_kept:
        sub = panel_long[panel_long["as_of"] == pd.Timestamp(as_of)]
        n_files = files_per_date[as_of]
        n_t = len(sub)
        feat_block = sub[base_order + exp_order]
        nn = int(feat_block.notna().to_numpy().sum())
        total_cells = n_t * feat_block.shape[1] if n_t else 0
        use_expansion = n_files >= EXPANSION_FILE_THRESHOLD
        lab21 = float(sub["forward_return_21d"].notna().mean()) if len(sub) else 0.0
        lab63 = float(sub["forward_return_63d"].notna().mean()) if len(sub) else 0.0
        mark = "last2" if as_of in last_two else ""
        print(
            f"{as_of.isoformat()} | files={n_files} | tickers={n_t} | "
            f"non_nan_features={nn}/{total_cells} | exp={'on' if use_expansion else 'nan'} | "
            f"label21={lab21:.0%} label63={lab63:.0%} {mark}".strip()
        )

    print(f"FULL_PANEL shape={panel.shape} path={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
