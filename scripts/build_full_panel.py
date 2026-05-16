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
    compute_forward_returns,
    fetch_forward_price_pivot,
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


def _extend_pivot_with_daily_parquet(
    pivot: pd.DataFrame,
    tickers: list[str],
    *,
    raw_parquet: Path,
    extend_through: pd.Timestamp,
) -> pd.DataFrame:
    """快照拼接的行情往往止于截面日，用工行长表补足后续交易日以便计算末期的 forward 标签."""
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
    merged_part = pd.concat([pivot[cols], wide[cols]], axis=0)
    merged_part = merged_part[~merged_part.index.duplicated(keep="last")].sort_index()
    new_index = pivot.index.union(merged_part.index).sort_values()
    out = pivot.reindex(index=new_index).copy()
    for c in cols:
        out[c] = merged_part[c].reindex(new_index)
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
    args = parser.parse_args()

    snapshot_root = _resolve_under_root(args.snapshot_dir)
    out_path = _resolve_under_root(args.output)
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
    label_cols = ["forward_return_21d", "forward_return_63d"]

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
    pivot = _price_pivot_from_snapshots(snapshot_root, dates_kept)
    raw_daily = (
        _resolve_under_root(args.price_panel)
        if args.price_panel is not None
        else PROJECT_ROOT / "data/raw/daily_prices_panel.parquet"
    )
    if raw_daily.is_file() and not pivot.empty:
        need_through = pd.Timestamp(latest) + timedelta(days=65)
        if pivot.index.max() < need_through:
            pivot = _extend_pivot_with_daily_parquet(
                pivot,
                all_tickers,
                raw_parquet=raw_daily,
                extend_through=need_through,
            )
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
