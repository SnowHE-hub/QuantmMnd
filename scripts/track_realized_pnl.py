"""实际收益追踪器 — Week 1 学习循环基础设施.

功能
====
1. 扫描 data/recommendations/ 下所有推荐记录
2. 计算从推荐日起 63 个交易日后的实际持仓收益
3. 与预测排名对比，计算 realized IC（实际收益 vs 预测分数的 Spearman）
4. 输出到 data/feedback/realized_pnl.parquet（DPO 偏好对原材料）

输出字段
========
  as_of_date        推荐日
  ticker            股票代码
  predicted_rank    lgbm_score 排名（1=最高分）
  predicted_score   lgbm_score 原始值
  entry_price       推荐日收盘价
  exit_price        63 交易日后收盘价
  actual_return_63d 实际收益率
  actual_rank       在推荐宇宙内的实际收益排名（1=最好）
  pnl_vs_median     实际收益 - 同期宇宙中位数（超额）
  hit               bool，actual_rank <= predicted_rank（"猜对了"）

汇总指标（data/feedback/realized_ic.json）
==========================================
  as_of → realized_ic  (Spearman 预测分 vs 实际收益)
  overall_ic            全部推荐的合并 IC
  overall_hit_rate      hit=True 占比
  n_records             总记录数

用法
====
  python scripts/track_realized_pnl.py \\
      --recs   data/recommendations/ \\
      --prices data/raw/alpha_prices_panel.parquet \\
      --out    data/feedback/ \\
      [--holding-days 63] [--panel data/panel/alpha_panel_v4.parquet]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)  # type: ignore[assignment]

HOLDING_DAYS = 63   # 交易日数


# ─────────────────────────────────────────────────────────────────────────────
# 数据加载
# ─────────────────────────────────────────────────────────────────────────────

def load_price_pivot(prices_path: Path) -> pd.DataFrame:
    """加载价格面板，pivot 为 (trade_date × ts_code) 宽表."""
    df = pd.read_parquet(prices_path, columns=["ts_code", "trade_date", "close"])
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    wide = df.pivot_table(index="trade_date", columns="ts_code", values="close", aggfunc="last")
    wide = wide.sort_index()
    logger.info(f"价格宽表: {wide.shape}（{wide.index[0].date()} → {wide.index[-1].date()}）")
    return wide


def scan_recommendation_files(recs_dir: Path) -> list[dict]:
    """扫描推荐目录，收集所有 JSON 推荐记录."""
    records = []

    # 格式 1: data/recommendations/{date}.json（扁平 JSON）
    for fp in sorted(recs_dir.glob("*.json")):
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
            as_of = data.get("as_of")
            top10 = data.get("top10", [])
            if as_of and top10:
                records.append({"as_of": as_of, "top10": top10, "source": str(fp)})
        except Exception:
            pass

    # 格式 2: data/recommendations/{date}/top10.json（子目录）
    for subdir in sorted(recs_dir.iterdir()):
        if not subdir.is_dir():
            continue
        fp = subdir / "top10.json"
        if not fp.exists():
            continue
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
            as_of = data.get("as_of") or subdir.name
            top10 = data.get("top10", [])
            if as_of and top10:
                # 去重（若上面已读过同日期的扁平 JSON）
                if not any(r["as_of"] == as_of for r in records):
                    records.append({"as_of": as_of, "top10": top10, "source": str(fp)})
        except Exception:
            pass

    logger.info(f"找到 {len(records)} 条推荐记录")
    return records


def get_trading_date(idx: pd.DatetimeIndex, target: pd.Timestamp, direction: int = 1) -> pd.Timestamp | None:
    """在交易日索引中找到最近的有效日期（direction=1 向后，-1 向前）."""
    if target in idx:
        return target
    if direction == 1:
        future = idx[idx >= target]
        return future[0] if len(future) > 0 else None
    else:
        past = idx[idx <= target]
        return past[-1] if len(past) > 0 else None


def nth_trading_day_after(idx: pd.DatetimeIndex, start: pd.Timestamp, n: int) -> pd.Timestamp | None:
    """从 start 之后的第 n 个交易日."""
    start_pos = idx.searchsorted(start)
    target_pos = start_pos + n
    if target_pos >= len(idx):
        return None
    return idx[target_pos]


# ─────────────────────────────────────────────────────────────────────────────
# 核心计算
# ─────────────────────────────────────────────────────────────────────────────

def compute_realized_pnl_for_rec(
    rec: dict,
    price_wide: pd.DataFrame,
    holding_days: int = HOLDING_DAYS,
) -> list[dict] | None:
    """计算单条推荐记录的实际收益.

    Returns:
        list of row dicts，每条对应一只股票；若无法计算则返回 None。
    """
    as_of_str = str(rec["as_of"])
    top10 = rec["top10"]
    if not top10:
        return None

    as_of_ts = pd.Timestamp(as_of_str)
    trading_idx = price_wide.index

    # 入场日：推荐日后第一个交易日
    entry_date = get_trading_date(trading_idx, as_of_ts, direction=1)
    if entry_date is None:
        logger.warning(f"{as_of_str}: 找不到入场交易日")
        return None

    # 出场日：持有 holding_days 个交易日后
    exit_date = nth_trading_day_after(trading_idx, entry_date, holding_days)
    if exit_date is None:
        logger.info(f"{as_of_str}: 持有期未满（需等到 {holding_days} 交易日后），跳过")
        return None

    rows = []
    for item in top10:
        ticker = item.get("ticker") or item.get("ts_code")
        if not ticker:
            continue

        score = item.get("lgbm_score") or item.get("lgbm_score_raw") or item.get("score")
        pred_rank = item.get("rank") or item.get("lgbm_rank")
        if score is None:
            continue

        if ticker not in price_wide.columns:
            continue

        entry_price = price_wide.loc[entry_date, ticker]
        exit_price = price_wide.loc[exit_date, ticker]

        if pd.isna(entry_price) or pd.isna(exit_price) or entry_price <= 0:
            continue

        actual_return = float((exit_price - entry_price) / entry_price)

        rows.append({
            "as_of_date": as_of_str,
            "ticker": ticker,
            "entry_date": str(entry_date.date()),
            "exit_date": str(exit_date.date()),
            "holding_days": holding_days,
            "predicted_rank": pred_rank,
            "predicted_score": float(score),
            "entry_price": float(entry_price),
            "exit_price": float(exit_price),
            "actual_return_63d": round(actual_return, 6),
            "key_factors": json.dumps(item.get("key_factors", {}), ensure_ascii=False),
        })

    if not rows:
        return None

    # 计算组内排名和超额
    actual_rets = [r["actual_return_63d"] for r in rows]
    sorted_rets = sorted(actual_rets, reverse=True)
    median_ret = float(np.median(actual_rets))

    for r in rows:
        r["actual_rank"] = sorted_rets.index(r["actual_return_63d"]) + 1
        r["pnl_vs_median"] = round(r["actual_return_63d"] - median_ret, 6)
        r["hit"] = (
            r["actual_rank"] <= r["predicted_rank"]
            if r["predicted_rank"] is not None
            else None
        )

    return rows


def compute_realized_ic(rows: list[dict]) -> float:
    """计算一组推荐的 realized IC（Spearman 预测分 vs 实际收益）."""
    scores = [r["predicted_score"] for r in rows]
    rets = [r["actual_return_63d"] for r in rows]
    if len(scores) < 3:
        return float("nan")
    rho, _ = stats.spearmanr(scores, rets)
    return float(rho) if np.isfinite(rho) else float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def run(
    recs_dir: Path,
    prices_path: Path,
    out_dir: Path,
    holding_days: int = HOLDING_DAYS,
    panel_path: Path | None = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 加载价格
    price_wide = load_price_pivot(prices_path)

    # 如果有 panel，用 panel 里的 forward_return_63d 做交叉验证
    panel_returns: dict[tuple[str, str], float] = {}
    if panel_path and panel_path.exists():
        logger.info(f"加载面板（用于交叉验证）：{panel_path}")
        panel = pd.read_parquet(panel_path, columns=["forward_return_63d"])
        for (as_of, ticker), row in panel.iterrows():
            if pd.notna(row["forward_return_63d"]):
                panel_returns[(str(as_of.date()), ticker)] = float(row["forward_return_63d"])

    # 扫描推荐记录
    recs = scan_recommendation_files(recs_dir)
    if not recs:
        logger.error("未找到任何推荐记录，退出")
        return {}

    all_rows: list[dict] = []
    ic_by_date: dict[str, float] = {}

    for rec in recs:
        rows = compute_realized_pnl_for_rec(rec, price_wide, holding_days)
        if not rows:
            continue

        # 交叉验证：与 panel 里的 forward_return_63d 对比
        for r in rows:
            key = (r["as_of_date"], r["ticker"])
            if key in panel_returns:
                r["panel_return_63d"] = round(panel_returns[key], 6)
                r["return_diff"] = round(r["actual_return_63d"] - panel_returns[key], 6)
            else:
                r["panel_return_63d"] = None
                r["return_diff"] = None

        ic = compute_realized_ic(rows)
        ic_by_date[rec["as_of"]] = round(ic, 6) if np.isfinite(ic) else None

        logger.info(
            f"  {rec['as_of']}: {len(rows)} 只股票，"
            f"平均收益={np.mean([r['actual_return_63d'] for r in rows])*100:.2f}%，"
            f"realized IC={ic:.4f}" if np.isfinite(ic) else
            f"  {rec['as_of']}: {len(rows)} 只股票，realized IC=N/A"
        )
        all_rows.extend(rows)

    if not all_rows:
        logger.warning("所有推荐记录均无法计算（持有期未满或价格缺失）")
        return {}

    # ── 保存 parquet ──────────────────────────────────────────────────────────
    out_path = out_dir / "realized_pnl.parquet"
    df = pd.DataFrame(all_rows)
    df["as_of_date"] = pd.to_datetime(df["as_of_date"])
    df = df.sort_values(["as_of_date", "actual_rank"])
    df.to_parquet(out_path, index=False)
    logger.info(f"✅ 实际收益记录：{len(df)} 条 → {out_path}")
    # ── DB 双写（全量覆盖，失败不中断业务）──────────────────────────────────
    try:
        from app.db.writers import get_writer
        get_writer().write_realized_pnl(df, full_replace=True)
    except Exception as _dw_e:
        logger.warning(f"realized_pnl DB 双写跳过: {_dw_e}")

    # ── 汇总指标 ──────────────────────────────────────────────────────────────
    valid_ics = [v for v in ic_by_date.values() if v is not None]
    hit_col = df["hit"].dropna()
    overall_ic = float(np.mean(valid_ics)) if valid_ics else float("nan")
    overall_hit = float(hit_col.mean()) if len(hit_col) > 0 else float("nan")

    summary = {
        "n_records": len(df),
        "n_rec_dates": len(ic_by_date),
        "holding_days": holding_days,
        "realized_ic_by_date": ic_by_date,
        "overall_ic": round(overall_ic, 6) if np.isfinite(overall_ic) else None,
        "overall_hit_rate": round(overall_hit, 4) if np.isfinite(overall_hit) else None,
        "avg_actual_return": round(float(df["actual_return_63d"].mean()), 6),
        "pct_positive": round(float((df["actual_return_63d"] > 0).mean()), 4),
    }
    ic_path = out_dir / "realized_ic.json"
    ic_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── 打印汇总 ──────────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print(f"  推荐日期数:       {summary['n_rec_dates']}")
    print(f"  总记录数:         {summary['n_records']}")
    print(f"  整体 Realized IC: {summary['overall_ic']}")
    print(f"  命中率（hit）:     {summary['overall_hit_rate']}")
    print(f"  平均实际收益:      {summary['avg_actual_return']*100:.2f}%")
    print(f"  正收益比例:        {summary['pct_positive']*100:.1f}%")
    print("─" * 60)
    for date_str, ic in sorted(ic_by_date.items()):
        ic_str = f"{ic:+.4f}" if ic is not None else "  N/A "
        print(f"  {date_str}: IC={ic_str}")
    print("═" * 60)
    print(f"\n  结果保存：{out_dir}")

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recs",   type=Path, default=Path("data/recommendations/"),
                   help="推荐记录目录")
    p.add_argument("--prices", type=Path, default=Path("data/raw/alpha_prices_panel.parquet"),
                   help="价格面板（含 close 列）")
    p.add_argument("--out",    type=Path, default=Path("data/feedback/"),
                   help="输出目录")
    p.add_argument("--panel",  type=Path, default=None,
                   help="因子面板（可选，用于与 forward_return_63d 交叉验证）")
    p.add_argument("--holding-days", type=int, default=HOLDING_DAYS,
                   help="持有天数（交易日，默认 63）")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run(
        recs_dir=args.recs,
        prices_path=args.prices,
        out_dir=args.out,
        holding_days=args.holding_days,
        panel_path=args.panel,
    )


if __name__ == "__main__":
    main()
