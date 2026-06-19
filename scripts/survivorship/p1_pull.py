"""scripts/survivorship/p1_pull.py — 幸存者修复 P1：全市场拉数（复用 backfill 基础设施）.

按已批准计划（docs/plans/survivorship_repair_plan.md）：
- Scope=A：全市场 SH+SZ A 股（list_status L+D+P），不纳入 BSE。
- 复用 scripts/backfill_tushare.py 的 token 加载 / loguru 防泄漏 / 断点续传 / 限频。
- daily / adj_factor 按 trade_date 全市场拉（含退市票），注入 backfill 配置后复用其循环。

模式：
  --probe        : 硬性中止条件检查——拉 stock_basic(L+D+P) + 抽 1 只退市票探其 daily 是否可得。
                   不可得（积分/接口不覆盖退市历史）→ 立即报告退出，不静默凑数。
  --stock-basic  : 拉 stock_basic(L+D+P) → data/lake/stock_basic_full.parquet，报票数/退市占比（排除 BSE）。
  --prices       : 拉 daily + adj_factor（全市场 by trade_date 2019-2026）→ lake，断点续传。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# 复用 backfill：其 import 时已 logging.disable + loguru.remove()（token 防泄漏）
import scripts.backfill_tushare as bf  # noqa: E402

STOCK_BASIC_FULL = REPO / "data" / "lake" / "stock_basic_full.parquet"
PRICE_START, PRICE_END = "2019-01-01", "2026-05-31"


def _is_sh_sz(ts_code: str) -> bool:
    """只保留 SH+SZ（排除 BSE 北交所：.BJ / 8xxxxx / 4xxxxx / 920xxx）。"""
    s = str(ts_code)
    if s.endswith(".BJ"):
        return False
    code = s.split(".")[0]
    return code[:1] in ("0", "3", "6") and not code.startswith(("8", "4", "92"))


def _stock_basic_all():
    """拉 L+D+P 全量（复用 provider._raw_stock_basic_all）。"""
    bf._load_token_into_env()
    from quantmind.data import tushare_provider as tp
    try:
        from loguru import logger as _lg
        _lg.remove()                      # provider import 会重注册 sink，再清一次
    except Exception:
        pass
    import pandas as pd
    sb = tp._raw_stock_basic_all()
    sb["ts_code"] = sb["ts_code"].astype(str)
    return sb


def cmd_probe() -> int:
    import pandas as pd
    sb = _stock_basic_all()
    sb_sh_sz = sb[sb["ts_code"].map(_is_sh_sz)]
    n = {s: int((sb_sh_sz["list_status"] == s).sum()) for s in ("L", "D", "P")}
    print(f"[probe] stock_basic L+D+P (SH+SZ): total={len(sb_sh_sz)} L={n['L']} D={n['D']} P={n['P']}")
    delisted = sb_sh_sz[(sb_sh_sz["list_status"] == "D") & sb_sh_sz["delist_date"].notna()]
    delisted = delisted[delisted["delist_date"] >= "20190101"]
    if delisted.empty:
        print("[probe] ⚠ 无 2019 后退市样本可探（异常，停下检查）"); return 2
    from quantmind.data import tushare_provider as tp
    ok = 0
    for _, r in delisted.head(3).iterrows():
        ts, ld, dd = r["ts_code"], str(r["list_date"]), str(r["delist_date"])
        start = max(ld, "20190101")
        px = tp._call("daily", ts_code=ts, start_date=start, end_date=dd)
        rows = 0 if px is None else len(px)
        rng = "" if (px is None or px.empty) else f"{px['trade_date'].min()}→{px['trade_date'].max()}"
        print(f"[probe] 退市票 {ts} list={ld} delist={dd} → daily rows={rows} {rng}")
        if rows > 0:
            ok += 1
    if ok == 0:
        print("\n❌ 硬性中止：Tushare 未返回退市票历史 daily（积分/接口不覆盖）。"
              "\n   按计划不得用替代源静默凑数 —— 停下报告，等指示。")
        return 3
    print(f"\n✅ probe 通过：{ok}/3 退市票 daily 可得；Tushare 覆盖退市历史。可进 --stock-basic / --prices。")
    return 0


def cmd_stock_basic() -> int:
    import pandas as pd
    sb = _stock_basic_all()
    sb_sh_sz = sb[sb["ts_code"].map(_is_sh_sz)].reset_index(drop=True)
    STOCK_BASIC_FULL.parent.mkdir(parents=True, exist_ok=True)
    sb_sh_sz.to_parquet(STOCK_BASIC_FULL)
    n = {s: int((sb_sh_sz["list_status"] == s).sum()) for s in ("L", "D", "P")}
    dl = sb_sh_sz[sb_sh_sz["delist_date"].notna() & (sb_sh_sz["delist_date"] >= "20190101")]
    print(f"[stock-basic] saved {STOCK_BASIC_FULL}")
    print(f"  total(SH+SZ)={len(sb_sh_sz)}  L={n['L']} D={n['D']} P={n['P']}")
    print(f"  2019后退市票={len(dl)}  退市占比={len(dl)/max(len(sb_sh_sz),1):.1%}")
    print(f"  对比现 v5 universe=1374 → 加回 ~{len(sb_sh_sz)-1374} 票")
    return 0


def cmd_prices(rate_interval: float) -> int:
    # 注入 daily / adj_factor 系列，复用 backfill_series（断点续传/按月flush/限频）
    bf.WINDOWS["daily"] = (PRICE_START, PRICE_END)
    bf.WINDOWS["adj_factor"] = (PRICE_START, PRICE_END)
    bf.API_NAME["daily"] = "daily"
    bf.API_NAME["adj_factor"] = "adj_factor"
    bf._load_token_into_env()
    from quantmind.data import tushare_provider as tp
    try:
        from loguru import logger as _lg
        _lg.remove()
    except Exception:
        pass
    tp._MIN_INTERVAL = max(tp._MIN_INTERVAL, rate_interval)
    print(f"=== P1 prices 拉取 min_interval={tp._MIN_INTERVAL}s ===")
    results = [bf.backfill_series("daily"), bf.backfill_series("adj_factor")]
    print("\n=== P1 prices 汇总 ===")
    for r in results:
        print(" ", r)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--stock-basic", action="store_true")
    ap.add_argument("--prices", action="store_true")
    ap.add_argument("--rate-interval", type=float, default=0.5)
    a = ap.parse_args()
    if a.probe:
        return cmd_probe()
    if a.stock_basic:
        return cmd_stock_basic()
    if a.prices:
        return cmd_prices(a.rate_interval)
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
