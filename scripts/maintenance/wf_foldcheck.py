# 触发：任何新 horizon / refit 节奏开跑前——查内部 ntest=0 fold 数必须为 0（防 fold 退化）。
"""P63-3 semi-annual fold 间距自检门：实跑前查 ntest=0 fold 数（内部必须为 0）。

semi-annual cutoffs = 每年 4 月底 / 10 月底（财报披露后 refit）2022-04→2025-10。
打印每 fold ntrain/nval/ntest + (horizon+embargo) vs cutoff 间距兼容性表。
内部 ntest=0 → FAIL（停）。尾部 ntest=0（标签耗尽）→ 自动收尾，允许。
"""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
import pandas as pd
from quantmind.data.lake import load_trading_calendar
from quantmind.backtest.wf_split import PurgedWalkForwardSplit

H, E, LOOKBACK, NVAL = 63, 63, 756, 2


def make_semiannual_cutoffs(cal, start, end):
    cal_sorted = sorted(cal); out = []
    for y in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
        for mmdd in ("04-30", "10-31"):
            d = pd.Timestamp(f"{y}-{mmdd}")
            if d < pd.Timestamp(start) or d > pd.Timestamp(end):
                continue
            after = [c for c in cal_sorted if c >= d]
            if after:
                out.append(after[0])
    return sorted(set(out))


def main():
    panel = pd.read_parquet(REPO / "data/panel/alpha_panel_weekly_v6.parquet", columns=["forward_return_63d"])
    as_of = sorted(panel.index.get_level_values("as_of").unique())
    lab = panel["forward_return_63d"]
    valid_last = max(a for a in as_of if lab.xs(a, level="as_of").notna().any())
    print(f"as_of={len(as_of)} {as_of[0].date()}→{as_of[-1].date()}; 63d标签有效末日={valid_last.date()}")

    cal = load_trading_calendar(); pos = {d: i for i, d in enumerate(sorted(cal))}
    cut = make_semiannual_cutoffs(cal, "2022-04-01", "2025-10-31")
    print(f"\nsemi-annual cutoffs ({len(cut)}): {[c.date().isoformat() for c in cut]}")
    # 间距兼容性表
    print(f"\n=== 兼容性自检: (H+E)={H+E}td vs cutoff 间距 ===")
    cal_pos = {d: i for i, d in enumerate(sorted(cal))}
    for i in range(1, len(cut)):
        gap = cal_pos[cut[i]] - cal_pos[cut[i-1]]
        ok = "OK" if gap > E else "⚠窗口将坍缩"
        print(f"  {cut[i-1].date()}→{cut[i].date()}: gap={gap}td  (need >E={E})  {ok}")

    sp = PurgedWalkForwardSplit(cal, horizon=H, embargo=E, mode="rolling",
                                rolling_lookback_td=LOOKBACK, n_val=NVAL)
    folds = sp.make_folds(as_of, cut, oos_start=pd.Timestamp("2022-01-01"), oos_end=as_of[-1])
    print(f"\n=== fold 自检表 ({len(folds)} folds) ===")
    print(f"{'fold':>4} {'cutoff':>11} {'ntrain':>7} {'nval':>5} {'ntest':>6}  test_span")
    empties = []
    for i, f in enumerate(folds):
        nt = len(f.test_dates)
        span = f"{pd.Timestamp(f.test_dates[0]).date()}..{pd.Timestamp(f.test_dates[-1]).date()}" if nt else "—"
        print(f"{i:>4} {pd.Timestamp(f.cutoff).date()!s:>11} {len(f.train_dates):>7} {len(f.val_dates):>5} {nt:>6}  {span}")
        if nt == 0:
            empties.append(i)
    # 尾部连续 ntest=0 = 自动收尾；中间 ntest=0 = 退化
    interior_empty = [i for i in empties if i < len(folds) - sum(1 for j in range(len(folds)-1, -1, -1)
                                                                 if len(folds[j].test_dates) == 0 and all(
                                                                     len(folds[k].test_dates) == 0 for k in range(j, len(folds))))]
    trailing_empty = [i for i in empties if i not in interior_empty]
    usable = [i for i in range(len(folds)) if len(folds[i].test_dates) > 0]
    total_test = sum(len(f.test_dates) for f in folds)
    print(f"\n内部 ntest=0 fold: {interior_empty}  尾部(标签耗尽自动收尾): {trailing_empty}")
    print(f"有效 fold: {len(usable)}  总 test as_of: {total_test}")
    print("=== 自检门:", "PASS ✅（内部无坍缩，尾部自动收尾）" if not interior_empty else "FAIL ❌（内部坍缩，停）", "===")


if __name__ == "__main__":
    main()
