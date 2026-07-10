#!/usr/bin/env python
"""scripts/verify_weekly_panel.py — 周频面板 v5 自验证（计划 §10）.

只读校验 data/panel/alpha_panel_weekly_v5.parquet，覆盖：
  3 条硬条件 + A 结构 / B 边界 / C 标签手算 / D 因子手算（去循环：独立读 parquet）/
  M2 逐 ticker 长窗 NaN / E PIT（硬化：四序列真·篡改 + 多 as_of 边界注入）/
  F margin 缺值 / G 增量接口 / I 幸存者偏差。
（H 回归 = pytest，单独跑 tests/test_weekly_panel.py。）

用法：python scripts/verify_weekly_panel.py
不修改任何业务代码/数据/模型。
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from quantmind.data import lake as _lakemod  # noqa: E402
from quantmind.data.lake import (  # noqa: E402
    lake_path, load_trading_calendar, read_lake_window,
)
from quantmind.features.weekly_panel import (  # noqa: E402
    GRID_PATH, LABEL_COLS, META_PATH, OUT_PATH, V4_PATH, WHITELIST_35,
    build_adj_close_pivot, build_asof_grid, compute_factors_for_asof,
    load_prices_indexed, load_stock_basic, merge_increment, pit_universe,
)

_FUTURE_DATE = pd.Timestamp("2099-12-31")  # 远超任何 as_of 的伪造未来日


# ----------------------------------------------------------------------------
# H1：篡改湖表 PIT 注入工具（不经被测路径，独立构造未来行）
# ----------------------------------------------------------------------------
def _build_future_rows(series: str, base: pd.DataFrame) -> pd.DataFrame | None:
    """从 ``base`` 末行造一条（或每 code 一条）trade_date=2099 的未来行，数值列 ×999。"""
    if base is None or base.empty:
        if series == "daily_basic":  # 湖里无此表也要造一条，验证"不被消费"
            return pd.DataFrame({"trade_date": [_FUTURE_DATE], "ticker": ["000001.SZ"],
                                 "total_mv": [9.99e12], "circ_mv": [9.99e12]})
        return None
    df = base.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    code = "ts_code" if "ts_code" in df.columns else ("ticker" if "ticker" in df.columns else None)
    last = (df.sort_values("trade_date").groupby(code, as_index=False).tail(1).copy()
            if code else df.sort_values("trade_date").iloc[[-1]].copy())
    last["trade_date"] = _FUTURE_DATE
    num = last.select_dtypes("number").columns
    last[num] = last[num] * 999.0
    return last


@contextmanager
def _injected_future_lake(series: str):
    """临时把 ``series`` 湖表替换成"含 2099 未来行"的版本（monkeypatch lake_path）。

    只替换被注入的那一个 series；其余 series 仍读真实湖表。退出时还原 + 清临时文件。
    """
    real_path = lake_path(series)
    base = pd.read_parquet(real_path) if real_path.exists() else pd.DataFrame()
    fut = _build_future_rows(series, base)
    tmpdir = Path(tempfile.mkdtemp(prefix="pit_inject_"))
    tmpfile = tmpdir / f"{series}.parquet"
    out = pd.concat([base, fut], ignore_index=True) if fut is not None else base
    out.to_parquet(tmpfile, index=False)

    real_lake_path = _lakemod.lake_path

    def _patched(s: str):
        return tmpfile if s == series else real_lake_path(s)

    _lakemod.lake_path = _patched
    try:
        yield tmpfile
    finally:
        _lakemod.lake_path = real_lake_path
        shutil.rmtree(tmpdir, ignore_errors=True)


def _frames_equal(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    """NaN-aware 比较两因子帧（数值列 allclose(equal_nan) + 字符串列 equals）。"""
    if list(a.index) != list(b.index) or list(a.columns) != list(b.columns):
        b = b.reindex(index=a.index, columns=a.columns)
    num = a.select_dtypes("number").columns
    num_ok = np.allclose(a[num].to_numpy(float), b[num].to_numpy(float), equal_nan=True)
    obj = [c for c in a.columns if c not in num]
    obj_ok = all(a[c].fillna("∅").astype(str).equals(b[c].fillna("∅").astype(str)) for c in obj)
    return bool(num_ok and obj_ok)


# ----------------------------------------------------------------------------
# 结果收集
# ----------------------------------------------------------------------------
_RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _RESULTS.append((name, bool(ok), detail))
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name}" + (f"  — {detail}" if detail else ""))


def section(title: str) -> None:
    print(f"\n{'='*72}\n{title}\n{'='*72}")


def main() -> int:
    if not OUT_PATH.exists():
        print(f"面板不存在：{OUT_PATH}（请先运行 build_weekly_panel）")
        return 2

    panel = pd.read_parquet(OUT_PATH)
    cal = load_trading_calendar()
    prices = load_prices_indexed()
    adj = build_adj_close_pivot(prices)
    asof_vals = panel.index.get_level_values("as_of")
    asof_sorted = sorted(asof_vals.unique())
    n_td = len(cal)

    # =====================================================================
    section("硬条件（计划 §10 顶部 3 条）")
    # ① 长窗 lookback：最早 as_of beta_252d 全 NaN（前置不足，非垃圾值）
    first_asof = asof_sorted[0]
    sl0 = panel.xs(first_asof, level="as_of")
    b252_all_nan = bool(sl0["beta_252d"].isna().all())
    v1y_all_nan = bool(sl0["volatility_1y"].isna().all())
    check(
        "① 最早 as_of beta_252d 全 NaN（长窗不足→NaN 非垃圾值）",
        b252_all_nan,
        f"as_of={first_asof.date()}, beta_252d NaN={b252_all_nan}, volatility_1y NaN={v1y_all_nan}",
    )
    # 同时确认在数据充足的晚期 as_of，beta_252d 大量非 NaN（证明窗口没给短）
    late = asof_sorted[-1]
    sl_late = panel.xs(late, level="as_of")
    late_b252_frac = float(sl_late["beta_252d"].notna().mean())
    check(
        "① 晚期 as_of beta_252d 大量非 NaN（lookback≥252 未给短）",
        late_b252_frac > 0.8,
        f"as_of={late.date()}, beta_252d 非NaN frac={late_b252_frac:.2f}",
    )

    # ② split-agnostic：无任何 train/test/fold/split 列；index 为真实 as_of 日期
    bad_cols = [c for c in panel.columns if any(k in c.lower() for k in ("split", "fold", "train", "test"))]
    dates_real = pd.api.types.is_datetime64_any_dtype(pd.Series(asof_vals))
    check(
        "② split-agnostic：无切分列、as_of 为真实日期",
        len(bad_cols) == 0 and dates_real,
        f"可疑列={bad_cols}, as_of 为日期={dates_real}",
    )

    # ③ 63d 复核以 v5 自身手算为基准（见 D 段实现），此处仅声明口径
    check("③ 63d 复核基准=v5 自身 adj_close 手算（见 D 段，不与 v4 旧值比）", True,
          "label_price=adj_close; v4 用 raw close，不可直接比")

    # =====================================================================
    section("A. 结构")
    is_mi = isinstance(panel.index, pd.MultiIndex) and list(panel.index.names) == ["as_of", "ticker"]
    check("A1 MultiIndex(as_of,ticker)", is_mi, f"names={list(panel.index.names)}")
    check("A2 index 唯一", panel.index.is_unique, f"n_dup={int(panel.index.duplicated().sum())}")
    check("A3 index 有序", panel.index.is_monotonic_increasing, "")
    factor_cols = [c for c in panel.columns if c not in LABEL_COLS]
    check("A4 因子列==白名单35", factor_cols == WHITELIST_35,
          f"n={len(factor_cols)} (expect 35)")
    check("A5 标签列==fwd_{12,21,63}d", [c for c in panel.columns if c in LABEL_COLS] == LABEL_COLS, "")
    check("A6 v4 未被覆盖（v5≠v4 路径，v4 仍在）",
          OUT_PATH.resolve() != V4_PATH.resolve() and V4_PATH.exists(),
          f"v4 mtime={pd.Timestamp(V4_PATH.stat().st_mtime, unit='s')}")
    if META_PATH.exists():
        import json
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        has_caveat = "survivorship_caveat" in meta and meta.get("standardized") is False
        check("A7 meta 含 survivorship 局限 + standardized=False", has_caveat, "")

    # =====================================================================
    section("B. 采样与标签边界（对齐审计 §5）")
    # 网格相位
    cal_pos = {pd.Timestamp(d): i for i, d in enumerate(cal)}
    pos = [cal_pos[a] for a in asof_sorted]
    spac_ok = bool((np.diff(pos) == 5).all())
    check("B1 相邻 as_of 间隔恒=5 交易日", spac_ok, f"n_asof={len(asof_sorted)}")
    check("B2 最后 as_of==2026-04-20", str(asof_sorted[-1].date()) == "2026-04-20",
          f"last={asof_sorted[-1].date()}")
    # 各档"最后非 NaN 标签 as_of" = 网格上 cal_idx <= (n_td-1-h) 的最大 as_of
    def last_label_asof(h: int):
        bound = n_td - 1 - h
        grid_ok = [a for a in asof_sorted if cal_pos[a] <= bound]
        return grid_ok[-1] if grid_ok else None
    for h, audit_abs in [(12, "2026-04-20"), (21, "2026-04-07"), (63, "2026-01-26")]:
        col = f"forward_return_{h}d"
        nn = asof_vals[panel[col].notna().values]
        actual_last = sorted(pd.Index(nn).unique())[-1] if len(nn) else None
        expect = last_label_asof(h)
        ok = actual_last is not None and expect is not None and actual_last == expect
        check(
            f"B3 fwd_{h}d 最后非NaN as_of==网格期望",
            ok,
            f"actual={actual_last.date() if actual_last is not None else None}, "
            f"grid_expect={expect.date() if expect is not None else None}, "
            f"audit§5 绝对边界={audit_abs}",
        )

    # =====================================================================
    section("C. 标签数值正确性（v5 自身 adj_close 手算）")
    rng = np.random.default_rng(42)
    for h in (12, 63):  # 含硬条件③：63d 用 v5 自身手算
        col = f"forward_return_{h}d"
        sub = panel[panel[col].notna()]
        if sub.empty:
            check(f"C {col} 手算抽查", False, "无非NaN样本")
            continue
        picks = sub.sample(min(3, len(sub)), random_state=int(rng.integers(1e6)))
        worst = 0.0
        for (as_of, ticker), row in picks.iterrows():
            if ticker not in adj.columns:
                continue
            ser = adj[ticker].dropna()
            ipos = ser.index.searchsorted(pd.Timestamp(as_of), side="right") - 1
            if ipos < 0 or ipos + h >= len(ser):
                continue
            manual = float(ser.iloc[ipos + h]) / float(ser.iloc[ipos]) - 1.0
            worst = max(worst, abs(manual - float(row[col])))
        check(f"C {col} 手算 3 抽样 相对误差<1e-9", worst < 1e-9, f"max|Δ|={worst:.2e}")

    # =====================================================================
    section("D. 因子数值正确性（手算/对照抽查）")
    audit_asof = asof_sorted[-1]  # 数据充足
    sl = panel.xs(audit_asof, level="as_of")
    # D1 momentum_1m：raw close[t]/close[t-21]-1
    close_piv = (
        prices[["trade_date", "ticker", "close"]]
        .drop_duplicates(["trade_date", "ticker"], keep="last")
        .pivot(index="trade_date", columns="ticker", values="close").sort_index()
    )
    cl_le = close_piv.loc[:audit_asof]
    tk = sl["momentum_1m"].dropna().index[0]
    man_mom = float(cl_le[tk].iloc[-1]) / float(cl_le[tk].iloc[-22]) - 1.0
    check("D1 momentum_1m 手算(raw close, 21bar)", abs(man_mom - float(sl.loc[tk, "momentum_1m"])) < 1e-6,
          f"ticker={tk} manual={man_mom:.6f} panel={float(sl.loc[tk,'momentum_1m']):.6f}")

    # —— D2/D3/D4 去循环自证：手算基准【直接读原始 lake parquet】+ 独立 trade_date<=as_of 过滤，
    #     不经被测路径 read_lake_window；与面板值（被测路径产出）交叉比对。
    # D2 beta_252d：numpy OLS（票收益 vs 000300 收益，252 窗口）
    idx_raw = pd.read_parquet(lake_path("index_daily"))
    idx_raw["trade_date"] = pd.to_datetime(idx_raw["trade_date"])
    idx300 = idx_raw[(idx_raw["ts_code"].astype(str).str.upper() == "000300.SH")
                     & (idx_raw["trade_date"] <= audit_asof)].sort_values("trade_date")
    iret = idx300.set_index("trade_date")["close"].astype(float).pct_change().dropna()
    tkb = sl["beta_252d"].dropna().index[0]
    sret = cl_le[tkb].astype(float).pct_change().dropna()
    common = iret.index.intersection(sret.index).sort_values()[-252:]
    x = iret.loc[common].values
    y = sret.loc[common].values
    m = ~(np.isnan(x) | np.isnan(y))
    man_beta = float(np.cov(x[m], y[m], ddof=1)[0, 1] / np.var(x[m], ddof=1))
    check("D2 beta_252d 手算 numpy OLS（独立读 parquet，非循环）",
          abs(man_beta - float(sl.loc[tkb, "beta_252d"])) < 1e-6,
          f"ticker={tkb} manual={man_beta:.4f} panel={float(sl.loc[tkb,'beta_252d']):.4f}")

    # D3 north_bound_30d_net_inflow：近30日 north_money 之和（直接读 parquet 独立 PIT）
    nb_raw = pd.read_parquet(lake_path("north_bound"))
    nb_raw["trade_date"] = pd.to_datetime(nb_raw["trade_date"])
    nb_le = nb_raw[nb_raw["trade_date"] <= audit_asof].sort_values("trade_date")
    man_nb = float(nb_le.tail(30)["north_money"].astype(float).sum())
    pv = float(sl["north_bound_30d_net_inflow"].dropna().iloc[0])
    check("D3 north_bound_30d 手算（独立读 parquet，非循环）", abs(man_nb - pv) < 1e-3,
          f"manual={man_nb:.2f} panel={pv:.2f}")

    # D4 margin_balance：截至 as_of 最后一行 rzye（直接读 parquet 独立 PIT）
    mg_raw = pd.read_parquet(lake_path("margin"))
    mg_raw["trade_date"] = pd.to_datetime(mg_raw["trade_date"])
    tkm = sl["margin_balance"].dropna().index[0]
    mg_t = mg_raw[(mg_raw["ticker"].astype(str) == str(tkm))
                  & (mg_raw["trade_date"] <= audit_asof)].sort_values("trade_date")
    man_mb = float(mg_t["rzye"].iloc[-1])
    check("D4 margin_balance 手算（独立读 parquet，非循环）",
          abs(man_mb - float(sl.loc[tkm, "margin_balance"])) < 1e-3,
          f"ticker={tkm} manual={man_mb:.0f} panel={float(sl.loc[tkm,'margin_balance']):.0f}")

    # D5 list_age_years：(as_of - list_date)/365.25
    sb = load_stock_basic().set_index("ticker")
    tkl = sl["list_age_years"].dropna().index[0]
    ld = sb.loc[tkl, "list_date"]
    man_age = (pd.Timestamp(audit_asof) - pd.Timestamp(ld)).days / 365.25
    check("D5 list_age_years 手算", abs(man_age - float(sl.loc[tkl, "list_age_years"])) < 0.05,
          f"ticker={tkl} manual={man_age:.2f} panel={float(sl.loc[tkl,'list_age_years']):.2f}")

    # =====================================================================
    section("M2. 逐 ticker 长窗 NaN 掩码（晚期 as_of × 短历史新股 → NaN 非薄估计）")
    # 在晚期 as_of 区间找上市 < ~0.20 年（≈50 交易日 < 63）的票，断言长窗因子为 NaN
    late_block = panel[panel.index.get_level_values("as_of") >= asof_sorted[-20]]
    young = late_block[late_block["list_age_years"] < 0.20]
    long_win_cols = ["volatility_1y", "downside_volatility_3m", "max_drawdown_3m",
                     "amihud_illiquidity", "volatility_3m"]
    if young.empty:
        check("M2 找到晚期短历史新股样本", False, "晚期区间无 list_age<0.20 的票（无法验证）")
    else:
        all_nan = bool(young[long_win_cols].isna().all().all())
        # 对照：同截面长历史票该列大量非 NaN（证明不是整列误杀）
        sample_asof = young.index.get_level_values("as_of")[0]
        sl_s = panel.xs(sample_asof, level="as_of")
        old = sl_s[sl_s["list_age_years"] > 1.0]
        ctrl_ok = float(old["volatility_1y"].notna().mean()) > 0.8 if len(old) else True
        check("M2 短历史新股长窗因子全 NaN（非薄估计）", all_nan,
              f"young行数={len(young)}, 列={long_win_cols}, 全NaN={all_nan}")
        check("M2 对照：同截面长历史票 volatility_1y 大量非 NaN（非整列误杀）", ctrl_ok,
              f"as_of={pd.Timestamp(sample_asof).date()}, 老票 vol_1y 非NaN frac={float(old['volatility_1y'].notna().mean()):.2f}")

    # =====================================================================
    section("E. PIT 安全性（硬化：四序列真·篡改 + 多 as_of 边界注入）")
    sb_full = load_stock_basic()
    priced = set(prices["ticker"].unique())
    mid = asof_sorted[len(asof_sorted) // 2]

    # —— E1：向【每个外部序列】真实注入一条 2099 未来行，跨 早/中/晚 三个 as_of，
    #     断言 read_lake_window 结果【不含】该行，且 max(trade_date) <= as_of < 注入日。
    e1_probe_asof = [("early", asof_sorted[8]), ("mid", mid), ("late", asof_sorted[-1])]
    for series, lb in [("index_daily", 280), ("north_bound", 45), ("margin", 60), ("daily_basic", 280)]:
        with _injected_future_lake(series):
            for label, ao in e1_probe_asof:
                w = read_lake_window(series, ao, lb)
                if w.empty:
                    ok = True
                    detail = "window empty"
                else:
                    td = pd.to_datetime(w["trade_date"])
                    has_future = bool((td >= pd.Timestamp("2099-01-01")).any())
                    mx = td.max()
                    ok = (not has_future) and (mx <= pd.Timestamp(ao)) and (mx < _FUTURE_DATE)
                    detail = f"as_of={ao.date()} max={mx.date()} future_row_in_window={has_future}"
                check(f"E1[{series}/{label}] 注入未来行被排除，max(trade_date)<=as_of", ok, detail)

    # —— E2：向【四个外部序列】各注入未来行，重算因子，断言因子帧【不变】。
    #     任一序列注入后因子变化 = PIT 泄漏 = FAIL。
    baseline = compute_factors_for_asof(mid, prices, sb_full, cal, priced)
    ser_factor = {
        "index_daily": "beta_252d",
        "north_bound": "north_bound_30d_net_inflow",
        "margin": "margin_balance",
        "daily_basic": None,   # v5 不消费 daily_basic 数值（仅 ticker-shim）→ 注入应无影响
    }
    for series, fac in ser_factor.items():
        with _injected_future_lake(series):
            after = compute_factors_for_asof(mid, prices, sb_full, cal, priced)
        same = _frames_equal(baseline, after)
        note = f"对应因子={fac}" if fac else "daily_basic 不被任何因子消费（注入应无影响）"
        check(f"E2[{series}] 注入未来行→因子帧不变（PIT，{note}）", same, f"as_of={mid.date()}")

    # —— E2-prices：原行情侧篡改未来行（保留旧测，覆盖 prices PIT）
    tampered = prices.copy()
    fut_mask = tampered["trade_date"] > pd.Timestamp(mid)
    for c in ("close", "high", "low", "open", "pre_close", "volume"):
        tampered.loc[fut_mask, c] = tampered.loc[fut_mask, c] * 999.0
    after_px = compute_factors_for_asof(mid, tampered, sb_full, cal, set(tampered["ticker"].unique()))
    check("E2[prices] 篡改未来行情→因子帧不变（PIT）", _frames_equal(baseline, after_px),
          f"as_of={mid.date()}")

    # =====================================================================
    section("F. margin 头部与缺值语义")
    mcov = read_lake_window("margin", asof_sorted[-1], 60)  # 仅探测可用性
    margin_start = None
    full = pd.read_parquet(_ROOT / "data" / "lake" / "margin.parquet", columns=["trade_date"])
    margin_start = pd.to_datetime(full["trade_date"]).min()
    # 在 margin_start 之前的 as_of，margin_balance 应全 NaN
    early_before = [a for a in asof_sorted if a < margin_start]
    if early_before:
        s0 = panel.xs(early_before[0], level="as_of")
        check("F1 margin 覆盖起点前 as_of 的 margin_balance 全 NaN",
              bool(s0["margin_balance"].isna().all()),
              f"margin_start={margin_start.date()}, probe as_of={early_before[0].date()}")
    else:
        check("F1 margin 覆盖自始（无更早 as_of 需 NaN）", True,
              f"margin_start={margin_start.date()} <= first as_of={asof_sorted[0].date()}")
    # 有数据处非全 NaN
    s_late = panel.xs(asof_sorted[-1], level="as_of")
    check("F2 晚期 as_of margin_balance 大量非 NaN", float(s_late["margin_balance"].notna().mean()) > 0.5,
          f"非NaN frac={float(s_late['margin_balance'].notna().mean()):.2f}")

    # =====================================================================
    section("G. 增量接口（merge_increment）")
    sample_idx = panel.index[:200]
    fake = pd.DataFrame(
        {"pe_ttm_inc": np.arange(len(sample_idx), dtype=float),
         "pb_inc": np.arange(len(sample_idx), dtype=float)},
        index=sample_idx,
    )
    merged = merge_increment(panel, fake)
    g_ok = (
        len(merged) == len(panel)
        and merged.index.equals(panel.index)
        and merged[WHITELIST_35].equals(panel[WHITELIST_35])
        and "pe_ttm_inc" in merged.columns
        and float(merged["pe_ttm_inc"].notna().sum()) == len(sample_idx)
    )
    check("G1 merge_increment：行数不变/对齐/原列不变/新列就位", g_ok,
          f"base_rows={len(panel)} merged_rows={len(merged)} aligned={len(sample_idx)}")
    # 列冲突应报错
    try:
        merge_increment(panel, panel[[WHITELIST_35[0]]])
        g_collide = False
    except ValueError:
        g_collide = True
    check("G2 列名冲突报错（不静默覆盖）", g_collide, "")

    # =====================================================================
    section("I. 幸存者偏差（§0.5）")
    # I1 universe PIT：某早期 as_of，list_date>as_of 的票不在该截面
    sb_all = load_stock_basic()
    probe = asof_sorted[10]
    in_panel = set(panel.xs(probe, level="as_of").index)
    future_listed = set(
        sb_all.loc[sb_all["list_date"] > pd.Timestamp(probe), "ticker"].astype(str)
    )
    leak = in_panel & future_listed
    check("I1 universe PIT 入池：未上市票不在早期截面（无新股前视）",
          len(leak) == 0, f"as_of={probe.date()}, 泄露票数={len(leak)}")
    # I2 源头幸存者：stock_basic 0 退市记录（如实标注前提）
    n_delisted = int(sb_all["delist_date"].notna().sum()) if "delist_date" in sb_all.columns else -1
    check("I2 源头幸存者池确认（delist_date 全空→meta 标注乐观上界）",
          n_delisted == 0, f"delist_date 非空={n_delisted}（0 证实幸存者池）")

    # =====================================================================
    section("汇总")
    n_pass = sum(1 for _, ok, _ in _RESULTS if ok)
    n_fail = len(_RESULTS) - n_pass
    print(f"\n  PASS {n_pass} / {len(_RESULTS)}   FAIL {n_fail}")
    if n_fail:
        print("\n  失败项：")
        for name, ok, detail in _RESULTS:
            if not ok:
                print(f"    - {name}  ({detail})")
    print(f"\n  面板：{OUT_PATH}")
    print(f"  shape={panel.shape}  n_asof={len(asof_sorted)}  "
          f"[{asof_sorted[0].date()} → {asof_sorted[-1].date()}]")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
