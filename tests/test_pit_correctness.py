"""tests/test_pit_correctness.py — Point-in-Time 严格性测试套件.

这些测试调用真实 API（akshare/tushare），运行较慢。
执行：

    pytest -m pit          # 跑 PIT 测试
    pytest -m "not slow"   # 跳过 PIT 与其它慢测

10 个核心 PIT 严格性测试：

1.  T-Price-PIT          : Tushare daily ≤ as_of
2.  A-Price-PIT          : Akshare daily ≤ as_of
3.  T-Income-PIT         : Tushare income.f_ann_date ≤ as_of
4.  T-Balance-PIT        : Tushare balance_sheet.f_ann_date ≤ as_of
5.  T-Cashflow-PIT       : Tushare cashflow.f_ann_date ≤ as_of
6.  A-Income-PIT         : Akshare income (NOTICE_DATE) ≤ as_of
7.  Universe-Historical  : csi300 @2018 ≠ csi300 @2024 (反 survivorship bias)
8.  Universe-CrossCheck  : tushare 与 akshare 当前 csi300 一致性 ≥95%
9.  T-vs-A-FinDate       : tushare f_ann_date 与 akshare NOTICE_DATE 一致性
10. PIT-Cutoff-Boundary  : 已知报告期 (2023年报披露日 ~2024-04) 在边界两侧表现
"""

from __future__ import annotations

import os
from datetime import date

import pandas as pd
import pytest

# ---- 共用 skip 条件 ----
_REQUIRES_NET = pytest.mark.skipif(
    os.getenv("CI") and not os.getenv("RUN_NETWORK_TESTS"),
    reason="network tests skipped in CI by default",
)

_REQUIRES_TUSHARE = pytest.mark.skipif(
    not os.getenv("TUSHARE_TOKEN"), reason="TUSHARE_TOKEN not set"
)


# ============================================================================
# 1. Price PIT — Tushare
# ============================================================================


@pytest.mark.pit
@pytest.mark.slow
@_REQUIRES_TUSHARE
def test_tushare_price_pit_strict() -> None:
    """断言 tushare daily 不返回 > as_of 的行情."""
    from quantmind.data import TushareProvider

    p = TushareProvider()
    as_of = date(2024, 6, 30)
    df = p.get_price("300750.SZ", "2024-01-01", "2024-12-31", as_of=as_of)
    assert not df.empty
    assert df["trade_date"].max() <= pd.Timestamp(as_of)


# ============================================================================
# 2. Price PIT — Akshare
# ============================================================================


@pytest.mark.pit
@pytest.mark.slow
@pytest.mark.integration
def test_akshare_price_pit_strict() -> None:
    from quantmind.data import AkshareProvider
    from quantmind.data.base import DataProviderError

    p = AkshareProvider()
    as_of = date(2024, 6, 30)
    try:
        df = p.get_price("300750.SZ", "2024-01-01", "2024-12-31", as_of=as_of)
    except DataProviderError as e:
        pytest.skip(f"AkShare 网络/数据源不可用（非 PIT 逻辑失败）：{e}")
    assert not df.empty
    assert df["trade_date"].max() <= pd.Timestamp(as_of)


# ============================================================================
# 3-5. Tushare 三大报表 f_ann_date PIT
# ============================================================================


@pytest.mark.pit
@pytest.mark.slow
@_REQUIRES_TUSHARE
@pytest.mark.parametrize("statement", ["income", "balance_sheet", "cashflow"])
def test_tushare_financials_f_ann_date_pit(statement: str) -> None:
    """断言所有财报行的 f_ann_date 严格 ≤ as_of."""
    from quantmind.data import TushareProvider

    p = TushareProvider()
    as_of = date(2024, 4, 1)  # 2023 年报多在 4 月 中下旬披露
    df = p.get_financials("300750.SZ", statement, as_of=as_of)
    assert not df.empty, f"{statement} should have data"
    assert "f_ann_date" in df.columns
    assert df["f_ann_date"].max() <= pd.Timestamp(as_of), (
        f"{statement} has f_ann_date > {as_of}: {df['f_ann_date'].max()}"
    )


# ============================================================================
# 6. Akshare income (NOTICE_DATE → f_ann_date) PIT
# ============================================================================


@pytest.mark.pit
@pytest.mark.slow
def test_akshare_income_notice_date_pit() -> None:
    from quantmind.data import AkshareProvider

    p = AkshareProvider()
    as_of = date(2024, 4, 1)
    df = p.get_financials("300750.SZ", "income", as_of=as_of)
    assert not df.empty
    assert "f_ann_date" in df.columns  # akshare 的 NOTICE_DATE 经映射重命名后
    assert df["f_ann_date"].max() <= pd.Timestamp(as_of)


# ============================================================================
# 7. Universe historical correctness (反 survivorship bias)
# ============================================================================


@pytest.mark.pit
@pytest.mark.slow
@_REQUIRES_TUSHARE
def test_universe_changes_over_time() -> None:
    """csi300 在 2020 vs 2024 应有显著差异（>=20只换仓）."""
    from quantmind.data import get_universe

    u_2020 = set(get_universe("csi300", as_of=date(2020, 6, 30)))
    u_2024 = set(get_universe("csi300", as_of=date(2024, 6, 30)))
    assert len(u_2020) >= 290
    assert len(u_2024) >= 290
    diff = u_2020.symmetric_difference(u_2024)
    assert len(diff) >= 40, (
        f"csi300 universe changed too little ({len(diff)}); survivorship bias suspect"
    )
    only_2020 = u_2020 - u_2024
    assert len(only_2020) >= 20, "几乎没有 2020 在 2024 退出的票，PIT 数据可能有问题"


# ============================================================================
# 8. Universe cross-validation (tushare vs akshare current)
# ============================================================================


@pytest.mark.pit
@pytest.mark.slow
@_REQUIRES_TUSHARE
def test_universe_cross_validate_current() -> None:
    """当前时点 tushare 与 akshare csi300 应 ≥95% 一致."""
    from quantmind.data import cross_validate_universe

    res = cross_validate_universe("csi300", as_of=date.today())
    assert res["tushare_count"] >= 290
    if res["akshare_count"] >= 290:  # akshare 偶尔失败
        assert res["agreement_ratio"] >= 0.95, (
            f"tushare vs akshare csi300 mismatch: ratio={res['agreement_ratio']:.3f}; "
            f"only_tushare={res['only_tushare']}, only_akshare={res['only_akshare']}"
        )


# ============================================================================
# 9. Tushare f_ann_date vs Akshare NOTICE_DATE 一致性
# ============================================================================


@pytest.mark.pit
@pytest.mark.slow
@_REQUIRES_TUSHARE
def test_tushare_vs_akshare_f_ann_date_alignment_post_ipo() -> None:
    """对同一 ticker / 同一**上市后**报告期，两源披露日应相差 <= 7 天.

    注意：上市前的历史财报，tushare 给 IPO 招股书披露日，akshare 给更早的口径，
    会出现几个月差异；这是合理且预期的行为。本测试聚焦 IPO 后年度。

    (tushare f_ann_date 是修订后的实际公告日，akshare NOTICE_DATE 是首次披露日)
    """
    from quantmind.data import AkshareProvider, TushareProvider

    ticker = "600519.SH"  # 茅台 — 上市于 2001-08-27，所有报告都是上市后
    as_of = date(2024, 12, 31)
    t_df = TushareProvider().get_financials(ticker, "income", as_of=as_of)
    a_df = AkshareProvider().get_financials(ticker, "income", as_of=as_of)

    # 取最近 5 年报告期（避免年代久远的历史口径差异）
    t_df = t_df[["report_date", "f_ann_date"]].rename(columns={"f_ann_date": "f_ann_date_tushare"})
    a_df = a_df[["report_date", "f_ann_date"]].rename(columns={"f_ann_date": "f_ann_date_akshare"})
    merged = t_df.merge(a_df, on="report_date", how="inner").dropna()
    cutoff = pd.Timestamp("2018-01-01")
    merged = merged[merged["report_date"] >= cutoff]
    assert len(merged) >= 8, f"merge 后样本太少: {len(merged)}"

    diffs = (merged["f_ann_date_tushare"] - merged["f_ann_date_akshare"]).abs()
    bad = merged[diffs > pd.Timedelta(days=7)]
    assert bad.empty, f"Some f_ann_date diff > 7 days for post-2018 reports: {bad}"


# ============================================================================
# 10. PIT cutoff boundary 测试（关键报告期）
# ============================================================================


@pytest.mark.pit
@pytest.mark.slow
@_REQUIRES_TUSHARE
def test_pit_cutoff_boundary_2023_annual_report() -> None:
    """2023 年报通常 4 月中下旬披露；as_of=2024-04-01 应**看不到** 2023 年报."""
    from quantmind.data import TushareProvider

    p = TushareProvider()
    ticker = "600519.SH"  # 茅台
    period_2023_q4 = pd.Timestamp("2023-12-31")

    # Before cutoff: 2024-04-01 之前还没披露年报
    df_before = p.get_financials(ticker, "income", as_of=date(2024, 4, 1))
    if not df_before.empty:
        early = df_before[df_before["report_date"] == period_2023_q4]
        assert early.empty, (
            f"2023 年报在 2024-04-01 之前应未披露，但被返回了: {early}"
        )

    # After cutoff: 2024-05-01 之后通常已披露
    df_after = p.get_financials(ticker, "income", as_of=date(2024, 5, 1))
    matched = df_after[df_after["report_date"] == period_2023_q4]
    assert not matched.empty, "2024-05-01 应能看到 2023 年报"
    assert matched["f_ann_date"].iloc[0] <= pd.Timestamp("2024-05-01")
