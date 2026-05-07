"""quantmind.features.fundamental — 基本面因子.

设计
====

每个因子函数签名统一为::

    def factor_name(snapshot: dict, as_of: date) -> pd.Series  # index=ticker

``snapshot`` 必须由 ``quantmind.data.load_snapshot(as_of)`` 加载，含：
    - ``daily_basic``      —— 估值因子主源（PE/PB/PS/市值）
    - ``financial_indicators`` —— 质量/成长因子主源（108 列预算指标）
    - ``financials_*``     —— 三大表，仅在需要原始字段计算复杂因子时用

经济含义参考
============

- Fama-French 三因子：``LogMarketCap`` (size), ``BM = 1/PB`` (value)
- Fama-French 五因子：补 ``ROE``（profitability）, ``Investment``
- Carhart：动量在 technical.py
- Asness Quality-Junk：``ROE_TTM``, ``Margins``, ``Stability``, ``Safety``
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from quantmind.features.utils import (
    latest_report_per_ticker,
    safe_divide,
    safe_log,
    ytd_to_ttm,
)

# ============================================================================
# 估值因子（valuation）—— 来自 daily_basic
# ============================================================================


def pe_ttm(snapshot: dict, as_of: date) -> pd.Series:  # noqa: ARG001
    """市盈率 TTM = 总市值 / TTM 净利润.

    Tushare daily_basic 已计算好；负 PE 设为 NaN（亏损公司用 PE 无意义）。
    """
    db = snapshot.get("daily_basic")
    if db is None or db.empty or "pe_ttm" not in db.columns:
        return pd.Series(dtype="float64")
    s = db.set_index("ticker")["pe_ttm"]
    return s.where(s > 0).rename("pe_ttm")


def pb(snapshot: dict, as_of: date) -> pd.Series:  # noqa: ARG001
    """市净率 = 总市值 / 净资产."""
    db = snapshot.get("daily_basic")
    if db is None or db.empty or "pb" not in db.columns:
        return pd.Series(dtype="float64")
    s = db.set_index("ticker")["pb"]
    return s.where(s > 0).rename("pb")


def ps_ttm(snapshot: dict, as_of: date) -> pd.Series:  # noqa: ARG001
    """市销率 TTM = 总市值 / TTM 营收."""
    db = snapshot.get("daily_basic")
    if db is None or db.empty or "ps_ttm" not in db.columns:
        return pd.Series(dtype="float64")
    s = db.set_index("ticker")["ps_ttm"]
    return s.where(s > 0).rename("ps_ttm")


def book_to_market(snapshot: dict, as_of: date) -> pd.Series:
    """账面市值比 BM = 1/PB（Fama-French value 因子）."""
    s = pb(snapshot, as_of)
    return safe_divide(pd.Series(1.0, index=s.index), s).rename("book_to_market")


def earnings_yield(snapshot: dict, as_of: date) -> pd.Series:
    """盈利收益率 EP = 1/PE_TTM."""
    s = pe_ttm(snapshot, as_of)
    return safe_divide(pd.Series(1.0, index=s.index), s).rename("earnings_yield")


def dividend_yield_ttm(snapshot: dict, as_of: date) -> pd.Series:  # noqa: ARG001
    """股息收益率 TTM."""
    db = snapshot.get("daily_basic")
    if db is None or db.empty or "dv_ttm" not in db.columns:
        return pd.Series(dtype="float64")
    return db.set_index("ticker")["dv_ttm"].rename("dividend_yield_ttm")


# ============================================================================
# 规模因子（size）
# ============================================================================


def log_market_cap(snapshot: dict, as_of: date) -> pd.Series:  # noqa: ARG001
    """对数总市值（万元）— Fama-French SMB 的核心."""
    db = snapshot.get("daily_basic")
    if db is None or db.empty or "total_mv" not in db.columns:
        return pd.Series(dtype="float64")
    return safe_log(db.set_index("ticker")["total_mv"]).rename("log_market_cap")


def log_circ_market_cap(snapshot: dict, as_of: date) -> pd.Series:  # noqa: ARG001
    """对数流通市值（万元）— A 股更常用，因有大股东限售."""
    db = snapshot.get("daily_basic")
    if db is None or db.empty or "circ_mv" not in db.columns:
        return pd.Series(dtype="float64")
    return safe_log(db.set_index("ticker")["circ_mv"]).rename("log_circ_market_cap")


# ============================================================================
# 质量因子（quality）—— 来自 fina_indicator
# ============================================================================


def roe_ttm(snapshot: dict, as_of: date) -> pd.Series:  # noqa: ARG001
    """净资产收益率 TTM（%）— Asness Quality 核心.

    fina_indicator.roe 已是 YTD 百分比；roe_yearly 是滚动年化值。
    """
    ind = snapshot.get("financial_indicators")
    if ind is None or ind.empty:
        return pd.Series(dtype="float64")
    latest = latest_report_per_ticker(ind, date_col="ann_date")
    col = "roe_yearly" if "roe_yearly" in latest.columns else "roe"
    return latest[col].rename("roe_ttm")


def roa_ttm(snapshot: dict, as_of: date) -> pd.Series:  # noqa: ARG001
    """总资产收益率 TTM（%）."""
    ind = snapshot.get("financial_indicators")
    if ind is None or ind.empty:
        return pd.Series(dtype="float64")
    latest = latest_report_per_ticker(ind, date_col="ann_date")
    col = "roa_yearly" if "roa_yearly" in latest.columns else "roa"
    return latest[col].rename("roa_ttm")


def gross_margin(snapshot: dict, as_of: date) -> pd.Series:  # noqa: ARG001
    """毛利率（%）."""
    ind = snapshot.get("financial_indicators")
    if ind is None or ind.empty:
        return pd.Series(dtype="float64")
    latest = latest_report_per_ticker(ind, date_col="ann_date")
    col = "grossprofit_margin" if "grossprofit_margin" in latest.columns else "gross_margin"
    return latest[col].rename("gross_margin")


def net_margin(snapshot: dict, as_of: date) -> pd.Series:  # noqa: ARG001
    """净利率（%）."""
    ind = snapshot.get("financial_indicators")
    if ind is None or ind.empty:
        return pd.Series(dtype="float64")
    latest = latest_report_per_ticker(ind, date_col="ann_date")
    return latest.get("netprofit_margin", pd.Series(dtype="float64")).rename("net_margin")


def debt_to_assets(snapshot: dict, as_of: date) -> pd.Series:  # noqa: ARG001
    """资产负债率（%）— 杠杆指标."""
    ind = snapshot.get("financial_indicators")
    if ind is None or ind.empty:
        return pd.Series(dtype="float64")
    latest = latest_report_per_ticker(ind, date_col="ann_date")
    return latest.get("debt_to_assets", pd.Series(dtype="float64")).rename("debt_to_assets")


def current_ratio(snapshot: dict, as_of: date) -> pd.Series:  # noqa: ARG001
    """流动比率."""
    ind = snapshot.get("financial_indicators")
    if ind is None or ind.empty:
        return pd.Series(dtype="float64")
    latest = latest_report_per_ticker(ind, date_col="ann_date")
    return latest.get("current_ratio", pd.Series(dtype="float64")).rename("current_ratio")


def asset_turnover(snapshot: dict, as_of: date) -> pd.Series:  # noqa: ARG001
    """总资产周转率 — 杜邦分解 component."""
    ind = snapshot.get("financial_indicators")
    if ind is None or ind.empty:
        return pd.Series(dtype="float64")
    latest = latest_report_per_ticker(ind, date_col="ann_date")
    return latest.get("assets_turn", pd.Series(dtype="float64")).rename("asset_turnover")


def equity_multiplier(snapshot: dict, as_of: date) -> pd.Series:  # noqa: ARG001
    """权益乘数 = 总资产/股东权益（杠杆维度）."""
    ind = snapshot.get("financial_indicators")
    if ind is None or ind.empty:
        return pd.Series(dtype="float64")
    latest = latest_report_per_ticker(ind, date_col="ann_date")
    return latest.get("assets_to_eqt", pd.Series(dtype="float64")).rename("equity_multiplier")


# ============================================================================
# 成长因子（growth）—— YoY 已由 fina_indicator 给出
# ============================================================================


def revenue_yoy(snapshot: dict, as_of: date) -> pd.Series:  # noqa: ARG001
    """总营收同比增速（%）."""
    ind = snapshot.get("financial_indicators")
    if ind is None or ind.empty:
        return pd.Series(dtype="float64")
    latest = latest_report_per_ticker(ind, date_col="ann_date")
    col = "tr_yoy" if "tr_yoy" in latest.columns else "or_yoy"
    return latest[col].rename("revenue_yoy")


def operating_profit_yoy(snapshot: dict, as_of: date) -> pd.Series:  # noqa: ARG001
    """营业利润同比（%）."""
    ind = snapshot.get("financial_indicators")
    if ind is None or ind.empty:
        return pd.Series(dtype="float64")
    latest = latest_report_per_ticker(ind, date_col="ann_date")
    return latest.get("op_yoy", pd.Series(dtype="float64")).rename("operating_profit_yoy")


def net_profit_yoy(snapshot: dict, as_of: date) -> pd.Series:  # noqa: ARG001
    """净利润同比（%）."""
    ind = snapshot.get("financial_indicators")
    if ind is None or ind.empty:
        return pd.Series(dtype="float64")
    latest = latest_report_per_ticker(ind, date_col="ann_date")
    col = "netprofit_yoy" if "netprofit_yoy" in latest.columns else "dt_netprofit_yoy"
    return latest[col].rename("net_profit_yoy")


def quarterly_revenue_yoy(snapshot: dict, as_of: date) -> pd.Series:  # noqa: ARG001
    """单季度营收同比（%）— 比累计 YoY 更敏感."""
    ind = snapshot.get("financial_indicators")
    if ind is None or ind.empty:
        return pd.Series(dtype="float64")
    latest = latest_report_per_ticker(ind, date_col="ann_date")
    return latest.get("q_sales_yoy", pd.Series(dtype="float64")).rename("quarterly_revenue_yoy")


# ============================================================================
# 现金流质量
# ============================================================================


def accruals(snapshot: dict, as_of: date) -> pd.Series:
    """应计利润 = (净利润 - 经营现金流) / 总资产.

    高 accruals = 利润含金量低（盈余管理嫌疑），是经典的反向因子。
    """
    inc = snapshot.get("financials_income")
    cf = snapshot.get("financials_cashflow")
    bs = snapshot.get("financials_balance_sheet")
    if any(x is None or x.empty for x in (inc, cf, bs)):
        return pd.Series(dtype="float64")

    inc_l = latest_report_per_ticker(inc)
    cf_l = latest_report_per_ticker(cf)
    bs_l = latest_report_per_ticker(bs)

    # net income (n_income or n_income_attr_p)
    ni = inc_l.get("n_income_attr_p", inc_l.get("n_income", pd.Series(dtype="float64")))
    ocf = cf_l.get("n_cashflow_act", pd.Series(dtype="float64"))
    ta = bs_l.get("total_assets", pd.Series(dtype="float64"))

    common = ni.index.intersection(ocf.index).intersection(ta.index)
    out = safe_divide(ni.loc[common] - ocf.loc[common], ta.loc[common])
    return out.rename("accruals")


def ocf_to_revenue_ttm(snapshot: dict, as_of: date) -> pd.Series:
    """经营现金流 / 营收 TTM —— 现金流质量."""
    cf = snapshot.get("financials_cashflow")
    inc = snapshot.get("financials_income")
    if cf is None or cf.empty or inc is None or inc.empty:
        return pd.Series(dtype="float64")
    ocf_ttm = ytd_to_ttm(cf, "n_cashflow_act", as_of=as_of)
    rev_ttm = ytd_to_ttm(inc, "total_revenue", as_of=as_of)
    common = ocf_ttm.index.intersection(rev_ttm.index)
    return safe_divide(ocf_ttm.loc[common], rev_ttm.loc[common]).rename("ocf_to_revenue_ttm")


# ============================================================================
# 公开 API：批量计算所有基本面因子
# ============================================================================


# (factor_name, function) — 顺序即输出列顺序
FUNDAMENTAL_FACTORS = [
    # 估值 (5)
    ("pe_ttm", pe_ttm),
    ("pb", pb),
    ("ps_ttm", ps_ttm),
    ("book_to_market", book_to_market),
    ("earnings_yield", earnings_yield),
    ("dividend_yield_ttm", dividend_yield_ttm),
    # 规模 (2)
    ("log_market_cap", log_market_cap),
    ("log_circ_market_cap", log_circ_market_cap),
    # 质量 (8)
    ("roe_ttm", roe_ttm),
    ("roa_ttm", roa_ttm),
    ("gross_margin", gross_margin),
    ("net_margin", net_margin),
    ("debt_to_assets", debt_to_assets),
    ("current_ratio", current_ratio),
    ("asset_turnover", asset_turnover),
    ("equity_multiplier", equity_multiplier),
    # 成长 (4)
    ("revenue_yoy", revenue_yoy),
    ("operating_profit_yoy", operating_profit_yoy),
    ("net_profit_yoy", net_profit_yoy),
    ("quarterly_revenue_yoy", quarterly_revenue_yoy),
    # 现金流质量 (2)
    ("accruals", accruals),
    ("ocf_to_revenue_ttm", ocf_to_revenue_ttm),
]


def compute_all_fundamental_factors(
    snapshot: dict, as_of: date
) -> pd.DataFrame:
    """计算所有基本面因子，返回 DataFrame(index=ticker, columns=factor_name)."""
    out = {}
    for name, fn in FUNDAMENTAL_FACTORS:
        try:
            s = fn(snapshot, as_of)
        except Exception:  # noqa: BLE001
            s = pd.Series(dtype="float64", name=name)
        out[name] = s
    df = pd.DataFrame(out)
    df.index.name = "ticker"
    return df


__all__ = [
    "FUNDAMENTAL_FACTORS",
    "accruals",
    "asset_turnover",
    "book_to_market",
    "compute_all_fundamental_factors",
    "current_ratio",
    "debt_to_assets",
    "dividend_yield_ttm",
    "earnings_yield",
    "equity_multiplier",
    "gross_margin",
    "log_circ_market_cap",
    "log_market_cap",
    "net_margin",
    "net_profit_yoy",
    "ocf_to_revenue_ttm",
    "operating_profit_yoy",
    "pb",
    "pe_ttm",
    "ps_ttm",
    "quarterly_revenue_yoy",
    "revenue_yoy",
    "roa_ttm",
    "roe_ttm",
]
