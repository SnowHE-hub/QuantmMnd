"""探针脚本：测试 akshare 在本项目所需 8 类数据上的可用性.

输出每个接口的：
    - 是否可调通
    - 返回的列名
    - 数据规模与示例
    - 关键字段（如披露日）是否存在
    - 调用耗时

运行：
    python scripts/probe_akshare.py
"""

from __future__ import annotations

import time
import warnings
from typing import Any

import akshare as ak  # type: ignore[import-untyped]
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)


# ---------- 辅助打印 ----------

C = {
    "OK": "\033[92m",
    "WARN": "\033[93m",
    "FAIL": "\033[91m",
    "INFO": "\033[94m",
    "BOLD": "\033[1m",
    "DIM": "\033[2m",
    "END": "\033[0m",
}


def section(title: str) -> None:
    print(f"\n{C['BOLD']}{'═' * 72}{C['END']}")
    print(f"{C['BOLD']}  {title}{C['END']}")
    print(f"{C['BOLD']}{'═' * 72}{C['END']}")


def ok(msg: str) -> None:
    print(f"{C['OK']}[ OK ]{C['END']} {msg}")


def warn(msg: str) -> None:
    print(f"{C['WARN']}[WARN]{C['END']} {msg}")


def fail(msg: str) -> None:
    print(f"{C['FAIL']}[FAIL]{C['END']} {msg}")


def info(msg: str) -> None:
    print(f"{C['INFO']}[INFO]{C['END']} {msg}")


def show_df(df: pd.DataFrame, n: int = 3) -> None:
    print(f"  {C['DIM']}shape={df.shape}, columns={list(df.columns)[:15]}{C['END']}")
    if len(df) > 0:
        print(f"  {C['DIM']}sample (first {min(n, len(df))} rows):{C['END']}")
        with pd.option_context("display.max_colwidth", 30):
            print(df.head(n).to_string(max_cols=10, index=False))


# ---------- 单项探针 ----------

results: dict[str, dict[str, Any]] = {}


def probe(name: str, func_or_name: Any, **kwargs: Any) -> pd.DataFrame | None:
    """统一调用器：捕获异常、计时、记录.

    func_or_name 可以是 callable，也可以是字符串（自动从 ak module 找）.
    """
    if isinstance(func_or_name, str):
        if not hasattr(ak, func_or_name):
            warn(f"{name}: API '{func_or_name}' 在当前 akshare 版本中不存在")
            results[name] = {"status": "missing", "elapsed": 0.0, "api": func_or_name}
            return None
        func = getattr(ak, func_or_name)
    else:
        func = func_or_name

    print(
        f"\n  {C['BOLD']}→ {func.__name__}("
        f"{', '.join(f'{k}={v!r}' for k, v in kwargs.items())}){C['END']}"
    )
    t0 = time.monotonic()
    try:
        df = func(**kwargs)
        elapsed = time.monotonic() - t0
        if df is None or (hasattr(df, "empty") and df.empty):
            warn(f"返回为空 ({elapsed:.2f}s)")
            results[name] = {"status": "empty", "elapsed": elapsed, "api": func.__name__}
            return None
        ok(f"成功 ({elapsed:.2f}s)")
        show_df(df)
        results[name] = {
            "status": "ok",
            "elapsed": elapsed,
            "shape": df.shape,
            "columns": list(df.columns),
            "api": func.__name__,
        }
        return df
    except Exception as e:
        elapsed = time.monotonic() - t0
        fail(f"失败 ({elapsed:.2f}s): {type(e).__name__}: {str(e)[:150]}")
        results[name] = {
            "status": "fail",
            "elapsed": elapsed,
            "error": str(e)[:200],
            "api": func.__name__,
        }
        return None


# ============================================================================
# 测试用例
# ============================================================================

TICKER_NUMERIC = "300750"   # 宁德时代（不带后缀，akshare 习惯）
TICKER_WITH_PREFIX = "SZ300750"  # 部分接口要带交易所前缀
TICKER_MAOTAI = "600519"
START_DATE = "20240101"
END_DATE = "20240331"


def test_1_daily_price() -> None:
    section("1. 日线行情 (daily price)")
    info("用例：宁德时代 2024-01-01 ~ 2024-03-31 前复权")
    probe(
        "daily_price_qfq",
        ak.stock_zh_a_hist,
        symbol=TICKER_NUMERIC,
        period="daily",
        start_date=START_DATE,
        end_date=END_DATE,
        adjust="qfq",
    )
    info("用例：不复权（用于事件研究）")
    probe(
        "daily_price_raw",
        ak.stock_zh_a_hist,
        symbol=TICKER_NUMERIC,
        period="daily",
        start_date=START_DATE,
        end_date=END_DATE,
        adjust="",
    )


def test_2_valuation() -> None:
    section("2. 估值数据 (PE / PB)")
    info("用例：个股最新基础信息（含总市值/PE 快照，含行业）")
    probe("individual_info", "stock_individual_info_em", symbol=TICKER_NUMERIC)
    info("用例：个股历史 PE/PB（百度）")
    probe("valuation_history_baidu", "stock_zh_valuation_baidu", symbol=TICKER_NUMERIC,
          indicator="总市值", period="近一年")
    info("用例：A 股全市场 PB（每日截面）")
    probe("a_all_pb_today", "stock_a_all_pb")
    info("用例：大盘历史 PE/PB")
    probe("market_pe_lg", "stock_market_pe_lg", symbol="上证")
    probe("market_pb_lg", "stock_market_pb_lg", symbol="上证")


def test_3_financial_statements() -> None:
    section("3. 三大报表 (balance / income / cashflow)")
    info("用例：宁德时代年度资产负债表（东财）")
    probe("balance_sheet_yearly", "stock_balance_sheet_by_yearly_em", symbol=TICKER_WITH_PREFIX)
    info("用例：宁德时代年度利润表")
    probe("income_yearly", "stock_profit_sheet_by_yearly_em", symbol=TICKER_WITH_PREFIX)
    info("用例：宁德时代年度现金流量表")
    probe("cashflow_yearly", "stock_cash_flow_sheet_by_yearly_em", symbol=TICKER_WITH_PREFIX)
    info("用例：按报告期资产负债表（PIT 关键，含披露日）")
    probe("balance_sheet_by_report", "stock_balance_sheet_by_report_em", symbol=TICKER_WITH_PREFIX)
    info("用例：按报告期利润表")
    probe("profit_sheet_by_report", "stock_profit_sheet_by_report_em", symbol=TICKER_WITH_PREFIX)
    info("用例：按报告期现金流量表")
    probe("cashflow_by_report", "stock_cash_flow_sheet_by_report_em", symbol=TICKER_WITH_PREFIX)
    info("用例：财务摘要（同花顺）")
    probe("financial_abstract_ths", "stock_financial_abstract_ths", symbol=TICKER_NUMERIC,
          indicator="按报告期")
    info("用例：财务分析指标（含 ROE/ROA/毛利率等比率）")
    probe("financial_analysis_indicator", "stock_financial_analysis_indicator_em",
          symbol=TICKER_NUMERIC, indicator="按报告期")


def test_4_disclosure_date() -> None:
    section("4. 披露日 (PIT 关键)")
    info("用例：业绩快报（含披露日字段）")
    probe("yjbb_em", "stock_yjbb_em", date="20240331")
    info("用例：业绩预告")
    probe("yjyg_em", "stock_yjyg_em", date="20240331")
    info("用例：业绩预约披露时间表 (yysj)")
    probe("yysj_em", "stock_yysj_em", date="20240331")


def test_5_index_constituents() -> None:
    section("5. 历史成分股 (survivorship bias)")
    info("用例：沪深 300 当前成分股（中证指数官方）")
    probe("csi300_csindex", "index_stock_cons_csindex", symbol="000300")
    info("用例：沪深 300 当前权重")
    probe("csi300_weight_csindex", "index_stock_cons_weight_csindex", symbol="000300")
    info("用例：沪深 300 成分股（新浪，备选）")
    probe("csi300_sina", "index_stock_cons_sina", symbol="000300")
    info("用例：成分股通用接口")
    probe("index_stock_cons", "index_stock_cons", symbol="000300")
    info("用例：申万行业成分股")
    probe("sw_components", "index_component_sw", symbol="801011")


def test_6_north_bound() -> None:
    section("6. 北向资金 (A 股专属情绪因子)")
    info("用例：沪深港通历史 (hist_em)")
    probe("hsgt_hist", "stock_hsgt_hist_em", symbol="北向资金")
    info("用例：北向资金板块排名")
    probe("hsgt_board_rank", "stock_hsgt_board_rank_em", symbol="北向资金增持行业排行",
          indicator="今日")
    info("用例：北向资金个股持股")
    probe("hsgt_individual", "stock_hsgt_individual_em", stock=TICKER_NUMERIC)


def test_7_news() -> None:
    section("7. 个股新闻")
    info("用例：宁德时代相关新闻 (东财)")
    probe("news_em", "stock_news_em", symbol=TICKER_NUMERIC)


def test_8_research_reports() -> None:
    section("8. 券商研报")
    info("用例：宁德时代相关研报 (东财)")
    probe("research_report_em", "stock_research_report_em", symbol=TICKER_NUMERIC)


def test_9_extras() -> None:
    section("9. 加分项：停牌、融资融券、个股公告")
    info("用例：当日停牌列表")
    probe("stock_tfp", "stock_tfp_em", date="20240105")
    info("用例：融资融券（上交所）")
    probe("margin_sse", "stock_margin_sse", start_date="20240101", end_date="20240131")
    info("用例：上市公司股东减持")
    probe("hold_management", "stock_hold_management_person_em", symbol="减持")
    info("用例：个股最近公告（含披露日）")
    probe("individual_notice", "stock_individual_notice_report", symbol=TICKER_NUMERIC)


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    info(f"akshare 版本: {ak.__version__ if hasattr(ak, '__version__') else 'unknown'}")
    info(f"测试股票: 宁德时代 ({TICKER_NUMERIC} / {TICKER_WITH_PREFIX})")
    info(f"测试时段: {START_DATE} ~ {END_DATE}")

    test_1_daily_price()
    test_2_valuation()
    test_3_financial_statements()
    test_4_disclosure_date()
    test_5_index_constituents()
    test_6_north_bound()
    test_7_news()
    test_8_research_reports()
    test_9_extras()

    # ---------- 总结表 ----------
    section("Summary")
    by_status: dict[str, list[str]] = {"ok": [], "empty": [], "fail": []}
    for name, r in results.items():
        by_status.setdefault(r["status"], []).append(name)

    print(f"\n  {C['OK']}✓ 成功 ({len(by_status['ok'])} 项):{C['END']}")
    for n in by_status["ok"]:
        r = results[n]
        print(f"    - {n:35s} ({r['shape']}, {r['elapsed']:.2f}s)")
    if by_status["empty"]:
        print(f"\n  {C['WARN']}∅ 返回空 ({len(by_status['empty'])} 项):{C['END']}")
        for n in by_status["empty"]:
            print(f"    - {n}")
    if by_status["fail"]:
        print(f"\n  {C['FAIL']}✗ 失败 ({len(by_status['fail'])} 项):{C['END']}")
        for n in by_status["fail"]:
            r = results[n]
            print(f"    - {n}: {r['error']}")


if __name__ == "__main__":
    main()
