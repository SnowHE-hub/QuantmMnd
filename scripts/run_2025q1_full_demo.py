"""
2025 Q1 全A股漏斗演示脚本
=============================
演示日期：2025-01-02（2025年开年第一个交易日）
起始宇宙：全A股 ~5517 只
数据来源：Token B（tsy.xiaodefa.cn 代理）批量拉取
目标产出：Top-15 推荐股票 + 6-Agent 分析 + 历史回测验证

漏斗层次：
  Layer 1：基础质量（非ST、上市满6个月、市值≥20亿）
  Layer 2：流动性（日均换手率≥0.5%、流通市值≥10亿）
  Layer 3：技术趋势（45日价格历史 → 动量、RSI、布林带）
  Layer 4：因子评分（Alpha 1374 用完整71因子, 其余用技术因子）
  Layer 5：LGBM 排名 → Top-50
  Layer 6：HRP 仓位优化 → Top-15
  System 2：6-Agent 投资分析
  System 3：历史回测验证
"""

import sys
import os
import time
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# ── 常量 ─────────────────────────────────────────────────────────────────────
TOKEN_B = "5caf9b3022e13d4e915df0af19a076130287cb7837c0b020290691c8"
PROXY_URL = "http://tsy.xiaodefa.cn"
OFFICIAL_URL = "https://api.tushare.pro"

# 注：Token B 代理不稳定，统一改用 Token A 官方端点拉市场数据
# Token B 保留作为备用标识（演示中会体现双Token策略）
DEMO_DATE = pd.Timestamp("2025-01-02")
DEMO_DATE_STR = "20250102"

PRICE_PANEL = ROOT / "data/raw/alpha_prices_panel.parquet"
FACTOR_PANEL = ROOT / "data/panel/alpha_panel_v3.parquet"
MODEL_PATH = ROOT / "models/lgbm_v5_alpha_63d.pkl"
MODELS_DIR = ROOT / "models"
OUT_DIR = ROOT / "reports/demo/2025-01-02-fullA"

# 历史价格起止（用于技术指标，需要 ~60 交易日）
HIST_START_STR = "20241001"

LAYER1_MIN_MV = 20_0000   # 万元，即 20亿人民币
LAYER2_MIN_TURNOVER = 0.3  # 日换手率 %
LAYER2_MIN_CIRC_MV = 10_0000  # 万元，即 10亿
TOP_N_LGBM = 50
TOP_N_FINAL = 15
ALPHA_AS_OF = pd.Timestamp("2024-12-31")

# ── 计时工具 ────────────────────────────────────────────────────────────────

TIMINGS = {}

class Timer:
    def __init__(self, name: str):
        self.name = name
    def __enter__(self):
        self._t0 = time.time()
        print(f"\n{'='*60}")
        print(f"[{self.name}] 开始...")
        return self
    def __exit__(self, *_):
        elapsed = time.time() - self._t0
        TIMINGS[self.name] = elapsed
        print(f"[{self.name}] 完成 ✓  耗时 {elapsed:.1f}s")


# ── API 实例 ────────────────────────────────────────────────────────────────

def get_api_a():
    """
    Token A on official Tushare — 全市场参考数据 + 价格快照
    (Token B 代理 tsy.xiaodefa.cn 不稳定，已降级为备用)
    """
    import tushare as ts
    token_a = os.environ.get("TUSHARE_TOKEN", "")
    ts.set_token(token_a)
    pro = ts.pro_api()
    return pro


def rate_limit_sleep(n_calls_done: int, calls_per_min: int = 150):
    """保守限速：150次/min ≈ 0.4s/call"""
    if n_calls_done % 10 == 0 and n_calls_done > 0:
        time.sleep(0.4)


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 0: 获取全A股基础数据
# ══════════════════════════════════════════════════════════════════════════════

def fetch_daily_basic_snapshot(api) -> pd.DataFrame:
    """daily_basic(2025-01-02): 单日全市场估值/流动性快照"""
    df = api.daily_basic(
        trade_date=DEMO_DATE_STR,
        fields="ts_code,close,pe,pe_ttm,pb,ps_ttm,total_mv,circ_mv,turnover_rate,volume_ratio,free_share"
    )
    # total_mv / circ_mv 单位是万元
    return df


def fetch_daily_snapshot(api) -> pd.DataFrame:
    """daily(2025-01-02): 单日全市场价格"""
    df = api.daily(
        trade_date=DEMO_DATE_STR,
        fields="ts_code,open,high,low,close,pre_close,pct_chg,vol,amount"
    )
    return df


def fetch_stock_universe(api) -> pd.DataFrame:
    """stock_basic: 全A股列表 → ~5500 只（Token A 官方端点）"""
    df = api.stock_basic(
        exchange="", list_status="L",
        fields="ts_code,name,industry,list_date,market"
    )
    df["list_date"] = pd.to_datetime(df["list_date"], format="%Y%m%d", errors="coerce")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 1-2: 基础质量 + 流动性过滤
# ══════════════════════════════════════════════════════════════════════════════

def apply_layer1(stock_basic: pd.DataFrame, daily_basic: pd.DataFrame) -> pd.DataFrame:
    """Layer 1: 非ST + 上市满6个月 + 市值≥20亿"""
    # 合并
    df = stock_basic.merge(daily_basic, on="ts_code", how="inner")

    n0 = len(df)
    # 非 ST
    df = df[~df["name"].str.contains("ST|退", na=False)]
    n1 = len(df)

    # 上市满6个月
    cutoff = DEMO_DATE - pd.DateOffset(months=6)
    df = df[df["list_date"] <= cutoff]
    n2 = len(df)

    # 市值 ≥ 20亿（total_mv 单位万元）
    df = df[df["total_mv"] >= LAYER1_MIN_MV]
    n3 = len(df)

    # 市场类型：主板/创业板/科创板（排除北交所）
    df = df[~df["market"].isin(["北交所", "BJ"])]
    n4 = len(df)

    print(f"  Layer 1 全A股 → 非ST → 满6月 → 市值≥20亿 → 非北交所")
    print(f"  {n0} → {n1} → {n2} → {n3} → {n4}")
    return df


def apply_layer2(df: pd.DataFrame) -> pd.DataFrame:
    """Layer 2: 流动性过滤（换手率 + 流通市值）"""
    n0 = len(df)
    # 换手率 ≥ 0.3%
    df = df[df["turnover_rate"] >= LAYER2_MIN_TURNOVER]
    n1 = len(df)
    # 流通市值 ≥ 10亿
    df = df[df["circ_mv"] >= LAYER2_MIN_CIRC_MV]
    n2 = len(df)
    # 排除成交量异常（涨跌停无法交易）
    df = df[df["volume_ratio"].between(0.2, 8.0)]
    n3 = len(df)

    print(f"  Layer 2 换手率≥0.3% → 流通市值≥10亿 → 排除涨跌停")
    print(f"  {n0} → {n1} → {n2} → {n3}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 3: 技术趋势（需要历史价格）
# ══════════════════════════════════════════════════════════════════════════════

def fetch_price_history(api_b, tickers: list[str]) -> pd.DataFrame:
    """
    策略：按日期批量拉全市场日线（45个交易日），
    Token B 代理只支持 daily(trade_date=...) 的 all-stock 模式，
    每次拉一天 ~5000 条，45天 ≈ 3分钟
    """
    # 获取交易日历（Token B 代理支持 trade_cal）
    trade_cal = api_b.trade_cal(
        exchange="SSE",
        start_date=HIST_START_STR,
        end_date=DEMO_DATE_STR,
        fields="cal_date,is_open"
    )
    trade_dates = (
        trade_cal[trade_cal["is_open"] == 1]
        .sort_values("cal_date")["cal_date"]
        .tolist()
    )
    # 取最近 45 个交易日
    trade_dates = trade_dates[-45:]
    print(f"  拉取 {len(trade_dates)} 个交易日历史行情 [{trade_dates[0]} ~ {trade_dates[-1]}]")

    ticker_set = set(tickers)
    frames = []
    for i, d in enumerate(trade_dates):
        try:
            df = api_b.daily(
                trade_date=d,
                fields="ts_code,trade_date,open,high,low,close,pre_close,vol,amount"
            )
            # 只保留目标股票
            df = df[df["ts_code"].isin(ticker_set)]
            frames.append(df)
        except Exception as e:
            print(f"    ⚠ {d} 失败: {e}")
        # 限速：~150 req/min
        time.sleep(0.4)
        if (i + 1) % 15 == 0:
            print(f"    ... 已拉取 {i+1}/{len(trade_dates)} 天")

    if not frames:
        return pd.DataFrame()
    price_df = pd.concat(frames, ignore_index=True)
    price_df["trade_date"] = pd.to_datetime(price_df["trade_date"], format="%Y%m%d", errors="coerce")
    price_df = price_df.sort_values(["ts_code", "trade_date"])
    print(f"  历史行情: {len(price_df)} 条, {price_df['ts_code'].nunique()} 只")
    return price_df


def compute_technical_features(price_df: pd.DataFrame) -> pd.DataFrame:
    """计算技术指标特征"""
    results = []
    grouped = price_df.groupby("ts_code")

    for ts_code, g in grouped:
        g = g.sort_values("trade_date").reset_index(drop=True)
        close = g["close"].values
        vol = g["vol"].values
        n = len(close)

        if n < 10:
            continue

        # 动量
        mom_1m = (close[-1] / close[max(0, n - 21)] - 1) if n >= 21 else np.nan
        mom_3m = (close[-1] / close[max(0, n - 45)] - 1) if n >= 45 else np.nan

        # 波动率
        rets = pd.Series(close).pct_change().dropna().values
        vol_3m = float(np.std(rets) * np.sqrt(252)) if len(rets) >= 10 else np.nan
        down_rets = rets[rets < 0]
        downside_vol = float(np.std(down_rets) * np.sqrt(252)) if len(down_rets) >= 5 else np.nan

        # RSI (14)
        rsi_14 = np.nan
        if len(rets) >= 14:
            gains = np.where(rets[-14:] > 0, rets[-14:], 0)
            losses = np.where(rets[-14:] < 0, -rets[-14:], 0)
            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)
            rs = avg_gain / avg_loss if avg_loss > 0 else 100
            rsi_14 = 100 - 100 / (1 + rs)

        # 布林带位置
        bollinger = np.nan
        if n >= 20:
            ma20 = np.mean(close[-20:])
            std20 = np.std(close[-20:])
            bollinger = (close[-1] - ma20) / (2 * std20 + 1e-9)

        # 换手率加速度（粗代理：近5日/近30日 vol 之比）
        volume_spike = np.nan
        if n >= 30:
            v5 = np.mean(vol[-5:])
            v30 = np.mean(vol[-30:])
            volume_spike = v5 / (v30 + 1e-9)

        # 距52周高点
        dist_52w_high = (close[-1] / np.max(close) - 1) if n > 0 else np.nan

        results.append({
            "ts_code": ts_code,
            "momentum_1m": mom_1m,
            "momentum_3m": mom_3m,
            "volatility_3m": vol_3m,
            "downside_volatility_3m": downside_vol,
            "rsi_14": rsi_14,
            "bollinger_position": bollinger,
            "volume_spike_5_30": volume_spike,
            "dist_52w_high": dist_52w_high,
        })

    tech_df = pd.DataFrame(results)
    return tech_df


def apply_layer3(candidates: pd.DataFrame, tech_df: pd.DataFrame) -> pd.DataFrame:
    """Layer 3: 技术趋势过滤"""
    df = candidates.merge(tech_df, on="ts_code", how="left")

    n0 = len(df)
    # 动量过滤：近1月非极端下跌（> -15%）
    df = df[df["momentum_1m"].isna() | (df["momentum_1m"] > -0.15)]
    n1 = len(df)

    # RSI 不超买（< 80）
    df = df[df["rsi_14"].isna() | (df["rsi_14"] < 80)]
    n2 = len(df)

    # 布林带位置非极端（< 1.5，即价格不高于上轨 1.5 倍）
    df = df[df["bollinger_position"].isna() | (df["bollinger_position"] < 1.5)]
    n3 = len(df)

    print(f"  Layer 3 动量≥-15% → RSI<80 → 布林带<1.5")
    print(f"  {n0} → {n1} → {n2} → {n3}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 4: 因子评分（Alpha 1374 完整因子 + 其余技术因子）
# ══════════════════════════════════════════════════════════════════════════════

def load_alpha_factors() -> pd.DataFrame:
    """加载 Alpha 1374 的 2024-12-31 因子面板"""
    df = pd.read_parquet(FACTOR_PANEL)
    df = df.reset_index()
    # 取最新一期
    df = df[df["as_of"] == ALPHA_AS_OF].copy()
    df = df.rename(columns={"ticker": "ts_code"})
    print(f"  Alpha 1374 因子面板: {len(df)} 行, {df['ts_code'].nunique()} 只 @ {ALPHA_AS_OF.date()}")
    return df


def build_feature_matrix(candidates: pd.DataFrame, alpha_factors: pd.DataFrame, tech_df: pd.DataFrame) -> pd.DataFrame:
    """
    构建用于 LGBM 评分的特征矩阵（37个因子，与 lgbm_v5_alpha_63d.pkl 训练时一致）：
    - Alpha 1374 中的股票：优先用完整因子面板
    - 其余股票：用 tech_df + daily_basic 中的 PE/PB/市值等，缺失值 NaN（LGBM 可处理）
    """
    # 完整 37 特征（与模型训练时一致）
    MODEL_FEATURES = [
        "turnover_3m_avg", "volatility_3m", "margin_buy_intensity", "turnover_rate_quantile",
        "volatility_1y", "downside_volatility_3m", "amplitude_quantile", "volume_spike_5_30",
        "book_to_market", "dividend_yield_ttm", "momentum_1m", "accruals", "earnings_yield",
        "bollinger_position", "rsi_14", "relative_strength_vs_csi300_60d", "momentum_3m",
        "pb", "reversal_1w", "max_drawdown_3m", "free_float_ratio", "margin_buy_amount_20d",
        "ocf_to_revenue_ttm", "pe_ttm", "north_hold_amount", "margin_balance_change_20d",
        "relative_strength_vs_csi300_120d", "momentum_6m", "north_hold_amount_change_20d",
        "margin_short_ratio", "list_age_years", "beta_60d", "north_hold_ratio",
        "ps_ttm", "log_market_cap", "beta_252d", "north_hold_ratio_change_60d"
    ]

    alpha_codes = set(alpha_factors["ts_code"].tolist())
    in_alpha = candidates[candidates["ts_code"].isin(alpha_codes)].copy()
    not_in_alpha = candidates[~candidates["ts_code"].isin(alpha_codes)].copy()

    # ── Alpha 1374 股票：merge 完整因子面板 ──
    alpha_factor_cols = ["ts_code"] + [c for c in MODEL_FEATURES if c in alpha_factors.columns]
    alpha_feat = in_alpha.merge(alpha_factors[alpha_factor_cols], on="ts_code", how="left")

    # 用 tech_df 的实时值覆盖（更贴近 2025-01-02 市场状态）
    tech_update_cols = ["ts_code", "momentum_1m", "momentum_3m", "volatility_3m",
                        "downside_volatility_3m", "rsi_14", "bollinger_position",
                        "volume_spike_5_30", "dist_52w_high"]
    tech_avail = [c for c in tech_update_cols if c in tech_df.columns]
    drop_cols = [c for c in tech_avail if c in alpha_feat.columns and c != "ts_code"]
    alpha_feat = alpha_feat.drop(columns=drop_cols, errors="ignore")
    alpha_feat = alpha_feat.merge(tech_df[tech_avail], on="ts_code", how="left")

    # ── 非 Alpha 股票：技术因子 + daily_basic 估值 ──
    not_alpha_feat = not_in_alpha.merge(tech_df, on="ts_code", how="left")

    # 从 candidates（含 daily_basic 字段）补充估值
    daily_basic_cols = ["ts_code", "pe_ttm", "pb", "ps_ttm", "total_mv", "circ_mv",
                        "turnover_rate", "free_share"]
    daily_avail = [c for c in daily_basic_cols if c in candidates.columns]
    not_alpha_feat = not_in_alpha.merge(tech_df, on="ts_code", how="left")
    # pe/pb/ps 来自 daily_basic（已在 candidates 里）
    for col in ["pe_ttm", "pb", "ps_ttm", "total_mv", "circ_mv", "turnover_rate", "free_share"]:
        if col in not_in_alpha.columns and col not in not_alpha_feat.columns:
            not_alpha_feat[col] = not_in_alpha[col].values

    # 派生特征
    not_alpha_feat["turnover_3m_avg"] = not_alpha_feat.get("turnover_rate", np.nan)
    if "total_mv" in not_alpha_feat.columns:
        not_alpha_feat["log_market_cap"] = np.log(not_alpha_feat["total_mv"].clip(lower=1) * 1e4)  # 万→元→log
    if "list_date" in not_in_alpha.columns:
        not_alpha_feat["list_age_years"] = (DEMO_DATE - not_in_alpha["list_date"]).dt.days / 365.25

    # 合并两部分
    all_feat = pd.concat([alpha_feat, not_alpha_feat], ignore_index=True)

    # 确保所有 MODEL_FEATURES 列存在（NaN 表示缺失，LGBM 可处理）
    for col in MODEL_FEATURES:
        if col not in all_feat.columns:
            all_feat[col] = np.nan

    n_alpha_in = len(alpha_feat)
    n_not = len(not_alpha_feat)
    print(f"  特征矩阵: {len(all_feat)} 只股票")
    print(f"    Alpha 1374 中: {n_alpha_in} 只（完整37因子，最新实时技术指标）")
    print(f"    非 Alpha: {n_not} 只（PE/PB + 技术因子，其余NaN让LGBM处理）")
    return all_feat, MODEL_FEATURES


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 5: LGBM 排名
# ══════════════════════════════════════════════════════════════════════════════

def lgbm_rank(feat_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """用 lgbm_v5_alpha_63d 对候选股打分排名"""
    import pickle

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    X = feat_df[feature_cols].copy()
    # 缺失值用中位数填充
    for col in feature_cols:
        X[col] = X[col].fillna(X[col].median())

    scores = model.predict(X)
    feat_df = feat_df.copy()
    feat_df["lgbm_score"] = scores
    feat_df = feat_df.sort_values("lgbm_score", ascending=False).reset_index(drop=True)
    feat_df["rank"] = range(1, len(feat_df) + 1)
    return feat_df


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 6: HRP 仓位优化 → Top-N
# ══════════════════════════════════════════════════════════════════════════════

def hrp_select(top50_codes: list[str], price_panel: pd.DataFrame, top_n: int = TOP_N_FINAL) -> list[str]:
    """
    用 HRP（层次风险平价）从 Top-50 中选出 Top-15。
    利用已有的 alpha_prices_panel 历史数据。
    """
    try:
        from quantmind.portfolio.position_sizing import hrp_weights
        # 取过去 1 年的日收益率
        panel = price_panel[price_panel["ts_code"].isin(top50_codes)].copy()
        panel = panel[panel["trade_date"] <= DEMO_DATE]
        panel = panel[panel["trade_date"] >= DEMO_DATE - pd.DateOffset(years=1)]
        # 价格列
        price_col = "adj_close" if "adj_close" in panel.columns else "close"
        wide = (
            panel.pivot(index="trade_date", columns="ts_code", values=price_col)
            .sort_index()
        )
        returns = wide.pct_change().dropna(how="all")
        # 只保留有足够数据的股票
        valid = returns.columns[returns.count() >= 60].tolist()
        if len(valid) < top_n:
            # 不足则回退到 Top-N by LGBM score
            return top50_codes[:top_n]

        returns = returns[valid]
        weights = hrp_weights(returns, max_weight=0.15)
        # 取权重最大的 top_n 只
        top_selected = weights.nlargest(top_n).index.tolist()
        print(f"  HRP 从 {len(valid)} 只 → 选出 Top-{top_n}")
        return top_selected
    except Exception as e:
        print(f"  ⚠ HRP 失败（{e}），回退到 Top-{top_n} by LGBM score")
        return top50_codes[:top_n]


# ══════════════════════════════════════════════════════════════════════════════
# System 2: 6-Agent 分析
# ══════════════════════════════════════════════════════════════════════════════

def run_agent_analysis(tickers: list[str]) -> dict:
    """对 Top-15 股票运行 6-Agent 分析（无 LLM 调用，使用规则回退）"""
    try:
        from scripts.run_investment_pipeline import _load_price_df, run_six_agents
        from quantmind.agents.investment_agents.strategy_agent import StrategyAgent
    except ImportError as e:
        print(f"  ⚠ 导入失败: {e}，跳过 Agent 分析")
        return {}

    # 加载价格宽表（供 Agent 内部使用）
    try:
        price_df = _load_price_df()
    except Exception as e:
        print(f"  ⚠ 价格数据加载失败: {e}")
        price_df = None

    results = {}
    as_of_str = str(DEMO_DATE.date())

    context = {
        "price_df": price_df,
        "as_of": as_of_str,
        "market_regime": "bull_low_vol",
        "index_pe": 15.2,
        "risk_free_rate": 0.018,
    }

    # 从环境变量检测可用 provider
    import os as _os
    if _os.environ.get("DASHSCOPE_API_KEY", "").strip():
        llm_provider = "dashscope"
        llm_model = "qwen-plus"
    elif _os.environ.get("DEEPSEEK_API_KEY", "").strip():
        llm_provider = "deepseek"
        llm_model = "deepseek-chat"
    else:
        llm_provider = "none"
        llm_model = "qwen-plus"
    print(f"  LLM provider: {llm_provider} / {llm_model}")

    for i, ticker in enumerate(tickers):
        try:
            # 5-Agent 信号
            agent_signals = run_six_agents(
                ticker=ticker,
                as_of=as_of_str,
                context=context,
                provider=llm_provider,
                model=llm_model,
            )

            # StrategyAgent 综合评级（LLM 生成研报式分析）
            strategy_agent = StrategyAgent(
                ticker=ticker,
                as_of=as_of_str,
                context=context,
                agent_signals=agent_signals,
                provider=llm_provider,
                model=llm_model,
            )
            strategy = strategy_agent.analyze_with_llm()

            results[ticker] = strategy
            rating = getattr(strategy, "rating", "—") or "—"
            conf = getattr(strategy, "confidence_score", 0) or 0
            print(f"    [{i+1:2d}/{len(tickers)}] {ticker}: {rating} | 信心={float(conf):.2f}")
        except Exception as e:
            print(f"    [{i+1:2d}/{len(tickers)}] {ticker}: ⚠ {str(e)[:80]}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# System 3: 历史回测验证
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest_validation(agent_results: dict) -> dict:
    """对 Top-15 的 Agent 策略进行历史回测验证"""
    try:
        from scripts.validate_strategies import batch_validate
        from scripts.run_investment_pipeline import _load_price_df
    except ImportError as e:
        print(f"  ⚠ 导入失败: {e}，跳过回测验证")
        return {}

    # 加载价格宽表
    try:
        price_df = _load_price_df()
    except Exception as e:
        print(f"  ⚠ 价格数据加载失败: {e}")
        return {}

    strat_dicts = []
    for ticker, strat in agent_results.items():
        try:
            if hasattr(strat, "model_dump"):
                d = strat.model_dump()
            elif hasattr(strat, "__dict__"):
                d = strat.__dict__.copy()
            else:
                d = {}
            d["ticker"] = ticker
            d["as_of"] = str(DEMO_DATE.date())
            strat_dicts.append(d)
        except Exception:
            pass

    if not strat_dicts:
        return {}

    try:
        validation_results = batch_validate(strat_dicts, price_df=price_df, panel_df=None)
        print(f"  回测验证完成: {len(validation_results)} 条")
        return {r.ticker if hasattr(r, "ticker") else str(i): r
                for i, r in enumerate(validation_results)}
    except Exception as e:
        print(f"  ⚠ batch_validate 失败: {e}")
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# 生成 HTML 报告
# ══════════════════════════════════════════════════════════════════════════════

def build_html_report(
    funnel_stats: dict,
    ranked_df: pd.DataFrame,
    top15_codes: list[str],
    agent_results: dict,
    validation_results: dict,
    stock_info: pd.DataFrame,
) -> str:
    """生成完整的 HTML 投资报告"""

    # 股票信息索引
    info_idx = stock_info.set_index("ts_code") if "ts_code" in stock_info.columns else pd.DataFrame()

    def get_info(code, field, default="—"):
        try:
            return info_idx.loc[code, field]
        except Exception:
            return default

    def fmt_pct(v, digits=1):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        return f"{v*100:.{digits}f}%"

    def fmt_num(v, digits=2):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        return f"{v:.{digits}f}"

    # 漏斗统计
    funnel_rows = ""
    for step, (n_in, n_out, desc) in funnel_stats.items():
        pct = n_out / n_in * 100 if n_in > 0 else 0
        funnel_rows += f"""
        <tr>
          <td>{step}</td>
          <td>{desc}</td>
          <td class="num">{n_in:,}</td>
          <td class="num">{n_out:,}</td>
          <td class="num">{pct:.1f}%</td>
        </tr>"""

    # Top-50 LGBM 排名表
    top50_df = ranked_df[ranked_df["ts_code"].isin(
        ranked_df.head(TOP_N_LGBM)["ts_code"].tolist()
    )].head(TOP_N_LGBM)

    top50_rows = ""
    for _, row in top50_df.iterrows():
        is_top15 = "★ " if row["ts_code"] in top15_codes else ""
        name = get_info(row["ts_code"], "name", row.get("name", row["ts_code"]))
        industry = get_info(row["ts_code"], "industry", "—")
        pb_val = fmt_num(row.get("pb", np.nan))
        mom1m = fmt_pct(row.get("momentum_1m", np.nan))
        rsi = fmt_num(row.get("rsi_14", np.nan), 1)
        highlight = 'class="top15"' if row["ts_code"] in top15_codes else ""
        top50_rows += f"""
        <tr {highlight}>
          <td>{is_top15}{int(row['rank'])}</td>
          <td>{row['ts_code']}</td>
          <td>{name}</td>
          <td>{industry}</td>
          <td class="num">{fmt_num(row.get('lgbm_score', np.nan), 4)}</td>
          <td class="num">{pb_val}</td>
          <td class="num">{mom1m}</td>
          <td class="num">{rsi}</td>
        </tr>"""

    # Agent 分析表
    agent_rows = ""
    for rank_i, code in enumerate(top15_codes, 1):
        name = get_info(code, "name", code)
        industry = get_info(code, "industry", "—")
        strat = agent_results.get(code)
        val = validation_results.get(code)

        if strat:
            rating = getattr(strat, "rating", "—")
            conf = fmt_num(getattr(strat, "confidence_score", np.nan))
            action = getattr(strat, "action", "—")
            val_pnl = fmt_pct(getattr(val, "total_return", np.nan)) if val else "—"
            val_sharpe = fmt_num(getattr(val, "sharpe_ratio", np.nan)) if val else "—"
            val_status = getattr(val, "validation_status", "—") if val else "—"
        else:
            rating = conf = action = val_pnl = val_sharpe = val_status = "—"

        rating_cls = {"强烈买入": "buy-strong", "买入": "buy", "中性": "neutral",
                      "卖出": "sell", "强烈卖出": "sell-strong"}.get(rating, "")
        agent_rows += f"""
        <tr>
          <td><strong>{rank_i}</strong></td>
          <td>{code}</td>
          <td>{name}</td>
          <td>{industry}</td>
          <td><span class="badge {rating_cls}">{rating}</span></td>
          <td class="num">{conf}</td>
          <td>{action}</td>
          <td class="num">{val_pnl}</td>
          <td class="num">{val_sharpe}</td>
          <td><span class="badge {rating_cls}">{val_status}</span></td>
        </tr>"""

    # 耗时统计
    timing_rows = ""
    total_t = sum(TIMINGS.values())
    for step_name, elapsed in TIMINGS.items():
        timing_rows += f"<tr><td>{step_name}</td><td class='num'>{elapsed:.1f}s</td></tr>"
    timing_rows += f"<tr style='font-weight:bold'><td>总计</td><td class='num'>{total_t:.1f}s</td></tr>"

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>QuantMind 全A股漏斗演示 — 2025年开年第一个交易日</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: -apple-system, 'PingFang SC', 'Microsoft YaHei', sans-serif;
            background: #0f1117; color: #e2e8f0; padding: 20px; }}
    h1 {{ font-size: 1.8em; color: #7c3aed; margin-bottom: 4px; }}
    h2 {{ font-size: 1.2em; color: #a78bfa; margin: 24px 0 8px;
          border-left: 3px solid #7c3aed; padding-left: 10px; }}
    .subtitle {{ color: #94a3b8; font-size: 0.9em; margin-bottom: 24px; }}
    .meta-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 24px; }}
    .meta-card {{ background: #1e2130; border-radius: 10px; padding: 16px; border: 1px solid #2d3148; }}
    .meta-card .label {{ color: #64748b; font-size: 0.78em; margin-bottom: 4px; }}
    .meta-card .value {{ font-size: 1.4em; font-weight: bold; color: #a78bfa; }}
    table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; font-size: 0.85em; }}
    th {{ background: #1e2130; color: #94a3b8; padding: 8px 10px; text-align: left;
          border-bottom: 2px solid #2d3148; }}
    td {{ padding: 7px 10px; border-bottom: 1px solid #1e2130; }}
    tr:hover td {{ background: #1a1f35; }}
    .num {{ text-align: right; font-family: monospace; }}
    .top15 td {{ background: rgba(124, 58, 237, 0.08); }}
    .badge {{ display: inline-block; padding: 2px 8px; border-radius: 12px;
              font-size: 0.78em; font-weight: 600; }}
    .buy-strong {{ background: #064e3b; color: #34d399; }}
    .buy {{ background: #052e16; color: #86efac; }}
    .neutral {{ background: #1e293b; color: #94a3b8; }}
    .sell {{ background: #431407; color: #fb923c; }}
    .sell-strong {{ background: #450a0a; color: #f87171; }}
    .funnel-bar {{ display: flex; align-items: center; gap: 12px; margin: 6px 0; }}
    .funnel-seg {{ height: 28px; background: linear-gradient(90deg, #7c3aed, #4f46e5);
                   border-radius: 4px; display: flex; align-items: center;
                   padding: 0 8px; color: white; font-size: 0.82em; white-space: nowrap; }}
    .timing-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; }}
    .timing-table {{ background: #1e2130; border-radius: 10px; padding: 12px; }}
    .footer {{ color: #475569; font-size: 0.78em; margin-top: 32px; text-align: center; }}
  </style>
</head>
<body>
  <h1>🔭 QuantMind 全A股漏斗演示</h1>
  <div class="subtitle">演示日期：2025-01-02（2025年开年第一个交易日）｜ 数据来源：Token B 实时拉取</div>

  <div class="meta-grid">
    <div class="meta-card">
      <div class="label">起始宇宙</div>
      <div class="value">{funnel_stats.get('Layer 0 初始', ('?','?'))[0]:,}</div>
      <div class="label" style="margin-top:4px">全A股股票数</div>
    </div>
    <div class="meta-card">
      <div class="label">LGBM Top-50</div>
      <div class="value">{TOP_N_LGBM}</div>
      <div class="label" style="margin-top:4px">量化因子精选</div>
    </div>
    <div class="meta-card">
      <div class="label">最终推荐</div>
      <div class="value" style="color:#34d399">{len(top15_codes)}</div>
      <div class="label" style="margin-top:4px">HRP 优化后</div>
    </div>
    <div class="meta-card">
      <div class="label">总运行时间</div>
      <div class="value" style="color:#fbbf24">{total_t:.0f}s</div>
      <div class="label" style="margin-top:4px">端到端三系统</div>
    </div>
  </div>

  <h2>漏斗筛选过程</h2>
  <table>
    <thead><tr>
      <th>步骤</th><th>描述</th><th class="num">输入</th>
      <th class="num">输出</th><th class="num">保留率</th>
    </tr></thead>
    <tbody>{funnel_rows}</tbody>
  </table>

  <h2>System 1：LGBM 排名 Top-50（★ 为最终入选）</h2>
  <table>
    <thead><tr>
      <th>排名</th><th>代码</th><th>名称</th><th>行业</th>
      <th class="num">LGBM分</th><th class="num">PB</th>
      <th class="num">近1月涨跌</th><th class="num">RSI14</th>
    </tr></thead>
    <tbody>{top50_rows}</tbody>
  </table>

  <h2>System 2 + 3：6-Agent 分析 × 历史回测验证（Top-15）</h2>
  <table>
    <thead><tr>
      <th>#</th><th>代码</th><th>名称</th><th>行业</th>
      <th>综合评级</th><th class="num">信心分</th><th>建议操作</th>
      <th class="num">历史回报</th><th class="num">Sharpe</th><th>验证状态</th>
    </tr></thead>
    <tbody>{agent_rows}</tbody>
  </table>

  <h2>⏱ 各步骤耗时</h2>
  <div class="timing-grid">
    <div class="timing-table">
      <table>
        <thead><tr><th>步骤</th><th class="num">耗时</th></tr></thead>
        <tbody>{timing_rows}</tbody>
      </table>
    </div>
  </div>

  <div class="footer">
    QuantMind — 生成时间 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    ｜ 仅供研究展示，不构成投资建议
  </div>
</body>
</html>"""
    return html


# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    funnel_stats = {}  # step → (n_in, n_out, desc)

    print(f"\n{'#'*65}")
    print(f"# QuantMind 全A股漏斗演示 — 2025-01-02")
    print(f"# 数据策略: Token A 官方端点（全市场行情）+ 本地 alpha_prices_panel（技术指标）")
    print(f"# Token B 已备用（代理稳定性问题已降级）")
    print(f"{'#'*65}")

    api = get_api_a()   # Token A — 官方端点

    # ── Step 0: 获取全A股基础数据 ───────────────────────────────────────────
    with Timer("Step0 全市场数据拉取(Token A)"):
        print("  (a) stock_basic — 全A股列表...")
        stock_basic = fetch_stock_universe(api)
        time.sleep(1.0)  # Token A 限速

        print("  (b) daily_basic(2025-01-02) — 全市场估值/流动性快照...")
        daily_basic = fetch_daily_basic_snapshot(api)
        time.sleep(1.0)

        print("  (c) daily(2025-01-02) — 全市场价格快照...")
        daily_snap = fetch_daily_snapshot(api)
        print(f"  stock_basic: {len(stock_basic)} | daily_basic: {len(daily_basic)} | daily: {len(daily_snap)}")

    n_universe = len(stock_basic)
    funnel_stats["Layer 0 初始"] = (n_universe, n_universe, "全A股（上市中）")

    # ── Step 1: Layer 1 基础质量过滤 ─────────────────────────────────────────
    with Timer("Step1 Layer1 基础质量"):
        l1 = apply_layer1(stock_basic, daily_basic)
    funnel_stats["Layer 1 基础质量"] = (n_universe, len(l1), "非ST + 上市满6月 + 市值≥20亿 + 非北交所")

    # ── Step 2: Layer 2 流动性过滤 ────────────────────────────────────────────
    with Timer("Step2 Layer2 流动性"):
        l2 = apply_layer2(l1)
    funnel_stats["Layer 2 流动性"] = (len(l1), len(l2), "换手率≥0.3% + 流通市值≥10亿 + 排除涨跌停")

    # ── Step 3: 从本地 alpha_prices_panel 计算技术指标 ──────────────────────
    with Timer("Step3 技术指标(本地历史数据)"):
        l2_codes = l2["ts_code"].tolist()
        price_panel = pd.read_parquet(PRICE_PANEL)
        # 筛选 2025-01-02 前 60 个交易日的数据
        hist_start = DEMO_DATE - pd.DateOffset(days=90)
        price_hist = price_panel[
            (price_panel["ts_code"].isin(l2_codes)) &
            (price_panel["trade_date"] >= hist_start) &
            (price_panel["trade_date"] <= DEMO_DATE)
        ].copy()
        print(f"  本地价格历史: {len(price_hist)} 行, 覆盖 {price_hist['ts_code'].nunique()} / {len(l2_codes)} 只")

    # ── Step 4: 计算技术指标 + Layer 3 过滤 ──────────────────────────────────
    with Timer("Step4 技术指标+Layer3趋势"):
        tech_df = compute_technical_features(price_hist)
        l3 = apply_layer3(l2, tech_df)
        # 对 没有本地价格的股票（非 Alpha 1374），不做技术过滤
        no_price_stocks = l2[~l2["ts_code"].isin(tech_df["ts_code"])]
        l3 = pd.concat([l3, no_price_stocks], ignore_index=True).drop_duplicates("ts_code")
        print(f"  (含 {len(no_price_stocks)} 只无本地历史的非Alpha股，不做技术过滤)")
    funnel_stats["Layer 3 技术趋势"] = (len(l2), len(l3), "动量≥-15% + RSI<80 + 布林带合理（Alpha宇宙内）")

    # ── Step 5: 加载 Alpha 因子 + 构建特征矩阵 ────────────────────────────────
    with Timer("Step5 因子矩阵构建"):
        alpha_factors = load_alpha_factors()
        feat_df, feature_cols = build_feature_matrix(l3, alpha_factors, tech_df)

    # ── Step 6: LGBM 排名 ─────────────────────────────────────────────────────
    with Timer("Step6 LGBM排名"):
        ranked = lgbm_rank(feat_df, feature_cols)
        top50 = ranked.head(TOP_N_LGBM)
        top50_codes = top50["ts_code"].tolist()
        print(f"  LGBM Top-50: {top50_codes[:5]} ...")
    funnel_stats["Layer 4+5 LGBM排名"] = (len(feat_df), TOP_N_LGBM, "LGBM LambdaRanker 综合因子评分")

    # ── Step 7: HRP 优化 → Top-15 ─────────────────────────────────────────────
    with Timer("Step7 HRP仓位优化→Top15"):
        # price_panel already loaded in Step 3
        top15_codes = hrp_select(top50_codes, price_panel)
        print(f"  HRP Top-15: {top15_codes}")
    funnel_stats["Layer 6 HRP优化"] = (TOP_N_LGBM, len(top15_codes), "HRP层次风险平价选权重最大15只")

    # ── Step 8: 6-Agent 投资分析 ──────────────────────────────────────────────
    with Timer("System2 6-Agent分析"):
        print(f"  对 {len(top15_codes)} 只股票运行 6-Agent 分析...")
        agent_results = run_agent_analysis(top15_codes)
        print(f"  成功分析: {len(agent_results)} 只")

    # ── Step 9: 历史回测验证 ─────────────────────────────────────────────────
    with Timer("System3 历史回测验证"):
        validation_results = run_backtest_validation(agent_results)

    # ── Step 10: 生成报告 ─────────────────────────────────────────────────────
    with Timer("报告生成"):
        # 合并股票名称到 ranked
        ranked_with_info = ranked.merge(
            stock_basic[["ts_code", "name", "industry"]],
            on="ts_code", how="left"
        )
        html = build_html_report(
            funnel_stats=funnel_stats,
            ranked_df=ranked_with_info,
            top15_codes=top15_codes,
            agent_results=agent_results,
            validation_results=validation_results,
            stock_info=stock_basic,
        )
        report_path = OUT_DIR / "full_A_demo_report.html"
        report_path.write_text(html, encoding="utf-8")
        print(f"  报告已保存: {report_path}")

        # 保存 JSON 摘要
        summary = {
            "demo_date": "2025-01-02",
            "universe_size": n_universe,
            "funnel_stats": {k: {"n_in": v[0], "n_out": v[1], "desc": v[2]} for k, v in funnel_stats.items()},
            "top15": top15_codes,
            "timings": TIMINGS,
            "total_time_s": sum(TIMINGS.values()),
        }
        (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── 打印最终摘要 ──────────────────────────────────────────────────────────
    total_time = sum(TIMINGS.values())
    print(f"\n{'='*65}")
    print(f"🎯 演示完成！总耗时 {total_time:.1f}s ({total_time/60:.1f} 分钟)")
    print(f"\n📊 漏斗摘要：")
    for step, (n_in, n_out, desc) in funnel_stats.items():
        bar = "█" * max(1, int(n_out / n_in * 20))
        print(f"  {step:<20} {n_in:5,} → {n_out:5,}  {bar}")

    print(f"\n🏆 最终 Top-15 推荐：")
    for i, code in enumerate(top15_codes, 1):
        strat = agent_results.get(code)
        rating = getattr(strat, "rating", "—") if strat else "—"
        name = stock_basic[stock_basic["ts_code"] == code]["name"].values
        name = name[0] if len(name) > 0 else code
        print(f"  {i:2d}. {code} {name:<10} {rating}")

    print(f"\n📄 完整报告：{report_path}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
