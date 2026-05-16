"""quantmind.backtest.agent_backtest — Agent 决策回测（核武器）.

核心理念：
  让 Agent/LLM 对历史每个调仓时点的"当时数据"做出投资决策，
  再用真实的未来数据验证 alpha。

完整流程（每个调仓日 Q）：
  1. 加载该日 PIT 快照（严格只用 <= Q 的历史数据）
  2. LGBMRanker 粗排：全域 → Top-50
  3. LLMListwiseReranker 精排：Top-50 → Top-10
  4. 构建等权组合，持有到下一个调仓日
  5. 记录真实未来收益（事后验证）

基线对比：
  - random_baseline：随机选 10 只
  - lgbm_only：只用 LGBM，不用 LLM rerank
  - equal_weight_index：等权沪深300

PIT 安全原则：
  - 所有决策只使用 <= decision_date 的因子和财务数据
  - LLM 推理摘要只包含当时可见信息
  - 未来收益仅用于事后分析，严格与决策隔离
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

__all__ = ["AgentBacktester", "AgentBacktestResult", "RebalanceRecord"]

_TRADING_DAYS_PER_YEAR = 252


@dataclass
class RebalanceRecord:
    """单次调仓记录."""

    decision_date: date
    universe_size: int          # 决策时的候选股票数

    # 各策略选股结果
    agent_picks: list[str]       # Agent (LGBM + LLM) 选出的股票
    lgbm_picks: list[str]        # 只用 LGBM 的选股
    random_picks: list[str]      # 随机选股（基线）
    index_picks: list[str]       # 等权指数（基线）

    # LGBM 因子分数（Top-50）
    lgbm_scores: dict[str, float] = field(default_factory=dict)

    # LLM rerank 理由摘要
    llm_reasoning: str = ""

    # 持有期收益（事后填充）
    agent_return: float | None = None
    lgbm_return: float | None = None
    random_return: float | None = None
    index_return: float | None = None

    # 持有天数
    holding_days: int = 0


@dataclass
class AgentBacktestResult:
    """Agent 回测结果."""

    start_date: date
    end_date: date
    rebalance_freq: str
    n_rebalances: int

    # 各策略收益序列（按调仓期）
    agent_returns: pd.Series = field(default_factory=pd.Series)
    lgbm_returns: pd.Series = field(default_factory=pd.Series)
    random_returns: pd.Series = field(default_factory=pd.Series)
    index_returns: pd.Series = field(default_factory=pd.Series)

    # 调仓记录
    records: list[RebalanceRecord] = field(default_factory=list)

    # 各策略累积性能
    metrics: dict[str, dict[str, Any]] = field(default_factory=dict)

    def summary_table(self) -> pd.DataFrame:
        """生成策略对比摘要表."""
        rows = []
        for name, ret_series in [
            ("Agent (LGBM+LLM)", self.agent_returns),
            ("LGBM Only", self.lgbm_returns),
            ("Random", self.random_returns),
            ("Equal Weight Index", self.index_returns),
        ]:
            if ret_series.empty:
                continue
            cumret = (1 + ret_series).prod() - 1
            ann_ret = ret_series.mean() * (_TRADING_DAYS_PER_YEAR / max(1, len(ret_series)))
            sharpe = ret_series.mean() / (ret_series.std() + 1e-10) * np.sqrt(
                _TRADING_DAYS_PER_YEAR / max(1, len(ret_series))
            )
            rows.append({
                "Strategy": name,
                "Total Return": f"{cumret:.2%}",
                "Ann Return": f"{ann_ret:.2%}",
                "Sharpe": f"{sharpe:.3f}",
                "Win Rate": f"{(ret_series > 0).mean():.1%}",
                "N Periods": len(ret_series),
            })
        return pd.DataFrame(rows)


class AgentBacktester:
    """Agent 决策历史回测器.

    Args:
        features_df:     因子特征 DataFrame（MultiIndex date/ticker）
        prices_df:       OHLCV 价格（MultiIndex date/ticker）
        lgbm_model_path: 训练好的 LGBM 模型路径（可选）
        llm_provider:    LLM 提供商（默认 ollama）
        llm_model:       LLM 模型名（默认 qwen2.5:7b）
        top_lgbm_n:      LGBM 粗排数量（默认 50）
        top_llm_n:       LLM 精排最终选股数（默认 10）
        seed:            随机种子（随机基线用）
    """

    def __init__(
        self,
        features_df: pd.DataFrame | None = None,
        prices_df: pd.DataFrame | None = None,
        lgbm_model_path: str | Path | None = None,
        llm_provider: str = "ollama",
        llm_model: str = "qwen2.5:7b",
        top_lgbm_n: int = 50,
        top_llm_n: int = 10,
        seed: int = 42,
    ) -> None:
        self.features_df = features_df
        self.prices_df = prices_df
        self.lgbm_model_path = Path(lgbm_model_path) if lgbm_model_path else None
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.top_lgbm_n = top_lgbm_n
        self.top_llm_n = top_llm_n
        self.seed = seed
        self._lgbm_model = None
        self._rng = random.Random(seed)

    # ── 主入口 ────────────────────────────────────────────────────────────────

    def run(
        self,
        start_date: date,
        end_date: date,
        universe: list[str],
        rebalance_freq: str = "Q",  # Q=季度, M=月度
        features_df: pd.DataFrame | None = None,
        prices_df: pd.DataFrame | None = None,
    ) -> AgentBacktestResult:
        """运行 Agent 决策回测.

        Args:
            start_date:     回测开始日
            end_date:       回测结束日
            universe:       候选股票池
            rebalance_freq: 调仓频率（Q=季度, M=月度）
            features_df:    可选：覆盖构造函数数据
            prices_df:      可选：覆盖构造函数数据

        Returns:
            AgentBacktestResult
        """
        feat = features_df if features_df is not None else self.features_df
        prices = prices_df if prices_df is not None else self.prices_df

        if prices is None:
            raise ValueError("prices_df 不能为空")

        # 生成调仓日序列
        rebalance_dates = self._generate_rebalance_dates(
            start_date, end_date, prices, rebalance_freq
        )

        if len(rebalance_dates) < 2:
            logger.warning("[AgentBacktest] 调仓日不足，无法运行")
            return AgentBacktestResult(
                start_date=start_date,
                end_date=end_date,
                rebalance_freq=rebalance_freq,
                n_rebalances=0,
            )

        logger.info(
            f"[AgentBacktest] {start_date}~{end_date} | "
            f"freq={rebalance_freq} | {len(rebalance_dates)-1} periods | "
            f"{len(universe)} stocks"
        )

        # 尝试加载 LGBM 模型
        lgbm_model = self._load_lgbm_model()

        records: list[RebalanceRecord] = []

        for i in range(len(rebalance_dates) - 1):
            decision_date = rebalance_dates[i]
            next_date = rebalance_dates[i + 1]

            logger.info(f"[AgentBacktest] 调仓 {i+1}/{len(rebalance_dates)-1}: {decision_date}")

            # ── 1. PIT 特征（严格只用 <= decision_date 的数据）────────────────
            pit_features = self._get_pit_features(feat, decision_date, universe)

            # 可交易股票（当日有价格）
            tradeable = self._get_tradeable(prices, decision_date, universe)
            if len(tradeable) < self.top_llm_n:
                logger.warning(f"[AgentBacktest] {decision_date}: 可交易股票不足 ({len(tradeable)})")
                continue

            # ── 2. LGBM 粗排 → Top-N ────────────────────────────────────────
            lgbm_scores, lgbm_top = self._lgbm_rank(
                lgbm_model, pit_features, tradeable, self.top_lgbm_n
            )

            # ── 3. LLM 精排 → Top-10 ────────────────────────────────────────
            agent_picks, llm_reasoning = self._llm_rerank(
                lgbm_top, lgbm_scores, decision_date
            )

            # ── 4. 各基线选股 ────────────────────────────────────────────────
            lgbm_picks = lgbm_top[:self.top_llm_n]
            random_picks = self._rng.sample(tradeable, min(self.top_llm_n, len(tradeable)))
            index_picks = tradeable[:self.top_llm_n]  # 简化：取前 N 只（等权指数基线）

            # ── 5. 计算持有期收益（事后）────────────────────────────────────
            holding_days = (next_date - decision_date).days
            agent_ret = self._compute_period_return(prices, agent_picks, decision_date, next_date)
            lgbm_ret = self._compute_period_return(prices, lgbm_picks, decision_date, next_date)
            random_ret = self._compute_period_return(prices, random_picks, decision_date, next_date)
            index_ret = self._compute_period_return(prices, index_picks, decision_date, next_date)

            record = RebalanceRecord(
                decision_date=decision_date,
                universe_size=len(tradeable),
                agent_picks=agent_picks,
                lgbm_picks=lgbm_picks,
                random_picks=random_picks,
                index_picks=index_picks,
                lgbm_scores=lgbm_scores,
                llm_reasoning=llm_reasoning,
                agent_return=agent_ret,
                lgbm_return=lgbm_ret,
                random_return=random_ret,
                index_return=index_ret,
                holding_days=holding_days,
            )
            records.append(record)

        # ── 构建结果 ──────────────────────────────────────────────────────────
        result = self._build_result(
            records, start_date, end_date, rebalance_freq
        )

        logger.info(f"[AgentBacktest] Done. {len(records)} periods.")
        self._log_summary(result)
        return result

    # ── LGBM 粗排 ─────────────────────────────────────────────────────────────

    def _lgbm_rank(
        self,
        model: Any,
        pit_features: pd.DataFrame | None,
        tradeable: list[str],
        top_n: int,
    ) -> tuple[dict[str, float], list[str]]:
        """LGBM 因子打分并排名."""
        if model is None or pit_features is None or pit_features.empty:
            # 无模型时用随机分数
            scores = {t: self._rng.random() for t in tradeable}
            top = sorted(scores, key=lambda x: scores[x], reverse=True)[:top_n]
            return scores, top

        try:
            # 取最新截面（decision_date 的特征）
            feature_cols = [c for c in pit_features.columns
                           if c not in ("date", "ticker", "label")]
            available = [t for t in tradeable if t in pit_features.index]
            if not available:
                scores = {t: 0.0 for t in tradeable}
                return scores, tradeable[:top_n]

            X = pit_features.loc[available, feature_cols].fillna(0)
            raw_scores = model.predict(X)
            scores = dict(zip(available, raw_scores.tolist()))

            # 没有分数的股票给 0
            for t in tradeable:
                if t not in scores:
                    scores[t] = 0.0

            top = sorted(scores, key=lambda x: scores[x], reverse=True)[:top_n]
            return scores, top
        except Exception as e:
            logger.warning(f"[AgentBacktest] LGBM 评分失败: {e}")
            scores = {t: 0.0 for t in tradeable}
            return scores, tradeable[:top_n]

    # ── LLM 精排 ──────────────────────────────────────────────────────────────

    def _llm_rerank(
        self,
        candidates: list[str],
        scores: dict[str, float],
        decision_date: date,
    ) -> tuple[list[str], str]:
        """调用 LLM 对 Top-N 候选进行 listwise 精排."""
        if not candidates:
            return [], "候选列表为空"

        try:
            from quantmind.core.llm_router import LLMRouter

            router = LLMRouter(provider=self.llm_provider, model=self.llm_model)

            # 构造精排 Prompt
            prompt = self._build_rerank_prompt(candidates, scores, decision_date)

            response = router.chat(prompt, max_tokens=512, temperature=0.1)
            content = response.content if hasattr(response, "content") else str(response)

            # 解析 LLM 输出的排序
            ranked = self._parse_rerank_output(content, candidates)
            top = ranked[:self.top_llm_n]
            return top, content[:500]

        except ImportError:
            logger.warning("[AgentBacktest] LLMRouter 不可用，使用 LGBM 分数直接取 Top-N")
            top = candidates[:self.top_llm_n]
            return top, "LLM 不可用，使用 LGBM 分数"
        except Exception as e:
            logger.warning(f"[AgentBacktest] LLM rerank 失败: {e}，使用 LGBM 分数")
            top = candidates[:self.top_llm_n]
            return top, f"LLM 失败（{e}），使用 LGBM 分数"

    def _build_rerank_prompt(
        self,
        candidates: list[str],
        scores: dict[str, float],
        decision_date: date,
    ) -> str:
        """构造 LLM 精排 Prompt."""
        ticker_list = "\n".join(
            f"  {i+1}. {t} (LGBM score: {scores.get(t, 0):.4f})"
            for i, t in enumerate(candidates[:20])  # 最多展示20只
        )
        return f"""你是一位量化投资组合经理，截止日期 {decision_date}。

以下是 LGBM 因子模型筛选的候选股票（已按因子分数排序）：
{ticker_list}

请从中选出最值得投资的 {self.top_llm_n} 只股票，考虑以下维度：
1. 因子分数高（量化信号强）
2. 分散化（不集中于单一行业）
3. 排除明显风险股（如退市风险、ST股票）

请只输出股票代码列表，格式如下（每行一个代码）：
SELECTED:
600519.SH
000858.SZ
...（共{self.top_llm_n}只）"""

    def _parse_rerank_output(
        self,
        content: str,
        candidates: list[str],
    ) -> list[str]:
        """解析 LLM 输出中的股票代码列表."""
        import re
        # 找 "SELECTED:" 之后的内容
        lines = content.split("\n")
        in_selected = False
        parsed = []
        for line in lines:
            line = line.strip()
            if "SELECTED:" in line.upper():
                in_selected = True
                continue
            if in_selected:
                # 匹配股票代码格式
                m = re.search(r"\d{6}\.[A-Z]+", line)
                if m:
                    ticker = m.group()
                    if ticker in candidates:
                        parsed.append(ticker)
                elif line and not line.startswith("#"):
                    # 尝试纯数字 + 市场后缀
                    pass

        # 如果解析失败，返回原顺序
        if not parsed:
            return candidates
        # 去重保序，补充未解析的
        seen = set(parsed)
        for c in candidates:
            if c not in seen:
                parsed.append(c)
        return parsed

    # ── 收益计算 ──────────────────────────────────────────────────────────────

    def _compute_period_return(
        self,
        prices: pd.DataFrame,
        picks: list[str],
        start: date,
        end: date,
    ) -> float | None:
        """计算等权组合从 start 到 end 的持有期收益."""
        if not picks:
            return None

        returns = []
        for ticker in picks:
            try:
                r = self._get_stock_return(prices, ticker, start, end)
                if r is not None:
                    returns.append(r)
            except Exception:
                pass

        if not returns:
            return None

        # 等权组合收益
        return float(np.mean(returns))

    def _get_stock_return(
        self,
        prices: pd.DataFrame,
        ticker: str,
        start: date,
        end: date,
    ) -> float | None:
        """计算单只股票持有期收益."""
        try:
            if isinstance(prices.index, pd.MultiIndex):
                ticker_prices = prices.xs(ticker, level="ticker")["close"]
            else:
                ticker_prices = prices[ticker] if ticker in prices.columns else None

            if ticker_prices is None or ticker_prices.empty:
                return None

            # 找最近的 start 和 end 价格
            start_ts = pd.Timestamp(start)
            end_ts = pd.Timestamp(end)

            available = ticker_prices.dropna()
            if available.empty:
                return None

            # 找 start 之后最近的价格
            start_prices = available[available.index >= start_ts]
            if start_prices.empty:
                return None
            p_start = float(start_prices.iloc[0])

            # 找 end 之前最近的价格
            end_prices = available[available.index <= end_ts]
            if end_prices.empty:
                return None
            p_end = float(end_prices.iloc[-1])

            if p_start <= 0:
                return None
            return (p_end - p_start) / p_start

        except Exception:
            return None

    # ── 辅助方法 ─────────────────────────────────────────────────────────────

    def _generate_rebalance_dates(
        self,
        start: date,
        end: date,
        prices: pd.DataFrame,
        freq: str,
    ) -> list[date]:
        """生成调仓日期序列（取最近的交易日）."""
        # 获取所有交易日
        if isinstance(prices.index, pd.MultiIndex):
            all_dates = sorted(
                set(d.date() if hasattr(d, "date") else d
                    for d in prices.index.get_level_values(0))
            )
        else:
            all_dates = sorted(
                d.date() if hasattr(d, "date") else d
                for d in prices.index.unique()
            )

        trading_set = set(all_dates)

        # 生成候选调仓日
        rebalance_dates = []
        current = start
        while current <= end:
            # 找当月/当季最后一个交易日
            if freq == "Q":
                # 季末月份：3, 6, 9, 12
                quarter_end_month = ((current.month - 1) // 3 + 1) * 3
                if current.month != quarter_end_month:
                    # 跳到季末
                    import calendar
                    _, last_day = calendar.monthrange(current.year, quarter_end_month)
                    current = date(current.year, quarter_end_month, last_day)
                else:
                    import calendar
                    _, last_day = calendar.monthrange(current.year, current.month)
                    current = date(current.year, current.month, last_day)
            else:  # M
                import calendar
                _, last_day = calendar.monthrange(current.year, current.month)
                current = date(current.year, current.month, last_day)

            # 往前找最近交易日
            rd = current
            for _ in range(10):
                if rd in trading_set:
                    if start <= rd <= end:
                        rebalance_dates.append(rd)
                    break
                rd -= timedelta(days=1)

            # 步进到下一期
            if freq == "Q":
                current = date(current.year + (current.month > 9), (current.month % 12) + 1, 1)
                # 更简单：加3个月
                month = current.month + 3 - 12 if current.month + 3 > 12 else current.month + 3
                year = current.year + (1 if current.month + 3 > 12 else 0)
                current = date(year, month, 1)
            else:
                month = current.month % 12 + 1
                year = current.year + (1 if current.month == 12 else 0)
                current = date(year, month, 1)

        # 去重排序
        rebalance_dates = sorted(set(rebalance_dates))

        # 加入 end_date（用于计算最后一期收益）
        end_rd = end
        for _ in range(10):
            if end_rd in trading_set:
                if end_rd not in rebalance_dates:
                    rebalance_dates.append(end_rd)
                break
            end_rd -= timedelta(days=1)

        return sorted(set(rebalance_dates))

    def _get_pit_features(
        self,
        features: pd.DataFrame | None,
        decision_date: date,
        universe: list[str],
    ) -> pd.DataFrame | None:
        """获取 PIT 特征（最新截面）."""
        if features is None:
            return None
        try:
            ts = pd.Timestamp(decision_date)
            if isinstance(features.index, pd.MultiIndex):
                dates = features.index.get_level_values(0)
                # 取 <= decision_date 的最近一个截面
                available_dates = sorted(set(dates[dates <= ts]))
                if not available_dates:
                    return None
                latest = available_dates[-1]
                pit = features.xs(latest, level=0)
                pit = pit.loc[pit.index.isin(universe)]
                return pit
            else:
                pit = features.loc[features.index <= ts]
                return pit
        except Exception:
            return None

    def _get_tradeable(
        self,
        prices: pd.DataFrame,
        decision_date: date,
        universe: list[str],
    ) -> list[str]:
        """获取当日可交易股票."""
        try:
            ts = pd.Timestamp(decision_date)
            if isinstance(prices.index, pd.MultiIndex):
                today = prices.xs(ts, level=0) if ts in prices.index.get_level_values(0) else pd.DataFrame()
                if today.empty:
                    # 找最近交易日
                    dates = sorted(set(prices.index.get_level_values(0)))
                    avail = [d for d in dates if d <= ts]
                    if not avail:
                        return universe[:50]
                    today = prices.xs(avail[-1], level=0)
                return [t for t in universe if t in today.index]
            else:
                return universe
        except Exception:
            return universe[:50]

    def _load_lgbm_model(self) -> Any:
        """懒加载 LGBM 模型."""
        if self._lgbm_model is not None:
            return self._lgbm_model
        if self.lgbm_model_path is None:
            return None
        try:
            import lightgbm as lgb
            self._lgbm_model = lgb.Booster(model_file=str(self.lgbm_model_path))
            logger.info(f"[AgentBacktest] LGBM 模型加载: {self.lgbm_model_path}")
            return self._lgbm_model
        except Exception as e:
            logger.warning(f"[AgentBacktest] LGBM 模型加载失败: {e}")
            return None

    def _build_result(
        self,
        records: list[RebalanceRecord],
        start_date: date,
        end_date: date,
        freq: str,
    ) -> AgentBacktestResult:
        """构建 AgentBacktestResult."""
        def _to_series(returns_list: list[float | None], dates: list[date]) -> pd.Series:
            vals = {d: r for d, r in zip(dates, returns_list) if r is not None}
            s = pd.Series(vals, name="return")
            s.index = pd.to_datetime(s.index)
            return s

        dates = [r.decision_date for r in records]
        agent_rets = [r.agent_return for r in records]
        lgbm_rets = [r.lgbm_return for r in records]
        random_rets = [r.random_return for r in records]
        index_rets = [r.index_return for r in records]

        result = AgentBacktestResult(
            start_date=start_date,
            end_date=end_date,
            rebalance_freq=freq,
            n_rebalances=len(records),
            agent_returns=_to_series(agent_rets, dates),
            lgbm_returns=_to_series(lgbm_rets, dates),
            random_returns=_to_series(random_rets, dates),
            index_returns=_to_series(index_rets, dates),
            records=records,
        )

        # 计算各策略性能指标
        from quantmind.backtest.metrics import PerformanceMetrics
        pm = PerformanceMetrics()
        for name, series in [
            ("agent", result.agent_returns),
            ("lgbm_only", result.lgbm_returns),
            ("random", result.random_returns),
            ("index", result.index_returns),
        ]:
            if not series.empty:
                result.metrics[name] = pm.compute_all(series)

        return result

    def _log_summary(self, result: AgentBacktestResult) -> None:
        """打印简单摘要."""
        try:
            table = result.summary_table()
            logger.info(f"\n{table.to_string(index=False)}")
        except Exception:
            pass
