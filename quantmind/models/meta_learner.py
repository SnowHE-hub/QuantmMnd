"""quantmind.models.meta_learner — Agent 融合 Meta-Learner.

原理
====
替代 6 个 Investment Agent 的固定等权融合，学习一个 Ridge 回归：

    final_score = Σ w_i(regime) × agent_proxy_score_i + ε

其中 agent_proxy_score 是基于面板因子构造的 6 维代理分数（当真实 Agent
模型分数未保存时使用），或直接使用真实 Agent 输出（当可用时）。

6 个 Agent 代理分数（Panel Factor Proxies）
==========================================
  Valuation:  book_to_market, earnings_yield, dividend_yield_ttm
  Momentum:   momentum_6m, momentum_3m, relative_strength_vs_csi300_60d
  Quality:    ocf_to_revenue_ttm, accruals（负向）
  Sentiment:  north_hold_ratio, margin_buy_amount_20d
  Risk:       volatility_3m（负向）, max_drawdown_3m（负向）
  Strategy:   综合得分（上述 5 个的等权均值，暂时）

训练策略
========
  - 训练数据：data/feedback/realized_pnl.parquet（63天真实收益）
  - 增量更新：每次新增反馈数据后重新 fit（扩展窗口）
  - 模型：Ridge(alpha=1.0)，L2 正则避免过拟合小样本
  - 特征：6 个代理分数 + regime_dummy（若有）

输出
====
  data/meta_learner/
    meta_learner.pkl         训练好的 Ridge 模型
    meta_learner_meta.json   系数解读 + 训练统计
    weight_history.jsonl     历次训练的权重变化记录

接口
====
  ml = AgentMetaLearner()
  ml.fit_from_feedback(feedback_path, panel)
  scores = ml.predict(agent_proxies_df)
  weights = ml.get_weights()  # 返回各 agent 当前权重
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)  # type: ignore[assignment]

__all__ = ["AgentMetaLearner", "compute_agent_proxies", "AGENT_PROXY_CONFIG"]

# ─────────────────────────────────────────────────────────────────────────────
# Agent Proxy 配置
# ─────────────────────────────────────────────────────────────────────────────

# 定义每个 Agent 的代理因子及权重
# 格式: {agent_name: [(factor_col, weight), ...]}
# 正权重 = 该因子越大 → agent 越看好；负权重 = 反向
AGENT_PROXY_CONFIG: dict[str, list[tuple[str, float]]] = {
    "valuation": [
        ("book_to_market",    0.35),  # 高账面市值比 = 便宜
        ("earnings_yield",    0.40),  # 高盈利收益率 = 便宜
        ("dividend_yield_ttm", 0.25), # 高股息率 = 便宜
    ],
    # 2026-05-24 IC 诊断修正：
    #   momentum_12m_skip_1m IC=+0.025 (63d) ✅ → 长期动量有效
    #   momentum_6m          IC≈0 → 中性，轻权
    #   momentum_1m          IC=-0.024 ❌ → 短期反转，负权重（超买回调）
    #   relative_strength_vs_csi300_60d IC=-0.029 ❌ → 移除（方向相反）
    "momentum": [
        ("momentum_12m_skip_1m",  0.50),  # 长期动量（12m skip 1m），IC>0 ✅
        ("momentum_6m",           0.20),  # 中期，IC≈0，轻权
        ("momentum_1m",          -0.30),  # 短期反转因子（负权 = 低短期超买更好）
    ],
    "quality": [
        ("ocf_to_revenue_ttm",  0.60),  # 高现金流质量 = 好
        ("accruals",           -0.40),  # 高应计项目 = 差（负向）
    ],
    "sentiment": [
        ("north_hold_ratio",         0.45),  # 北向持仓比例 = 外资认可
        ("margin_buy_amount_20d",    0.30),  # 融资买入 = 市场热度
        ("margin_buy_intensity",     0.25),  # 融资强度
    ],
    "risk": [
        ("volatility_3m",    -0.40),  # 高波动 = 高风险（负向）
        ("max_drawdown_3m",  -0.40),  # 大回撤 = 高风险（负向）
        ("beta_252d",        -0.20),  # 高 beta = 高系统风险
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# 代理分数计算
# ─────────────────────────────────────────────────────────────────────────────

def _zscore_series(s: pd.Series) -> pd.Series:
    """截面 z-score，处理常数列."""
    std = s.std()
    if std < 1e-9:
        return s * 0.0
    return (s - s.mean()) / std


def compute_agent_proxies(
    xs: pd.DataFrame,
    config: dict[str, list[tuple[str, float]]] | None = None,
) -> pd.DataFrame:
    """从单期截面数据计算 6 个 Agent 代理分数.

    Args:
        xs:      单期截面 DataFrame（行=ticker，列=因子）
        config:  AGENT_PROXY_CONFIG 或自定义配置

    Returns:
        DataFrame(index=ticker, columns=agent_names)，已 z-score 标准化
    """
    if config is None:
        config = AGENT_PROXY_CONFIG

    result: dict[str, pd.Series] = {}

    for agent_name, factor_weights in config.items():
        scores: pd.Series | None = None
        total_weight = 0.0

        for factor_col, weight in factor_weights:
            if factor_col not in xs.columns:
                continue
            fv = xs[factor_col].copy()
            fv_z = _zscore_series(fv.fillna(fv.median()))
            if scores is None:
                scores = fv_z * weight
            else:
                scores = scores + fv_z * weight
            total_weight += abs(weight)

        if scores is None or total_weight < 1e-9:
            result[agent_name] = pd.Series(0.0, index=xs.index)
        else:
            result[agent_name] = _zscore_series(scores / total_weight)

    # Strategy = 前 5 个 agent 的等权均值（先验：各 agent 权重相等）
    agent_cols = list(result.keys())
    if agent_cols:
        strategy_score = pd.concat(list(result.values()), axis=1).mean(axis=1)
        result["strategy"] = _zscore_series(strategy_score)

    return pd.DataFrame(result).reindex(xs.index)


# ─────────────────────────────────────────────────────────────────────────────
# Meta-Learner 主类
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MetaLearnerState:
    """持久化状态."""
    trained_at:     str = ""
    n_samples:      int = 0
    n_dates:        int = 0
    agent_names:    list[str] = field(default_factory=list)
    coefficients:   dict[str, float] = field(default_factory=dict)
    intercept:      float = 0.0
    train_r2:       float = float("nan")
    ridge_alpha:    float = 1.0
    weight_history: list[dict] = field(default_factory=list)


class AgentMetaLearner:
    """学习型 Agent 权重融合器.

    初次使用等权重（系数全为 1/6），随着 realized_pnl 数据积累后自动更新。

    用法
    ----
    ml = AgentMetaLearner()
    ml.fit_from_feedback(feedback_path, panel)   # 有反馈数据时训练
    scores = ml.predict_from_panel(xs)           # 单期截面打分
    weights = ml.get_weights()                   # 查看各 agent 权重
    ml.save(path)
    ml = AgentMetaLearner.load(path)
    """

    DEFAULT_AGENT_NAMES = ["valuation", "momentum", "quality", "sentiment", "risk", "strategy"]

    def __init__(
        self,
        ridge_alpha: float = 1.0,
        proxy_config: dict | None = None,
        save_dir: Path | None = None,
    ) -> None:
        if not _HAS_SKLEARN:
            raise ImportError("需要 scikit-learn：pip install scikit-learn")
        self.ridge_alpha   = ridge_alpha
        self.proxy_config  = proxy_config or AGENT_PROXY_CONFIG
        self.save_dir      = save_dir or Path("data/meta_learner")
        self._model: Ridge | None = None
        self._scaler: StandardScaler = StandardScaler()
        self._state = MetaLearnerState(ridge_alpha=ridge_alpha)
        self._fitted = False
        self._equal_weights = {a: 1.0 / len(self.DEFAULT_AGENT_NAMES) for a in self.DEFAULT_AGENT_NAMES}

    # ── 训练 ──────────────────────────────────────────────────────────────────

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> "AgentMetaLearner":
        """训练 Ridge meta-learner.

        Args:
            X:  (n_samples, n_agents) agent 代理分数矩阵
            y:  (n_samples,) 目标：实际收益排名（rank 或原始收益均可）
        """
        common = X.index.intersection(y.index)
        X_ = X.loc[common].fillna(0.0).values
        y_ = y.loc[common].values.astype(float)

        if len(common) < 10:
            logger.warning(f"训练样本太少（{len(common)}），维持等权重")
            return self

        X_scaled = self._scaler.fit_transform(X_)
        self._model = Ridge(alpha=self.ridge_alpha)
        self._model.fit(X_scaled, y_)
        self._fitted = True

        # 更新状态
        y_pred = self._model.predict(X_scaled)
        ss_res = np.sum((y_ - y_pred) ** 2)
        ss_tot = np.sum((y_ - y_.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 1e-9 else float("nan")

        agent_names = list(X.columns)
        coef_raw = self._model.coef_
        # 归一化系数为权重（确保正负方向合理）
        coef_abs = np.abs(coef_raw)
        coef_norm = coef_raw / (coef_abs.sum() + 1e-9)

        self._state.trained_at   = datetime.now().isoformat(timespec="seconds")
        self._state.n_samples    = len(common)
        self._state.agent_names  = agent_names
        self._state.coefficients = {n: round(float(c), 6) for n, c in zip(agent_names, coef_norm)}
        self._state.intercept    = float(self._model.intercept_)
        self._state.train_r2     = round(float(r2), 4) if np.isfinite(r2) else float("nan")

        self._state.weight_history.append({
            "ts":           self._state.trained_at,
            "n_samples":    self._state.n_samples,
            "coefficients": self._state.coefficients,
            "train_r2":     self._state.train_r2,
        })

        logger.info(
            f"[MetaLearner] 训练完成: n={len(common)}, R²={r2:.3f}  "
            f"权重: " + ", ".join(f"{n}={c:+.3f}" for n, c in self._state.coefficients.items())
        )
        return self

    def fit_from_panel(
        self,
        panel: pd.DataFrame,
        label_col: str = "forward_return_63d",
        train_dates: list[pd.Timestamp] | None = None,
    ) -> "AgentMetaLearner":
        """从因子面板构建 proxy 分数并训练.

        逐期计算 agent proxy scores，再与标签拼接训练 Ridge。
        """
        all_dates = sorted(panel.index.get_level_values("as_of").unique())
        if train_dates:
            dates = [d for d in all_dates if d in train_dates]
        else:
            dates = all_dates

        X_rows: list[pd.DataFrame] = []
        y_rows: list[pd.Series] = []

        for d in dates:
            xs = panel.xs(d, level="as_of")
            labels = xs[label_col].dropna()
            if len(labels) < 20:
                continue
            proxies = compute_agent_proxies(xs, self.proxy_config)
            proxies = proxies.reindex(labels.index).fillna(0.0)
            X_rows.append(proxies)
            y_rows.append(labels)

        if not X_rows:
            logger.warning("[MetaLearner] 无有效训练数据")
            return self

        X_all = pd.concat(X_rows, axis=0)
        y_all = pd.concat(y_rows, axis=0)
        self._state.n_dates = len(dates)

        return self.fit(X_all, y_all)

    def fit_from_feedback(
        self,
        feedback_path: Path,
        panel: pd.DataFrame,
    ) -> "AgentMetaLearner":
        """从 realized_pnl.parquet + 面板 proxy 分数训练.

        用实际收益（而非面板预设标签）作为训练目标，更接近真实反馈信号。
        """
        if not feedback_path.exists():
            logger.warning(f"[MetaLearner] 反馈文件不存在: {feedback_path}，改用面板标签训练")
            return self.fit_from_panel(panel)

        fb = pd.read_parquet(feedback_path)
        if len(fb) < 10:
            logger.warning(f"[MetaLearner] 反馈数据太少（{len(fb)}条），改用面板标签训练")
            return self.fit_from_panel(panel)

        all_dates = sorted(panel.index.get_level_values("as_of").unique())
        X_rows, y_rows = [], []

        for as_of_date in fb["as_of_date"].unique():
            as_of_ts = pd.Timestamp(as_of_date)
            # 找最近的面板期
            nearest = min(all_dates, key=lambda d: abs((d - as_of_ts).days))
            if abs((nearest - as_of_ts).days) > 45:
                continue

            xs = panel.xs(nearest, level="as_of")
            fb_day = fb[fb["as_of_date"] == as_of_date].set_index("ticker")
            common = xs.index.intersection(fb_day.index)
            if len(common) < 5:
                continue

            proxies = compute_agent_proxies(xs.loc[common], self.proxy_config).fillna(0.0)
            y = fb_day.loc[common, "actual_return_63d"]
            X_rows.append(proxies)
            y_rows.append(y)

        if not X_rows:
            logger.warning("[MetaLearner] 反馈数据与面板不匹配，改用面板标签训练")
            return self.fit_from_panel(panel)

        X_all = pd.concat(X_rows)
        y_all = pd.concat(y_rows)
        self._state.n_dates = len(X_rows)
        logger.info(f"[MetaLearner] 从反馈数据训练：{len(X_rows)} 期，{len(X_all)} 条记录")
        return self.fit(X_all, y_all)

    # ── 推理 ──────────────────────────────────────────────────────────────────

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """对 agent proxy 分数矩阵打分，返回最终 meta 分数."""
        if not self._fitted or self._model is None:
            # 未训练时等权融合
            return X.mean(axis=1).rename("meta_score")

        cols = self._state.agent_names
        X_ = X.reindex(columns=cols).fillna(0.0).values
        X_scaled = self._scaler.transform(X_)
        pred = self._model.predict(X_scaled)
        return pd.Series(pred, index=X.index, name="meta_score")

    def predict_from_panel(
        self,
        xs: pd.DataFrame,
        tickers: list[str] | None = None,
    ) -> pd.Series:
        """从单期截面面板数据计算 proxy → 打分."""
        if tickers:
            xs = xs.loc[xs.index.intersection(tickers)]
        proxies = compute_agent_proxies(xs, self.proxy_config)
        return self.predict(proxies)

    # ── 权重解读 ──────────────────────────────────────────────────────────────

    def get_weights(self) -> pd.Series:
        """返回各 agent 的当前有效权重."""
        if not self._fitted:
            return pd.Series(self._equal_weights, name="agent_weight")
        return pd.Series(self._state.coefficients, name="agent_weight")

    def summarize(self) -> dict:
        """返回 meta-learner 状态摘要."""
        weights = self.get_weights()
        top_agent = weights.abs().idxmax() if len(weights) > 0 else "N/A"
        return {
            "fitted":       self._fitted,
            "n_samples":    self._state.n_samples,
            "n_dates":      self._state.n_dates,
            "train_r2":     self._state.train_r2,
            "ridge_alpha":  self.ridge_alpha,
            "agent_weights": weights.to_dict(),
            "top_agent":    top_agent,
            "trained_at":   self._state.trained_at,
        }

    # ── 持久化 ────────────────────────────────────────────────────────────────

    def save(self, path: Path | None = None) -> Path:
        save_path = path or (self.save_dir / "meta_learner.pkl")
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "wb") as f:
            pickle.dump(self, f)

        # 可读的 meta JSON
        meta_path = save_path.with_suffix(".meta.json")
        meta_path.write_text(
            json.dumps(self.summarize(), ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        # 权重历史追加
        if self._state.weight_history:
            hist_path = save_path.parent / "weight_history.jsonl"
            with hist_path.open("a", encoding="utf-8") as f:
                for record in self._state.weight_history[-1:]:  # 只追加最新一条
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info(f"[MetaLearner] 已保存: {save_path}")
        return save_path

    @classmethod
    def load(cls, path: Path) -> "AgentMetaLearner":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise ValueError(f"{path} 不是 AgentMetaLearner 对象")
        logger.info(f"[MetaLearner] 已加载: {path}（训练于 {obj._state.trained_at}）")
        return obj

    @classmethod
    def load_or_create(cls, path: Path, **kwargs: Any) -> "AgentMetaLearner":
        """若模型文件存在则加载，否则创建新实例."""
        if path.exists():
            try:
                return cls.load(path)
            except Exception as e:
                logger.warning(f"[MetaLearner] 加载失败（{e}），创建新实例")
        return cls(**kwargs)
