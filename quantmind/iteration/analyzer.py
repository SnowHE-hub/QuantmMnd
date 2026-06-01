"""quantmind/iteration/analyzer.py — 30日模拟结果诊断.

读取 data/sim30d/ 的所有产物，输出结构化诊断报告 SimDiagnosis。

核心诊断维度
-----------
1. 因子 IC 分析     — 各维度 Spearman IC vs 1w/2w/21d/3m 四个 horizon
2. 选股漏斗         — System1/2/3 每层的命中率与过滤比
3. 行业归因         — 超额收益按行业分解（Top/Bottom 5）
4. 止损/持仓        — 亏损 > 5% 个股比例、超额持仓天数
5. 问题批次         — 找出 return_3m 表现最差的 N 个交易日
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]


# ─────────────────────────────────────────────────────────────────────────────
# SimDiagnosis 数据类
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SimDiagnosis:
    """结构化诊断报告."""

    # ── 基础摘要 ──────────────────────────────────────────────────────────────
    sim_dir:       str = ""
    n_days:        int = 0
    date_range:    str = ""

    # ── IR / 胜率 ─────────────────────────────────────────────────────────────
    ir_1w:         float = 0.0
    ir_2w:         float = 0.0
    ir_21d:        float = 0.0
    ir_3m:         float = 0.0
    win_rate_3m:   float = 0.0
    mean_return_3m: float = 0.0

    # ── 因子 IC（Spearman） ───────────────────────────────────────────────────
    ic_summary:    dict[str, dict[str, float]] = field(default_factory=dict)
    # eg. {"ic_1w": {"value_score": 0.03, "momentum_score": -0.05, ...}, ...}

    best_factor_3m:  str  = ""
    worst_factor_3m: str  = ""

    # ── 选股漏斗 ──────────────────────────────────────────────────────────────
    funnel:        dict[str, float] = field(default_factory=dict)
    # eg. {"s1_avg": 47.3, "s2_avg": 21.8, "s3_avg": 10.0, "s1_to_s3_ratio": 4.73}

    # ── 行业归因 ──────────────────────────────────────────────────────────────
    top_industries:    list[dict] = field(default_factory=list)
    bottom_industries: list[dict] = field(default_factory=list)

    # ── 持仓质量 ──────────────────────────────────────────────────────────────
    loss_pct_gt5:    float = 0.0   # 3m 亏损 > 5% 的股票占比
    drawdown_mean:   float = 0.0   # 平均最大回撤（若 positions 有）

    # ── 问题批次 ──────────────────────────────────────────────────────────────
    bad_batches:  list[dict] = field(default_factory=list)
    # eg. [{"date": "20251015", "return_3m_mean": -0.12, "n_stocks": 10}, ...]

    # ── 诊断结论 ──────────────────────────────────────────────────────────────
    issues:       list[str]  = field(default_factory=list)
    # eg. ["value_score IC_3m < 0，建议降权", "短期IR均负，不建议短线交易"]

    def to_dict(self) -> dict[str, Any]:
        import dataclasses
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SimDiagnosis":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_markdown(self) -> str:
        """生成 Markdown 格式诊断报告."""
        lines: list[str] = []
        lines.append("# 30日模拟诊断报告")
        lines.append(f"\n**模拟目录**: `{self.sim_dir}`  ")
        lines.append(f"**交易日数**: {self.n_days}  ")
        lines.append(f"**时间范围**: {self.date_range}")

        # IR
        lines.append("\n## 📊 收益摘要")
        lines.append("| Horizon | IR | 胜率 |")
        lines.append("|---------|-----|------|")
        lines.append(f"| 1周 | {self.ir_1w:+.3f} | — |")
        lines.append(f"| 2周 | {self.ir_2w:+.3f} | — |")
        lines.append(f"| 21日 | {self.ir_21d:+.3f} | — |")
        lines.append(f"| 3月 | {self.ir_3m:+.3f} | {self.win_rate_3m:.1%} |")
        lines.append(f"\n3月均值收益: **{self.mean_return_3m:+.2%}**")

        # IC
        lines.append("\n## 🔬 因子 IC（Spearman）")
        if self.ic_summary:
            horizons = list(self.ic_summary.keys())
            factors = list(next(iter(self.ic_summary.values())).keys())
            header = "| 因子 | " + " | ".join(horizons) + " |"
            sep    = "|" + "---|" * (len(horizons) + 1)
            lines.append(header)
            lines.append(sep)
            for f_name in factors:
                row = f"| {f_name} |"
                for h in horizons:
                    v = self.ic_summary[h].get(f_name, float("nan"))
                    row += f" {v:+.4f} |" if not np.isnan(v) else " N/A |"
                lines.append(row)
        lines.append(f"\n**最佳因子(3m)**: {self.best_factor_3m}  ")
        lines.append(f"**最差因子(3m)**: {self.worst_factor_3m}")

        # 漏斗
        lines.append("\n## 🔽 选股漏斗（日均）")
        for k, v in self.funnel.items():
            lines.append(f"- **{k}**: {v:.1f}")

        # 行业归因
        lines.append("\n## 🏭 行业归因（3m）")
        if self.top_industries:
            lines.append("### Top 表现行业")
            lines.append("| 行业 | 均值收益 | 上榜次数 |")
            lines.append("|------|----------|----------|")
            for ind in self.top_industries[:5]:
                lines.append(
                    f"| {ind.get('industry','')} "
                    f"| {ind.get('ret_3m_mean', 0):+.2%} "
                    f"| {ind.get('count', 0)} |"
                )
        if self.bottom_industries:
            lines.append("### Bottom 表现行业")
            lines.append("| 行业 | 均值收益 | 上榜次数 |")
            lines.append("|------|----------|----------|")
            for ind in self.bottom_industries[:5]:
                lines.append(
                    f"| {ind.get('industry','')} "
                    f"| {ind.get('ret_3m_mean', 0):+.2%} "
                    f"| {ind.get('count', 0)} |"
                )

        # 持仓质量
        lines.append("\n## 💼 持仓质量")
        lines.append(f"- 3m 亏损 > 5% 占比: **{self.loss_pct_gt5:.1%}**")
        if self.drawdown_mean:
            lines.append(f"- 平均最大回撤: **{self.drawdown_mean:.1%}**")

        # 问题批次
        if self.bad_batches:
            lines.append("\n## ⚠️ 问题批次（3m 收益最差）")
            lines.append("| 日期 | 3m均值收益 | 股票数 |")
            lines.append("|------|-----------|--------|")
            for b in self.bad_batches[:5]:
                lines.append(
                    f"| {b.get('date','')} "
                    f"| {b.get('return_3m_mean', 0):+.2%} "
                    f"| {b.get('n_stocks', 0)} |"
                )

        # 诊断结论
        if self.issues:
            lines.append("\n## 🩺 诊断结论")
            for issue in self.issues:
                lines.append(f"- {issue}")

        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# SimulationAnalyzer
# ─────────────────────────────────────────────────────────────────────────────

class SimulationAnalyzer:
    """读取 sim30d 产物，输出 SimDiagnosis.

    Parameters
    ----------
    sim_dir : Path | str | None
        模拟结果目录（含 stock_returns.parquet, positions.parquet,
        summary.json, daily/）。默认 data/sim30d/。
    """

    HORIZONS = ["1w", "2w", "21d", "3m"]
    FACTOR_COLS = [
        "value_score", "momentum_score", "quality_score",
        "technical_score", "composite_score", "lgbm_score",
    ]

    def __init__(self, sim_dir: Path | str | None = None) -> None:
        self._sim_dir = Path(sim_dir) if sim_dir else _ROOT / "data" / "sim30d"

    # ── 公开 API ──────────────────────────────────────────────────────────────

    def analyze(self) -> SimDiagnosis:
        """运行全量分析，返回 SimDiagnosis."""
        diag = SimDiagnosis(sim_dir=str(self._sim_dir))

        summary = self._load_summary()
        stock_returns = self._load_stock_returns()
        daily_files   = self._load_daily_files()

        # 基础摘要
        diag.n_days    = summary.get("n_days", 0)
        diag.date_range = summary.get("date_range", "")

        # IR / 胜率
        for horizon in ("1w", "2w", "21d", "3m"):
            h_data = summary.get("portfolio_returns", {}).get(horizon, {})
            setattr(diag, f"ir_{horizon}", h_data.get("ir", 0.0))
        diag.win_rate_3m    = summary.get("portfolio_returns", {}).get("3m", {}).get("win_rate", 0.0)
        diag.mean_return_3m = summary.get("portfolio_returns", {}).get("3m", {}).get("mean_return", 0.0)

        # 因子 IC
        if stock_returns is not None:
            diag.ic_summary      = self._analyze_factor_ic(stock_returns)
            best, worst          = self._best_worst_factor(diag.ic_summary)
            diag.best_factor_3m  = best
            diag.worst_factor_3m = worst

        # 漏斗
        if daily_files:
            diag.funnel = self._analyze_selection_funnel(daily_files)

        # 行业归因
        if stock_returns is not None and "industry" in stock_returns.columns:
            top_ind, bot_ind          = self._analyze_industry_attribution(stock_returns)
            diag.top_industries       = top_ind
            diag.bottom_industries    = bot_ind

        # 持仓质量
        if stock_returns is not None and "return_3m" in stock_returns.columns:
            diag.loss_pct_gt5 = self._analyze_stop_loss(stock_returns)

        # 问题批次
        if daily_files and stock_returns is not None:
            diag.bad_batches = self._identify_bad_batches(stock_returns, daily_files)

        # 诊断结论
        diag.issues = self._generate_issues(diag)

        return diag

    # ── 私有：数据加载 ────────────────────────────────────────────────────────

    def _load_summary(self) -> dict:
        p = self._sim_dir / "summary.json"
        if not p.exists():
            log.warning("[Analyzer] summary.json 不存在: %s", p)
            return {}
        try:
            with p.open(encoding="utf-8") as fh:
                raw = json.load(fh)
            # 兼容两种 summary 格式
            # 旧格式：直接有 ir_1w, win_rate_3m 等
            # 新格式：portfolio_returns 嵌套
            if "portfolio_returns" not in raw:
                raw = self._normalize_summary(raw)
            return raw
        except Exception as e:
            log.warning("[Analyzer] summary.json 读取失败: %s", e)
            return {}

    def _normalize_summary(self, raw: dict) -> dict:
        """兼容旧版 summary.json 格式 → 统一为嵌套格式."""
        pr: dict[str, dict] = {}
        for h in ("1w", "2w", "21d", "3m"):
            key = h.replace("d", "d")
            ir_key   = f"ir_{h}" if f"ir_{h}" in raw else None
            win_key  = f"win_rate_{h}" if f"win_rate_{h}" in raw else None
            mean_key = f"mean_return_{h}" if f"mean_return_{h}" in raw else None
            pr[h] = {
                "ir":          raw.get(ir_key or "",        0.0),
                "win_rate":    raw.get(win_key or "",       0.0),
                "mean_return": raw.get(mean_key or "",      0.0),
            }
        raw["portfolio_returns"] = pr
        if "n_days" not in raw:
            raw["n_days"] = raw.get("n", 0)
        return raw

    def _load_stock_returns(self) -> pd.DataFrame | None:
        p = self._sim_dir / "stock_returns.parquet"
        if not p.exists():
            log.warning("[Analyzer] stock_returns.parquet 不存在: %s", p)
            return None
        try:
            return pd.read_parquet(p)
        except Exception as e:
            log.warning("[Analyzer] stock_returns.parquet 读取失败: %s", e)
            return None

    def _load_daily_files(self) -> list[dict]:
        daily_dir = self._sim_dir / "daily"
        if not daily_dir.exists():
            return []
        files = sorted(daily_dir.glob("*.json"))
        results = []
        for fp in files:
            try:
                with fp.open(encoding="utf-8") as fh:
                    results.append(json.load(fh))
            except Exception as e:
                log.debug("[Analyzer] daily/%s 读取失败: %s", fp.name, e)
        return results

    # ── 私有：分析方法 ────────────────────────────────────────────────────────

    def _analyze_factor_ic(self, df: pd.DataFrame) -> dict[str, dict[str, float]]:
        """计算各因子对各 horizon 的 Spearman IC."""
        from scipy.stats import spearmanr

        factor_cols = [c for c in self.FACTOR_COLS if c in df.columns]
        result: dict[str, dict[str, float]] = {}

        for h in self.HORIZONS:
            ret_col = f"return_{h}"
            if ret_col not in df.columns:
                continue
            sub = df[[ret_col] + factor_cols].dropna()
            if len(sub) < 10:
                continue
            ic_row: dict[str, float] = {}
            for fc in factor_cols:
                try:
                    corr, _ = spearmanr(sub[fc], sub[ret_col])
                    ic_row[fc] = float(corr)
                except Exception:
                    ic_row[fc] = float("nan")
            result[f"ic_{h}"] = ic_row

        return result

    def _best_worst_factor(
        self, ic_summary: dict[str, dict[str, float]]
    ) -> tuple[str, str]:
        """返回 ic_3m 最高和最低的因子名."""
        ic3 = ic_summary.get("ic_3m", {})
        if not ic3:
            return "", ""
        valid = {k: v for k, v in ic3.items() if not np.isnan(v)}
        if not valid:
            return "", ""
        best  = max(valid, key=lambda k: valid[k])
        worst = min(valid, key=lambda k: valid[k])
        return best, worst

    def _analyze_selection_funnel(
        self, daily_files: list[dict]
    ) -> dict[str, float]:
        """统计 System1/2/3 各层平均候选数."""
        s1_counts, s3_counts = [], []
        for day in daily_files:
            s1 = day.get("system1_candidates", [])
            s3 = day.get("system3_final_list", [])
            if s1:
                s1_counts.append(len(s1))
            if s3:
                s3_counts.append(len(s3))

        s1_avg = float(np.mean(s1_counts)) if s1_counts else 0.0
        s3_avg = float(np.mean(s3_counts)) if s3_counts else 0.0
        ratio  = s1_avg / s3_avg if s3_avg > 0 else 0.0

        return {
            "s1_avg":         round(s1_avg, 1),
            "s3_avg":         round(s3_avg, 1),
            "s1_to_s3_ratio": round(ratio, 2),
        }

    def _analyze_industry_attribution(
        self, df: pd.DataFrame
    ) -> tuple[list[dict], list[dict]]:
        """按行业汇总 return_3m，返回 top5 / bottom5."""
        if "return_3m" not in df.columns or "industry" not in df.columns:
            return [], []

        grp = (
            df.dropna(subset=["return_3m", "industry"])
            .groupby("industry")["return_3m"]
            .agg(ret_3m_mean="mean", count="count")
            .reset_index()
            .sort_values("ret_3m_mean", ascending=False)
        )
        rows = grp.to_dict("records")
        return rows[:5], rows[-5:][::-1]

    def _analyze_stop_loss(self, df: pd.DataFrame) -> float:
        """计算 return_3m < -0.05 的股票占比."""
        if "return_3m" not in df.columns:
            return 0.0
        valid = df["return_3m"].dropna()
        if len(valid) == 0:
            return 0.0
        return float((valid < -0.05).mean())

    def _identify_bad_batches(
        self,
        df: pd.DataFrame,
        daily_files: list[dict],
        n: int = 5,
    ) -> list[dict]:
        """找出 return_3m 最差的 N 个交易日."""
        if "return_3m" not in df.columns or "date" not in df.columns:
            return []

        grp = (
            df.dropna(subset=["return_3m", "date"])
            .groupby("date")["return_3m"]
            .agg(return_3m_mean="mean", n_stocks="count")
            .reset_index()
            .sort_values("return_3m_mean")
            .head(n)
        )
        return grp.to_dict("records")

    def _generate_issues(self, diag: SimDiagnosis) -> list[str]:
        """基于诊断结果生成人类可读的结论列表."""
        issues: list[str] = []

        # IR
        if diag.ir_1w < 0 and diag.ir_2w < 0:
            issues.append("短期 IR（1w/2w）均为负，系统在短线交易场景下表现差，推荐持有 ≥ 3 个月")

        # 因子 IC
        ic3 = diag.ic_summary.get("ic_3m", {})
        for factor, ic_val in ic3.items():
            if np.isnan(ic_val):
                continue
            if ic_val < -0.03:
                issues.append(
                    f"{factor} 的 IC_3m = {ic_val:+.4f}（< -0.03），"
                    f"建议降低权重或反转方向"
                )
            elif ic_val > 0.08:
                issues.append(
                    f"{factor} 的 IC_3m = {ic_val:+.4f}（> 0.08），"
                    f"建议提升权重"
                )

        # 胜率
        if diag.win_rate_3m > 0 and diag.win_rate_3m < 0.5:
            issues.append(
                f"3m 胜率 {diag.win_rate_3m:.1%} 低于 50%，"
                f"建议检查选股过滤条件"
            )

        # 亏损
        if diag.loss_pct_gt5 > 0.3:
            issues.append(
                f"3m 亏损 > 5% 的股票占比 {diag.loss_pct_gt5:.1%}（> 30%），"
                f"建议加强下行风险管控"
            )

        # 漏斗
        if diag.funnel.get("s1_to_s3_ratio", 0) > 10:
            issues.append(
                f"选股漏斗压缩比过高（{diag.funnel['s1_to_s3_ratio']:.1f}×），"
                f"System3 最终候选池可能过于严苛"
            )

        if not issues:
            issues.append("未发现显著问题，系统运行正常")

        return issues
