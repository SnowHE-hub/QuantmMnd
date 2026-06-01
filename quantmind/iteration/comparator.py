"""quantmind/iteration/comparator.py — 对比两轮模拟迭代结果.

IterationComparator
  .compare(baseline_dir, new_dir) → ComparisonReport
  ComparisonReport.to_html_report() → HTML 字符串

对比维度
--------
- IR 变化（各 horizon）
- 胜率变化
- 因子 IC 改善量
- 持仓质量改善（亏损率）
- 行业集中度变化
- 问题批次数量变化
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from quantmind.iteration.analyzer import SimulationAnalyzer, SimDiagnosis

log = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]


# ─────────────────────────────────────────────────────────────────────────────
# ComparisonReport
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ComparisonReport:
    """两轮迭代对比报告."""

    baseline_dir: str = ""
    new_dir:      str = ""

    # IR 对比
    ir_delta: dict[str, float] = field(default_factory=dict)
    # eg. {"1w": +0.05, "2w": -0.02, "21d": +0.10, "3m": +0.08}

    # 胜率对比
    win_rate_3m_baseline: float = 0.0
    win_rate_3m_new:      float = 0.0
    win_rate_delta:       float = 0.0

    # 因子 IC 对比（3m）
    ic_delta_3m: dict[str, float] = field(default_factory=dict)
    # eg. {"value_score": +0.02, "quality_score": -0.01, ...}

    # 持仓质量
    loss_pct_baseline: float = 0.0
    loss_pct_new:      float = 0.0
    loss_pct_delta:    float = 0.0

    # 行业集中度
    industry_hhi_baseline: float = 0.0
    industry_hhi_new:      float = 0.0

    # 问题批次
    n_bad_batches_baseline: int = 0
    n_bad_batches_new:      int = 0

    # 综合判断
    overall_improved: bool      = False
    summary_lines:    list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        import dataclasses
        return dataclasses.asdict(self)

    def to_markdown(self) -> str:
        """生成 Markdown 格式对比报告."""
        lines: list[str] = []
        lines.append("# 迭代对比报告")
        lines.append(f"\n**基准轮**: `{self.baseline_dir}`  ")
        lines.append(f"**新一轮**: `{self.new_dir}`")

        # IR
        lines.append("\n## 📊 IR 变化")
        lines.append("| Horizon | 基准 | 新轮 | 变化 | 判断 |")
        lines.append("|---------|------|------|------|------|")
        for h in ("1w", "2w", "21d", "3m"):
            delta = self.ir_delta.get(h, 0.0)
            emoji = "🟢" if delta > 0 else ("🔴" if delta < 0 else "⚪")
            # 反推基准和新轮值
            base_val = getattr(self, f"_ir_baseline_{h}", None)
            new_val  = getattr(self, f"_ir_new_{h}", None)
            base_str = f"{base_val:+.3f}" if base_val is not None else "—"
            new_str  = f"{new_val:+.3f}"  if new_val  is not None else "—"
            lines.append(
                f"| {h} | {base_str} | {new_str} | {delta:+.3f} | {emoji} |"
            )

        # 胜率
        lines.append("\n## 🎯 胜率变化（3m）")
        wr_emoji = "🟢" if self.win_rate_delta > 0 else "🔴"
        lines.append(
            f"基准: **{self.win_rate_3m_baseline:.1%}**  →  "
            f"新轮: **{self.win_rate_3m_new:.1%}**  "
            f"({self.win_rate_delta:+.1%}) {wr_emoji}"
        )

        # 因子 IC
        lines.append("\n## 🔬 因子 IC 变化（3m Spearman）")
        if self.ic_delta_3m:
            lines.append("| 因子 | 变化 | 判断 |")
            lines.append("|------|------|------|")
            for fc, delta in self.ic_delta_3m.items():
                emoji = "🟢" if delta > 0.01 else ("🔴" if delta < -0.01 else "⚪")
                lines.append(f"| {fc} | {delta:+.4f} | {emoji} |")

        # 持仓质量
        lines.append("\n## 💼 持仓质量")
        loss_emoji = "🟢" if self.loss_pct_delta < 0 else "🔴"
        lines.append(
            f"亏损>5% 占比: {self.loss_pct_baseline:.1%} → {self.loss_pct_new:.1%} "
            f"({self.loss_pct_delta:+.1%}) {loss_emoji}"
        )

        # 行业
        if self.industry_hhi_baseline or self.industry_hhi_new:
            lines.append("\n## 🏭 行业集中度（HHI）")
            hhi_emoji = "🟢" if self.industry_hhi_new < self.industry_hhi_baseline else "🔴"
            lines.append(
                f"HHI: {self.industry_hhi_baseline:.4f} → {self.industry_hhi_new:.4f} "
                f"{hhi_emoji}"
            )

        # 问题批次
        lines.append("\n## ⚠️ 问题批次数")
        bad_emoji = "🟢" if self.n_bad_batches_new < self.n_bad_batches_baseline else "🔴"
        lines.append(
            f"基准: {self.n_bad_batches_baseline} 批 → 新轮: {self.n_bad_batches_new} 批 "
            f"{bad_emoji}"
        )

        # 综合
        lines.append("\n## 🏆 综合判断")
        lines.append("**整体改善**" if self.overall_improved else "**整体未见明显改善**")
        if self.summary_lines:
            for s in self.summary_lines:
                lines.append(f"- {s}")

        return "\n".join(lines)

    def to_html_report(self) -> str:
        """生成简洁 HTML 格式对比报告（可内嵌 Streamlit st.components.v1.html）."""
        md = self.to_markdown()
        # 简易 Markdown → HTML（仅处理常见标记）
        import re
        html_lines: list[str] = ["<html><body style='font-family:sans-serif;padding:20px'>"]
        for line in md.split("\n"):
            line = line.rstrip()
            if line.startswith("# "):
                html_lines.append(f"<h1>{line[2:]}</h1>")
            elif line.startswith("## "):
                html_lines.append(f"<h2>{line[3:]}</h2>")
            elif line.startswith("### "):
                html_lines.append(f"<h3>{line[4:]}</h3>")
            elif line.startswith("- "):
                html_lines.append(f"<li>{line[2:]}</li>")
            elif line.startswith("|"):
                # 表格行
                cells = [c.strip() for c in line.split("|")[1:-1]]
                if all(set(c) <= set("-: ") for c in cells):
                    continue  # 分隔行
                row_html = "".join(f"<td style='border:1px solid #ccc;padding:4px'>{c}</td>" for c in cells)
                html_lines.append(f"<tr>{row_html}</tr>")
            elif line.startswith("**") and line.endswith("**"):
                html_lines.append(f"<p><strong>{line[2:-2]}</strong></p>")
            elif line:
                # 粗体替换
                line = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", line)
                html_lines.append(f"<p>{line}</p>")
        html_lines.append("</body></html>")
        return "\n".join(html_lines)


# ─────────────────────────────────────────────────────────────────────────────
# IterationComparator
# ─────────────────────────────────────────────────────────────────────────────

class IterationComparator:
    """对比两个 sim30d 结果目录，生成 ComparisonReport.

    Parameters
    ----------
    baseline_dir : Path | str | None
        基准轮结果目录，默认 data/sim30d/。
    """

    def __init__(self, baseline_dir: Path | str | None = None) -> None:
        self._baseline_dir = (
            Path(baseline_dir) if baseline_dir else _ROOT / "data" / "sim30d"
        )

    def compare(
        self,
        new_dir: Path | str,
        baseline_dir: Path | str | None = None,
    ) -> ComparisonReport:
        """运行对比分析.

        Parameters
        ----------
        new_dir      : 新一轮模拟结果目录
        baseline_dir : 若指定，覆盖构造时的 baseline_dir
        """
        bd = Path(baseline_dir) if baseline_dir else self._baseline_dir
        nd = Path(new_dir)

        baseline_diag = SimulationAnalyzer(bd).analyze()
        new_diag      = SimulationAnalyzer(nd).analyze()

        return self._build_report(baseline_diag, new_diag, bd, nd)

    def compare_diagnoses(
        self,
        baseline_diag: SimDiagnosis,
        new_diag:      SimDiagnosis,
        baseline_dir:  Path | str = "",
        new_dir:       Path | str = "",
    ) -> ComparisonReport:
        """直接用两个已有的 SimDiagnosis 对象做对比（不重新读磁盘）."""
        return self._build_report(
            baseline_diag, new_diag,
            Path(baseline_dir), Path(new_dir),
        )

    # ── 私有 ──────────────────────────────────────────────────────────────────

    def _build_report(
        self,
        b: SimDiagnosis,
        n: SimDiagnosis,
        bd: Path,
        nd: Path,
    ) -> ComparisonReport:
        report = ComparisonReport(
            baseline_dir = str(bd),
            new_dir      = str(nd),
        )

        # IR 变化
        for h in ("1w", "2w", "21d", "3m"):
            b_ir = getattr(b, f"ir_{h}", 0.0)
            n_ir = getattr(n, f"ir_{h}", 0.0)
            report.ir_delta[h] = round(n_ir - b_ir, 4)
            # 保存基准/新轮原始值供 to_markdown 使用
            object.__setattr__(report, f"_ir_baseline_{h}", b_ir)
            object.__setattr__(report, f"_ir_new_{h}",      n_ir)

        # 胜率
        report.win_rate_3m_baseline = b.win_rate_3m
        report.win_rate_3m_new      = n.win_rate_3m
        report.win_rate_delta       = round(n.win_rate_3m - b.win_rate_3m, 4)

        # 因子 IC 对比（3m）
        b_ic3 = b.ic_summary.get("ic_3m", {})
        n_ic3 = n.ic_summary.get("ic_3m", {})
        all_factors = set(b_ic3) | set(n_ic3)
        for fc in all_factors:
            bv = b_ic3.get(fc, float("nan"))
            nv = n_ic3.get(fc, float("nan"))
            if not (np.isnan(bv) or np.isnan(nv)):
                report.ic_delta_3m[fc] = round(nv - bv, 4)

        # 持仓质量
        report.loss_pct_baseline = b.loss_pct_gt5
        report.loss_pct_new      = n.loss_pct_gt5
        report.loss_pct_delta    = round(n.loss_pct_gt5 - b.loss_pct_gt5, 4)

        # 行业集中度（HHI）
        report.industry_hhi_baseline = self._compute_hhi(b.top_industries + b.bottom_industries)
        report.industry_hhi_new      = self._compute_hhi(n.top_industries + n.bottom_industries)

        # 问题批次
        report.n_bad_batches_baseline = len(b.bad_batches)
        report.n_bad_batches_new      = len(n.bad_batches)

        # 综合判断
        report.overall_improved, report.summary_lines = self._judge(report)

        return report

    @staticmethod
    def _compute_hhi(industries: list[dict]) -> float:
        """赫芬达尔-赫希曼指数（行业集中度），基于 count 字段."""
        counts = [ind.get("count", 0) for ind in industries]
        total  = sum(counts)
        if total == 0:
            return 0.0
        return float(sum((c / total) ** 2 for c in counts))

    @staticmethod
    def _judge(report: ComparisonReport) -> tuple[bool, list[str]]:
        """综合多个指标，判断新轮是否优于基准."""
        score = 0
        lines: list[str] = []

        # IR_3m 改善 +2 分
        delta_3m = report.ir_delta.get("3m", 0)
        if delta_3m > 0.05:
            score += 2
            lines.append(f"IR_3m 明显提升 ({delta_3m:+.3f})")
        elif delta_3m > 0:
            score += 1
            lines.append(f"IR_3m 小幅提升 ({delta_3m:+.3f})")
        else:
            score -= 1
            lines.append(f"IR_3m 下降 ({delta_3m:+.3f})")

        # 胜率 +1 分
        if report.win_rate_delta > 0.02:
            score += 1
            lines.append(f"3m 胜率提升 ({report.win_rate_delta:+.1%})")
        elif report.win_rate_delta < -0.02:
            score -= 1
            lines.append(f"3m 胜率下降 ({report.win_rate_delta:+.1%})")

        # 亏损率 +1 分
        if report.loss_pct_delta < -0.05:
            score += 1
            lines.append(f"亏损率降低 ({report.loss_pct_delta:+.1%})")
        elif report.loss_pct_delta > 0.05:
            score -= 1
            lines.append(f"亏损率上升 ({report.loss_pct_delta:+.1%})")

        # 问题批次 +1 分
        if report.n_bad_batches_new < report.n_bad_batches_baseline:
            score += 1
            lines.append("问题批次减少")
        elif report.n_bad_batches_new > report.n_bad_batches_baseline:
            score -= 1
            lines.append("问题批次增加")

        improved = score > 0
        if improved:
            lines.insert(0, f"综合得分 {score:+d}，新轮优于基准 ✅")
        else:
            lines.insert(0, f"综合得分 {score:+d}，新轮未明显改善 ⚠️")

        return improved, lines
