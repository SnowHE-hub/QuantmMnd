#!/usr/bin/env python
"""scripts/run_iteration.py — 模拟迭代优化 CLI.

用法
----
# 分析当前 sim30d 结果
python scripts/run_iteration.py --mode analyze

# 生成参数调整建议（只打印，不写入）
python scripts/run_iteration.py --mode suggest

# 生成并应用建议到 strategy_config_v2.json
python scripts/run_iteration.py --mode apply

# 对比两轮结果（需指定 --baseline 和 --new）
python scripts/run_iteration.py --mode compare \\
    --baseline data/sim30d_v1 --new data/sim30d

# 对比 + 自动触发新一轮30日模拟
python scripts/run_iteration.py --mode compare --run-sim \\
    --baseline data/sim30d_v1 --new data/sim30d

# 输出为 Markdown 文件
python scripts/run_iteration.py --mode analyze --output report.md
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# ── 路径 ─────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from quantmind.iteration.analyzer   import SimulationAnalyzer
from quantmind.iteration.optimizer  import ParameterOptimizer
from quantmind.iteration.comparator import IterationComparator


# ─────────────────────────────────────────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QuantMind 模拟迭代优化 CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["analyze", "suggest", "apply", "compare"],
        required=True,
        help="运行模式",
    )
    parser.add_argument(
        "--sim-dir",
        default=str(_ROOT / "data" / "sim30d"),
        help="模拟结果目录（analyze/suggest/apply 模式使用）",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="基准轮目录（compare 模式必填）",
    )
    parser.add_argument(
        "--new",
        default=None,
        help="新一轮目录（compare 模式，默认与 --sim-dir 相同）",
    )
    parser.add_argument(
        "--config",
        default=str(_ROOT / "data" / "paper_trading" / "strategy_config_v2.json"),
        help="strategy_config_v2.json 路径",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="输出 Markdown 报告文件路径（可选）",
    )
    parser.add_argument(
        "--run-sim",
        action="store_true",
        help="compare 模式下，对比完成后自动运行30日模拟",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="apply 模式下，只打印不写文件",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="apply 模式下，只应用置信度最高的 N 条建议",
    )
    return parser.parse_args()


def _save_output(content: str, path: str | None) -> None:
    if path:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content, encoding="utf-8")
        print(f"\n📄 报告已保存到: {out_path}")


def _run_sim30d() -> int:
    """子进程运行30日模拟脚本."""
    sim_script = _ROOT / "scripts" / "simulate_30days.py"
    if not sim_script.exists():
        print(f"[ERROR] 模拟脚本不存在: {sim_script}")
        return 1
    print("\n🚀 启动 30 日模拟...")
    proc = subprocess.run(
        [sys.executable, str(sim_script)],
        cwd=str(_ROOT),
    )
    return proc.returncode


# ─────────────────────────────────────────────────────────────────────────────
# 各模式实现
# ─────────────────────────────────────────────────────────────────────────────

def mode_analyze(args: argparse.Namespace) -> None:
    sim_dir = Path(args.sim_dir)
    print(f"🔍 分析模拟结果: {sim_dir}")
    analyzer = SimulationAnalyzer(sim_dir)
    diag     = analyzer.analyze()

    report = diag.to_markdown()
    print("\n" + report)
    _save_output(report, args.output)

    # 退出码：有问题 → 1，无问题 → 0
    sys.exit(0 if "未发现显著问题" in (diag.issues[0] if diag.issues else "") else 0)


def mode_suggest(args: argparse.Namespace) -> None:
    sim_dir = Path(args.sim_dir)
    print(f"💡 生成参数建议（基于: {sim_dir}）")

    diag        = SimulationAnalyzer(sim_dir).analyze()
    optimizer   = ParameterOptimizer(config_path=args.config)
    suggestions = optimizer.generate_suggestions(diag)

    if not suggestions:
        print("✅ 没有需要调整的参数")
        return

    print(f"\n生成 {len(suggestions)} 条建议：")
    for i, s in enumerate(suggestions, 1):
        confidence_bar = "█" * int(s.confidence * 10)
        print(
            f"\n{i}. [{s.category.upper()}] {s.param_path}\n"
            f"   {s.old_value!r} → {s.new_value!r}\n"
            f"   置信度: {s.confidence:.0%} [{confidence_bar:<10}]\n"
            f"   原因: {s.reason}"
        )

    if args.output:
        lines = [f"# 参数调整建议（{len(suggestions)} 条）\n"]
        for s in suggestions:
            lines.append(f"## {s.param_path}")
            lines.append(f"- **旧值**: `{s.old_value}`")
            lines.append(f"- **新值**: `{s.new_value}`")
            lines.append(f"- **置信度**: {s.confidence:.0%}")
            lines.append(f"- **原因**: {s.reason}\n")
        _save_output("\n".join(lines), args.output)


def mode_apply(args: argparse.Namespace) -> None:
    sim_dir = Path(args.sim_dir)
    print(f"⚙️  应用参数建议（{'DRY-RUN' if args.dry_run else '写入'}）")

    diag        = SimulationAnalyzer(sim_dir).analyze()
    optimizer   = ParameterOptimizer(config_path=args.config)
    suggestions = optimizer.generate_suggestions(diag)

    if not suggestions:
        print("✅ 没有需要调整的参数")
        return

    # 按置信度排序，只应用 top-N
    suggestions.sort(key=lambda s: s.confidence, reverse=True)
    if args.top_n:
        suggestions = suggestions[: args.top_n]
        print(f"📌 只应用置信度最高的 {args.top_n} 条建议")

    print(f"\n即将应用 {len(suggestions)} 条建议：")
    for s in suggestions:
        print(f"  {s.param_path}: {s.old_value!r} → {s.new_value!r}")

    updated = optimizer.apply_suggestions(
        suggestions,
        config_path=args.config,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        print("\n[DRY-RUN] 配置更新预览（未写入）：")
        weights = (
            updated.get("system2_updates", {})
            .get("weights_calibrated", {})
        )
        if weights:
            print("  weights_calibrated:", json.dumps(weights, ensure_ascii=False))
    else:
        print(f"\n✅ 配置已写入: {args.config}")

    if args.output:
        lines = ["# 已应用的参数调整\n"]
        for s in suggestions:
            lines.append(f"- `{s.param_path}`: `{s.old_value}` → `{s.new_value}`")
        _save_output("\n".join(lines), args.output)


def mode_compare(args: argparse.Namespace) -> None:
    baseline_dir = Path(args.baseline) if args.baseline else Path(args.sim_dir)
    new_dir      = Path(args.new)      if args.new      else Path(args.sim_dir)

    if not args.baseline and not args.new:
        print("[ERROR] compare 模式需要 --baseline 和/或 --new 参数")
        sys.exit(1)

    print(f"🔄 对比两轮结果")
    print(f"  基准轮: {baseline_dir}")
    print(f"  新一轮: {new_dir}")

    comparator = IterationComparator(baseline_dir)
    report     = comparator.compare(new_dir, baseline_dir)

    md = report.to_markdown()
    print("\n" + md)
    _save_output(md, args.output)

    if report.overall_improved:
        print("\n✅ 整体改善，建议继续迭代")
    else:
        print("\n⚠️  未见显著改善，建议检查参数调整方向")

    if args.run_sim:
        rc = _run_sim30d()
        if rc != 0:
            print(f"[ERROR] 模拟运行失败 (rc={rc})")
            sys.exit(rc)


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    dispatch = {
        "analyze": mode_analyze,
        "suggest": mode_suggest,
        "apply":   mode_apply,
        "compare": mode_compare,
    }
    dispatch[args.mode](args)


if __name__ == "__main__":
    main()
