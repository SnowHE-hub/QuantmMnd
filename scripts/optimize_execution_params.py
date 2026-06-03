"""
scripts/optimize_execution_params.py — 执行参数网格搜索研究入口.

在 80 笔历史推荐上回放 576 组参数（6×6×4×4），找出 Pareto 最优。

输出：
  data/execution_research/grid_results.parquet     (576 行 × ~15 列)
  data/execution_research/pareto_frontier.parquet
  data/execution_research/best_params.json
  docs/execution_optimization_report.md

用法：
  conda run -n quantmind python scripts/optimize_execution_params.py
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.db.postgres import get_pg_engine
from quantmind.execution import (
    HistoricalReplayEngine,
    ExecutionParamOptimizer,
    load_historical_recommendations,
    preload_price_history,
    ReplayParams,
)


# ── 死扛基准（来自 E3 修正后的数据）──────────────────────────────────────────
HOLD_BASELINE = {
    "cum_return": 0.0522,
    "maxdd":      -0.0085,
    "sharpe":     3.67,
    "win_rate":   0.488,
    "avg_holding_days": 63,
}

# 当前生产参数（E3 默认值）
CURRENT_PARAMS = {
    "stop_loss":     -0.10,
    "target_price":   0.20,
    "trailing_stop": -0.15,
    "holding_days":   63,
}


def _format_param_value(v):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:+.2%}" if abs(v) < 1 else f"{v:.2f}"
    return str(v)


def _params_to_str(d: dict) -> str:
    sl = _format_param_value(d.get("stop_loss"))
    tg = _format_param_value(d.get("target_price"))
    tr = _format_param_value(d.get("trailing_stop"))
    hd = d.get("holding_days", "—")
    return f"sl={sl}, tg={tg}, trl={tr}, hd={hd}d"


def main():
    print("=" * 70)
    print("E3.5 执行参数网格搜索研究")
    print("=" * 70)

    out_dir = ROOT / "data" / "execution_research"
    out_dir.mkdir(parents=True, exist_ok=True)
    docs_dir = ROOT / "docs"

    # 1. 加载历史推荐
    eng = get_pg_engine()
    print("\n[1/4] 加载历史推荐 + 价格数据 ...")
    t0 = time.time()
    recs = load_historical_recommendations(eng)
    print(f"  推荐数: {len(recs)}")
    prices = preload_price_history(eng, recs, max_holding_days=200)
    n_bars = sum(len(df) for df in prices.values())
    print(f"  价格数据: {len(prices)} 个 ticker, {n_bars:,} 根 K 线")
    print(f"  加载耗时: {time.time() - t0:.1f}s")

    # 2. 跑网格搜索
    replay = HistoricalReplayEngine(recs, prices)
    opt = ExecutionParamOptimizer()
    n_combos = (len(opt.grid["stop_loss"]) * len(opt.grid["target_price"]) *
                len(opt.grid["trailing_stop"]) * len(opt.grid["holding_days"]))

    print(f"\n[2/4] 网格搜索 {n_combos} 组合 × {len(recs)} 订单 = "
          f"{n_combos * len(recs):,} 次回放 ...")
    t0 = time.time()
    def _cb(done, total):
        print(f"  进度: {done}/{total}")
    results = opt.run_grid_search(replay, progress_cb=_cb)
    print(f"  网格搜索耗时: {time.time() - t0:.1f}s")
    print(f"  结果 shape: {results.shape}")

    # 3. Pareto 前沿 + 单目标最优
    print("\n[3/4] 分析 Pareto 前沿 + 单目标最优 ...")
    pareto = opt.find_pareto_optimal(results)
    print(f"  Pareto 前沿: {len(pareto)} 个组合")

    best_sharpe = opt.recommend_best_params(results, criteria="sharpe")
    best_return = opt.recommend_best_params(results, criteria="return")
    best_winrate = opt.recommend_best_params(results, criteria="win_rate")
    # 带约束：MaxDD 不能比死扛差超过 50bp
    best_safe = opt.recommend_best_params(
        results, criteria="return",
        constraints={"maxdd_min": HOLD_BASELINE["maxdd"]},
    )

    # 击败基准统计
    beat_stats = opt.beat_baseline(
        results, HOLD_BASELINE["cum_return"], HOLD_BASELINE["maxdd"])
    print(f"  总组合: {beat_stats['total']}")
    print(f"  收益击败死扛: {beat_stats['beat_return']} "
          f"({beat_stats['beat_return_pct']*100:.1f}%)")
    print(f"  MaxDD 击败死扛: {beat_stats['beat_maxdd']} "
          f"({beat_stats['beat_maxdd_pct']*100:.1f}%)")
    print(f"  两者都击败: {beat_stats['beat_both']} "
          f"({beat_stats['beat_both_pct']*100:.1f}%)")

    # 4. 持久化
    print("\n[4/4] 保存结果 ...")
    results.to_parquet(out_dir / "grid_results.parquet", index=False)
    pareto.to_parquet(out_dir / "pareto_frontier.parquet", index=False)

    best_params_json = {
        "generated_at":     datetime.now().isoformat(timespec="seconds"),
        "n_combos":         int(beat_stats["total"]),
        "n_orders":         len(recs),
        "current_params":   CURRENT_PARAMS,
        "hold_baseline":    HOLD_BASELINE,
        "beat_baseline":    beat_stats,
        "best_sharpe":      best_sharpe,
        "best_return":      best_return,
        "best_winrate":     best_winrate,
        "best_safe_return": best_safe,   # 在 MaxDD ≥ 死扛约束下的收益最优
    }
    # 把 numpy 类型转 native
    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_clean(v) for v in o]
        if hasattr(o, "item") and callable(o.item):
            try:
                return o.item()
            except Exception:
                return o
        return o

    best_params_json = _clean(best_params_json)
    (out_dir / "best_params.json").write_text(
        json.dumps(best_params_json, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"  ✓ {out_dir / 'grid_results.parquet'}")
    print(f"  ✓ {out_dir / 'pareto_frontier.parquet'}")
    print(f"  ✓ {out_dir / 'best_params.json'}")

    # 5. 生成 Markdown 报告
    report_lines = [
        "# 执行参数优化研究报告（E3.5）",
        "",
        f"生成时间：{best_params_json['generated_at']}",
        f"样本：{len(recs)} 笔历史推荐",
        f"网格：{beat_stats['total']} 组参数（stop_loss × target × trailing × holding_days）",
        "",
        "## 死扛基准（参照点）",
        "",
        f"- 累计收益: **{HOLD_BASELINE['cum_return']*100:+.2f}%**",
        f"- MaxDD:   **{HOLD_BASELINE['maxdd']*100:.2f}%**",
        f"- Sharpe:  **{HOLD_BASELINE['sharpe']:.2f}**",
        f"- 胜率:    **{HOLD_BASELINE['win_rate']*100:.1f}%**",
        "",
        "## 网格搜索结果总览",
        "",
        f"- 总组合: {beat_stats['total']}",
        f"- 累计收益 > 死扛（>+5.22%）: **{beat_stats['beat_return']} "
        f"({beat_stats['beat_return_pct']*100:.1f}%)**",
        f"- MaxDD 优于死扛（>-0.85%）: **{beat_stats['beat_maxdd']} "
        f"({beat_stats['beat_maxdd_pct']*100:.1f}%)**",
        f"- 两者都击败死扛: **{beat_stats['beat_both']} "
        f"({beat_stats['beat_both_pct']*100:.1f}%)**",
        "",
        "## Top 10 by Sharpe",
        "",
    ]

    cols_to_show = ["stop_loss", "target_price", "trailing_stop", "holding_days",
                    "cum_return", "maxdd", "sharpe", "win_rate",
                    "avg_holding_days", "n_stop_loss", "n_target_hit"]
    top10_sharpe = results.nlargest(10, "sharpe")[cols_to_show].copy()
    report_lines.append(top10_sharpe.to_markdown(index=False, floatfmt=".4f"))

    report_lines.extend([
        "",
        "## Top 10 by 累计收益",
        "",
        results.nlargest(10, "cum_return")[cols_to_show].to_markdown(
            index=False, floatfmt=".4f"),
        "",
        "## Pareto 前沿（收益 vs MaxDD）",
        "",
        f"共 {len(pareto)} 个不被支配的组合：",
        "",
        pareto[cols_to_show].head(20).to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 推荐参数",
        "",
        f"### 按 Sharpe 最优 (sharpe={best_sharpe.get('sharpe', 0):.2f})",
        f"- 参数: {_params_to_str(best_sharpe)}",
        f"- 累计收益: {best_sharpe.get('cum_return', 0)*100:+.2f}%",
        f"- MaxDD:    {best_sharpe.get('maxdd', 0)*100:.2f}%",
        f"- 胜率:     {best_sharpe.get('win_rate', 0)*100:.1f}%",
        f"- 退出分布: 止损 {best_sharpe.get('n_stop_loss', 0)} / "
        f"止盈 {best_sharpe.get('n_target_hit', 0)} / "
        f"追踪 {best_sharpe.get('n_trailing_stop', 0)} / "
        f"到期 {best_sharpe.get('n_time_expired', 0)}",
        "",
        f"### 按累计收益最优 (return={best_return.get('cum_return', 0)*100:+.2f}%)",
        f"- 参数: {_params_to_str(best_return)}",
        f"- MaxDD: {best_return.get('maxdd', 0)*100:.2f}%",
        f"- Sharpe: {best_return.get('sharpe', 0):.2f}",
        "",
        f"### 安全收益最优（约束: MaxDD ≥ 死扛 {HOLD_BASELINE['maxdd']*100:.2f}%）",
    ])
    if best_safe:
        report_lines.extend([
            f"- 参数: {_params_to_str(best_safe)}",
            f"- 累计收益: {best_safe.get('cum_return', 0)*100:+.2f}%",
            f"- MaxDD:    {best_safe.get('maxdd', 0)*100:.2f}%",
        ])
    else:
        report_lines.append("- 约束下无可行参数")

    report_lines.extend([
        "",
        "### 当前生产参数",
        f"- {_params_to_str(CURRENT_PARAMS)}",
        "",
        "## 结论",
        "",
    ])

    # 自动结论
    if best_sharpe.get("cum_return", 0) > HOLD_BASELINE["cum_return"]:
        report_lines.append(
            f"✅ **找到优于死扛的参数组合**：按 Sharpe 最优的组合累计收益 "
            f"{best_sharpe['cum_return']*100:+.2f}% 超过死扛 "
            f"{HOLD_BASELINE['cum_return']*100:+.2f}% "
            f"({(best_sharpe['cum_return'] - HOLD_BASELINE['cum_return'])*100:+.2f}pp)。"
        )
    else:
        report_lines.append(
            f"⚠️ 即使 Sharpe 最优的参数 ({best_sharpe['cum_return']*100:+.2f}%) "
            f"仍未超过死扛 ({HOLD_BASELINE['cum_return']*100:+.2f}%)。"
            f"建议扩大网格或考虑动态参数。"
        )
    if best_safe and best_safe.get("cum_return", 0) > HOLD_BASELINE["cum_return"]:
        report_lines.append("")
        report_lines.append(
            f"✅ 在 MaxDD 不变差的约束下，仍能找到收益 "
            f"{best_safe['cum_return']*100:+.2f}% 的参数组合。"
        )

    report_lines.extend([
        "",
        "## 数据文件",
        "",
        "- `data/execution_research/grid_results.parquet` — 完整 576 组合结果",
        "- `data/execution_research/pareto_frontier.parquet` — Pareto 前沿子集",
        "- `data/execution_research/best_params.json` — 推荐参数 + 统计",
        "",
    ])

    report_path = docs_dir / "execution_optimization_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"  ✓ {report_path}")

    # 终端汇总
    print("\n" + "=" * 70)
    print("汇总")
    print("=" * 70)
    print(f"死扛基准:        cum={HOLD_BASELINE['cum_return']*100:+.2f}%, "
          f"dd={HOLD_BASELINE['maxdd']*100:.2f}%, sharpe={HOLD_BASELINE['sharpe']:.2f}")
    print(f"按 Sharpe 最优:  cum={best_sharpe['cum_return']*100:+.2f}%, "
          f"dd={best_sharpe['maxdd']*100:.2f}%, sharpe={best_sharpe['sharpe']:.2f}")
    print(f"  参数: {_params_to_str(best_sharpe)}")
    print(f"按收益最优:      cum={best_return['cum_return']*100:+.2f}%, "
          f"dd={best_return['maxdd']*100:.2f}%, sharpe={best_return['sharpe']:.2f}")
    print(f"  参数: {_params_to_str(best_return)}")


if __name__ == "__main__":
    main()
