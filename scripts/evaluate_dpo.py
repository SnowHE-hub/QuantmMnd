#!/usr/bin/env python3
"""scripts/evaluate_dpo.py — DPO 微调效果评估.

在 synthetic 测试集上对比：
  - Base Qwen2.5-1.5B（未微调，通过 Ollama 调用）
  - DPO 微调后（本地 LoRA 推理）

评估维度：
  1. grounding_score：reasoning 数字一致性（幻觉检测）
  2. reasoning 长度（字符数）
  3. JSON 格式成功率
  4. risk_warnings 数量（是否符合 ≥3 条）

输出 reports/dpo_evaluation.html（含对比表格和统计卡片）

用法::

    python scripts/evaluate_dpo.py                     # 仅评估 base 模型
    python scripts/evaluate_dpo.py --dpo-dir models/dpo_qwen  # 对比两者
    python scripts/evaluate_dpo.py --n 20 --provider ollama
"""

from __future__ import annotations

import argparse
import json
import re
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data" / "dpo"
MODELS_DIR   = PROJECT_ROOT / "models"
REPORTS_DIR  = PROJECT_ROOT / "reports"

DEFAULT_SYNTHETIC_DATA = DATA_DIR / "synthetic_100.jsonl"
DEFAULT_DPO_DIR        = MODELS_DIR / "dpo_qwen"
DEFAULT_REPORT         = REPORTS_DIR / "dpo_evaluation.html"


# ============================================================================
# 评估单条样本
# ============================================================================


def _try_parse_json(text: str):
    try:
        return json.loads(text.strip())
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return None


def eval_one(response_text: str, sample: dict) -> dict:
    """评估单条生成结果.

    Returns dict with keys:
      json_valid, n_rankings, has_portfolio_thesis, n_risk_warnings,
      grounding_score, response_len, elapsed_s
    """
    from quantmind.models.dpo_data_builder import _random_factor_row
    from quantmind.models.llm_reranker import RerankCandidate, RerankResult
    from quantmind.models.llm_reranker_eval import reasoning_grounding_score

    parsed = _try_parse_json(response_text)
    json_valid = parsed is not None and isinstance(parsed, dict)

    n_rankings = 0
    has_portfolio_thesis = False
    n_risk_warnings = 0
    grounding_score = float("nan")

    if json_valid:
        rankings = parsed.get("rankings", [])
        n_rankings = len(rankings) if isinstance(rankings, list) else 0
        thesis = parsed.get("portfolio_thesis", "")
        has_portfolio_thesis = bool(thesis and len(thesis) > 10)
        rw = parsed.get("risk_warnings", [])
        n_risk_warnings = len(rw) if isinstance(rw, list) else 0

        # 计算 grounding_score（取第一只股票）
        if n_rankings > 0 and isinstance(rankings[0], dict):
            first_ticker = rankings[0].get("ticker", "T001")
            first_reason = rankings[0].get("reason", "")
            # 创建虚拟 candidate（用 sample 中的因子数据）
            dummy_cand = RerankCandidate(
                ticker=first_ticker, lgbm_score=1.0, lgbm_rank=1,
                roe_ttm=_NP_RNG.uniform(0.05, 0.35),
                pe_ttm=_NP_RNG.uniform(10, 50),
                pb=_NP_RNG.uniform(1, 6),
                accruals=float(_NP_RNG.normal(0, 0.05)),
                distance_to_52w_high=float(_NP_RNG.uniform(-0.2, 0)),
                momentum_6m=float(_NP_RNG.normal(0.05, 0.15)),
                volatility_3m=_NP_RNG.uniform(0.15, 0.40),
            )
            dummy_result = RerankResult(rank=1, ticker=first_ticker, lgbm_rank=1,
                                        reason=first_reason)
            try:
                grounding_score = reasoning_grounding_score(dummy_result, dummy_cand)
            except Exception:
                grounding_score = float("nan")

    return {
        "json_valid": json_valid,
        "n_rankings": n_rankings,
        "has_portfolio_thesis": has_portfolio_thesis,
        "n_risk_warnings": n_risk_warnings,
        "grounding_score": grounding_score,
        "response_len": len(response_text),
    }


import numpy as _NP_RNG_module
_NP_RNG = _NP_RNG_module.random.default_rng(42)


# ============================================================================
# 运行评估
# ============================================================================


def run_evaluation(
    samples: list[dict],
    provider: str,
    model: str,
    label: str,
    use_local_dpo: bool = False,
    dpo_dir: str | Path | None = None,
) -> list[dict]:
    """对 samples 逐条推理并收集评估指标."""
    results = []

    if use_local_dpo:
        from quantmind.models.inference import load_dpo_model
        engine = load_dpo_model(dpo_dir or DEFAULT_DPO_DIR)

    for i, sample in enumerate(samples):
        prompt = sample.get("prompt", "")
        print(f"  [{label}] {i+1}/{len(samples)}", end="\r", flush=True)
        t0 = time.monotonic()

        try:
            if use_local_dpo:
                from quantmind.models.llm_reranker import _SYSTEM_PROMPT
                response_text = engine.generate(prompt, system=_SYSTEM_PROMPT, max_new_tokens=512)
            else:
                from quantmind.core.llm_router import LLMRouter
                router = LLMRouter()
                from quantmind.models.llm_reranker import _SYSTEM_PROMPT
                resp = router.chat(
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    provider=provider,
                    model=model,
                    max_tokens=512,
                    temperature=0.05,
                    fallback=None,
                )
                response_text = resp.content
        except Exception as e:  # noqa: BLE001
            response_text = ""
            print(f"\n    ⚠ [{label}] sample {i+1} error: {e}")

        elapsed = time.monotonic() - t0
        metrics = eval_one(response_text, sample)
        metrics["elapsed_s"] = round(elapsed, 2)
        metrics["sample_idx"] = i
        metrics["label"] = label
        results.append(metrics)

    print(f"\n  [{label}] 完成 {len(results)} 条")
    return results


# ============================================================================
# HTML 报告
# ============================================================================

_CSS = """
<style>
  * { box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', sans-serif;
    margin: 0; padding: 24px; background: #f8f9fa; color: #333;
  }
  h1 { color: #1a252f; border-bottom: 4px solid #2ecc71; padding-bottom: 12px; margin-top: 0; }
  h2 { color: #1a252f; margin-top: 28px; }
  .stats-row { display: flex; flex-wrap: wrap; gap: 16px; margin: 20px 0; }
  .stat-card {
    background: white; border-radius: 10px; padding: 16px 22px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08); min-width: 130px; text-align: center;
  }
  .stat-val { font-size: 1.9em; font-weight: 700; color: #2ecc71; line-height: 1.1; }
  .stat-val.blue  { color: #2980b9; }
  .stat-val.orange { color: #e67e22; }
  .stat-val.red   { color: #e74c3c; }
  .stat-val.purple { color: #8e44ad; }
  .stat-lbl { font-size: 0.82em; color: #777; margin-top: 4px; }
  table {
    width: 100%; border-collapse: collapse; background: white;
    border-radius: 10px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    margin-bottom: 24px;
  }
  thead tr { background: #1a252f; color: white; }
  th { padding: 12px 12px; font-size: 13px; text-align: center; }
  td { padding: 10px 12px; font-size: 12px; text-align: center; border-bottom: 1px solid #f0f0f0; }
  tr:hover td { background: #fafafa; }
  .good { color: #27ae60; font-weight: 600; }
  .bad  { color: #e74c3c; font-weight: 600; }
  footer { margin-top: 32px; color: #999; font-size: 0.8em; text-align: center; }
</style>
"""


def _agg(metrics: list[dict], key: str) -> float:
    vals = [m[key] for m in metrics if not (isinstance(m[key], float) and np.isnan(m[key]))]
    return float(np.mean(vals)) if vals else float("nan")


def build_html_report(
    base_metrics: list[dict],
    dpo_metrics: list[dict] | None,
    args_dict: dict,
) -> str:
    has_dpo = dpo_metrics is not None and len(dpo_metrics) > 0

    def fmt(v) -> str:
        if isinstance(v, float) and np.isnan(v):
            return "—"
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v)

    def pct(v) -> str:
        if isinstance(v, float) and np.isnan(v):
            return "—"
        return f"{v * 100:.1f}%"

    parts = [
        "<!DOCTYPE html><html><head>",
        '<meta charset="UTF-8">',
        "<title>QuantMind — DPO Evaluation</title>",
        _CSS,
        "</head><body>",
        "<h1>🧪 QuantMind — DPO 微调评估报告 (Phase 3.3)</h1>",
    ]

    # 配置信息
    parts.append(f"<p style='color:#666'>评估样本数: {len(base_metrics)} | "
                 f"Base 模型: {args_dict.get('provider')}/{args_dict.get('model')} | "
                 f"DPO 目录: {args_dict.get('dpo_dir', '—')}</p>")

    # 统计卡片（base）
    base_json_rate = _agg(base_metrics, "json_valid")
    base_grounding  = _agg(base_metrics, "grounding_score")
    base_rw_avg    = _agg(base_metrics, "n_risk_warnings")
    base_latency   = _agg(base_metrics, "elapsed_s")

    parts.append("<h2>📊 Base 模型评估指标</h2>")
    parts.append('<div class="stats-row">')

    def card(val, label, cls=""):
        return (f'<div class="stat-card"><div class="stat-val {cls}">{val}</div>'
                f'<div class="stat-lbl">{label}</div></div>')

    parts += [
        card(pct(base_json_rate), "JSON 格式成功率",
             "good" if base_json_rate > 0.8 else "bad"),
        card(fmt(base_grounding), "平均 grounding 分",
             "good" if base_grounding > 0.7 else "bad"),
        card(f"{base_rw_avg:.1f}", "平均风险提示数",
             "good" if base_rw_avg >= 3 else "orange"),
        card(f"{base_latency:.1f}s", "平均延迟", "blue"),
        card(f"{len(base_metrics)}", "评估样本数", "purple"),
    ]
    parts.append("</div>")

    # DPO 对比
    if has_dpo:
        dpo_json_rate = _agg(dpo_metrics, "json_valid")
        dpo_grounding  = _agg(dpo_metrics, "grounding_score")
        dpo_rw_avg    = _agg(dpo_metrics, "n_risk_warnings")
        dpo_latency   = _agg(dpo_metrics, "elapsed_s")

        parts.append("<h2>🚀 DPO 微调模型 vs Base 对比</h2>")
        parts.append('<div class="stats-row">')

        def delta_cls(dpo, base, higher_better=True):
            if np.isnan(dpo) or np.isnan(base):
                return ""
            return "good" if (dpo > base) == higher_better else "bad"

        parts += [
            card(f"{pct(dpo_json_rate)}<br><small style='color:#888'>base:{pct(base_json_rate)}</small>",
                 "JSON 成功率", delta_cls(dpo_json_rate, base_json_rate)),
            card(f"{fmt(dpo_grounding)}<br><small style='color:#888'>base:{fmt(base_grounding)}</small>",
                 "Grounding 分", delta_cls(dpo_grounding, base_grounding)),
            card(f"{dpo_rw_avg:.1f}<br><small style='color:#888'>base:{base_rw_avg:.1f}</small>",
                 "风险提示数", delta_cls(dpo_rw_avg, base_rw_avg)),
            card(f"{dpo_latency:.1f}s<br><small style='color:#888'>base:{base_latency:.1f}s</small>",
                 "延迟", "blue"),
        ]
        parts.append("</div>")

    # 明细表
    parts.append("<h2>📋 样本明细表</h2>")
    dpo_header = "<th>DPO JSON</th><th>DPO Grounding</th><th>DPO 风险数</th>" if has_dpo else ""
    parts.append(f"""
    <table><thead><tr>
      <th>#</th>
      <th>Base JSON</th><th>Base Grounding</th><th>Base 风险数</th><th>Base Len</th>
      {dpo_header}
    </tr></thead><tbody>
    """)

    for i, bm in enumerate(base_metrics[:50]):  # 最多显示 50 行
        dm = dpo_metrics[i] if has_dpo and i < len(dpo_metrics) else None

        def bool_cell(v): return f'<span class="good">✓</span>' if v else f'<span class="bad">✗</span>'
        def score_cell(v, thr=0.7):
            if np.isnan(v): return "—"
            cls = "good" if v >= thr else "bad"
            return f'<span class="{cls}">{v:.2f}</span>'

        dpo_cells = ""
        if dm:
            dpo_cells = (f"<td>{bool_cell(dm['json_valid'])}</td>"
                         f"<td>{score_cell(dm['grounding_score'])}</td>"
                         f"<td>{dm['n_risk_warnings']}</td>")

        parts.append(f"""
        <tr>
          <td>{i+1}</td>
          <td>{bool_cell(bm['json_valid'])}</td>
          <td>{score_cell(bm['grounding_score'])}</td>
          <td>{bm['n_risk_warnings']}</td>
          <td>{bm['response_len']}</td>
          {dpo_cells}
        </tr>""")

    parts.append("</tbody></table>")
    parts.append(f"<footer>QuantMind Phase 3.3 — DPO Evaluation | scripts/evaluate_dpo.py</footer>")
    parts.append("</body></html>")
    return "\n".join(parts)


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DPO 微调效果评估")
    p.add_argument("--data",     default=None, help="synthetic JSONL 路径")
    p.add_argument("--n",        type=int, default=10, help="评估样本数（默认10）")
    p.add_argument("--provider", default="ollama")
    p.add_argument("--model",    default="qwen2.5:7b")
    p.add_argument("--dpo-dir",  default=None, dest="dpo_dir",
                   help="DPO LoRA 目录（不指定则只评估 base）")
    p.add_argument("--report-out", default=None, dest="report_out")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"\n{'='*65}")
    print("QuantMind — Phase 3.3  DPO 评估")
    print(f"{'='*65}")

    # ── 准备数据 ─────────────────────────────────────────────────────────────
    data_path = Path(args.data) if args.data else DEFAULT_SYNTHETIC_DATA
    if not data_path.exists():
        print(f"  合成数据不存在，先生成 100 条 …")
        from quantmind.models.dpo_data_builder import build_synthetic_pairs, save_pairs
        pairs = build_synthetic_pairs(n=100)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        save_pairs(pairs, data_path)
        print(f"  ✅ 生成 → {data_path}")

    from quantmind.models.dpo_data_builder import load_pairs
    all_samples = load_pairs(data_path)
    samples = all_samples[: args.n]
    print(f"  评估样本: {len(samples)} 条（共 {len(all_samples)} 条）")

    # ── Base 模型评估 ─────────────────────────────────────────────────────────
    print(f"\n[1] 评估 Base 模型（{args.provider}/{args.model}）…")
    base_metrics = run_evaluation(
        samples, provider=args.provider, model=args.model,
        label="Base", use_local_dpo=False,
    )

    # ── DPO 模型评估（可选）──────────────────────────────────────────────────
    dpo_metrics = None
    dpo_dir = Path(args.dpo_dir) if args.dpo_dir else DEFAULT_DPO_DIR
    if dpo_dir.exists():
        print(f"\n[2] 评估 DPO 模型（{dpo_dir}）…")
        try:
            dpo_metrics = run_evaluation(
                samples, provider="local_dpo", model=str(dpo_dir),
                label="DPO", use_local_dpo=True, dpo_dir=dpo_dir,
            )
        except Exception as e:  # noqa: BLE001
            print(f"  ⚠ DPO 评估失败: {e}")
    else:
        print(f"\n[2] DPO 目录 {dpo_dir} 不存在，跳过对比评估")

    # ── 汇总打印 ───────────────────────────────────────────────────────────────
    def _agg(ms, k):
        vals = [m[k] for m in ms if not (isinstance(m[k], float) and np.isnan(m[k]))]
        return float(np.mean(vals)) if vals else float("nan")

    print(f"\n  Base 模型：JSON成功率={_agg(base_metrics,'json_valid'):.0%}  "
          f"Grounding={_agg(base_metrics,'grounding_score'):.3f}  "
          f"风险数均值={_agg(base_metrics,'n_risk_warnings'):.1f}")
    if dpo_metrics:
        print(f"  DPO  模型：JSON成功率={_agg(dpo_metrics,'json_valid'):.0%}  "
              f"Grounding={_agg(dpo_metrics,'grounding_score'):.3f}  "
              f"风险数均值={_agg(dpo_metrics,'n_risk_warnings'):.1f}")

    # ── 构建报告 ────────────────────────────────────────────────────────────
    report_out = Path(args.report_out) if args.report_out else DEFAULT_REPORT
    report_out = report_out if report_out.is_absolute() else PROJECT_ROOT / report_out
    report_out.parent.mkdir(parents=True, exist_ok=True)

    html = build_html_report(
        base_metrics=base_metrics,
        dpo_metrics=dpo_metrics,
        args_dict={
            "provider": args.provider, "model": args.model,
            "dpo_dir": str(dpo_dir), "n": len(samples),
        },
    )
    report_out.write_text(html, encoding="utf-8")

    print(f"\n{'='*65}")
    print(f"🎉  DPO 评估完成")
    print(f"{'='*65}")
    print(f"  报告 → {report_out}")


if __name__ == "__main__":
    main()
