#!/usr/bin/env python3
"""汇总三版 LGBM 训练 metrics + factor IC，生成 model_comparison.md。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--metrics",
        type=Path,
        nargs=3,
        default=[
            Path("reports/model_training/metrics_lgbm_v1_all_features.json"),
            Path("reports/model_training/metrics_lgbm_v2_base_features.json"),
            Path("reports/model_training/metrics_lgbm_v3_expansion_only.json"),
        ],
        metavar="JSON",
    )
    p.add_argument(
        "--importance",
        type=Path,
        nargs=3,
        default=[
            Path("reports/model_training/feature_importance_lgbm_v1_all_features.csv"),
            Path("reports/model_training/feature_importance_lgbm_v2_base_features.csv"),
            Path("reports/model_training/feature_importance_lgbm_v3_expansion_only.csv"),
        ],
    )
    p.add_argument(
        "--factor-ic",
        type=Path,
        default=Path("reports/model_training/factor_ic_analysis.csv"),
    )
    p.add_argument("--out", type=Path, default=Path("reports/model_training/model_comparison.md"))
    return p.parse_args()


def _load_metrics(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    rows = [
        ("v1 全特征", "63 个（含扩展，2019 起扩展列可为 NaN）", "16 期（2019–2022）", args.metrics[0], args.importance[0]),
        ("v2 基础", "41 个", "16 期", args.metrics[1], args.importance[1]),
        ("v3 扩展期", "63 个", "8 期（train as_of≥2021-01-01）", args.metrics[2], args.importance[2]),
    ]
    lines = [
        "# 模型版本对比（LightGBM LambdaRank）",
        "",
        "| 模型版本 | 特征集 | 训练期 | Test IC_mean | Test ICIR | IC>0 率 | best_iter |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    tops: list[set[str]] = []
    for label, feat_desc, train_desc, mj, ij in rows:
        m = _load_metrics(mj)
        if not m:
            lines.append(f"| {label} | {feat_desc} | {train_desc} | (无 metrics 文件) | — | — | — |")
            tops.append(set())
            continue
        icm = m.get("test_IC_mean", float("nan"))
        icir = m.get("test_ICIR", float("nan"))
        wr = m.get("test_IC_positive_rate", float("nan"))
        bi = m.get("best_iteration", 0)
        lines.append(
            f"| {label} | {feat_desc} | {train_desc} | {icm:+.6f} | {icir:+.6f} | {wr:.1%} | {bi} |"
        )
        if ij.is_file():
            dfi = pd.read_csv(ij)
            tops.append(set(dfi.sort_values("gain", ascending=False).head(10)["feature"].tolist()))
        else:
            tops.append(set())

    common = tops[0] & tops[1] & tops[2] if len(tops) == 3 else set()

    lines.extend([
        "",
        "## 三版本共同 Top10 特征（按 gain 截断重叠）",
        "",
        f"- 共同因子数: **{len(common)}**",
    ])
    if common:
        lines.append(f"- 列表: {', '.join(sorted(common))}")
    else:
        lines.append("- （无完全重叠，或缺少 importance CSV）")

    lines.extend(["", "## 单因子 IC 最强的 5 个（全样本 train 截面 IC 分析）", ""])
    if args.factor_ic.is_file():
        df = pd.read_csv(args.factor_ic)
        if "ICIR" in df.columns and "factor" in df.columns:
            df["abs_ICIR"] = df["ICIR"].abs()
            top5 = df.sort_values("abs_ICIR", ascending=False).head(5)
            lines.append("| factor | IC_mean | IC_std | ICIR | IC>0 率 |")
            lines.append("| --- | ---: | ---: | ---: | ---: |")
            for _, r in top5.iterrows():
                lines.append(
                    f"| {r['factor']} | {r['IC_mean']:+.6f} | {r.get('IC_std', float('nan')):+.6f} | "
                    f"{r['ICIR']:+.6f} | {r.get('IC_positive_rate', float('nan')):.2%} |"
                )
        else:
            lines.append("（factor_ic_analysis.csv 列不符合预期）")
    else:
        lines.append("（未找到 factor_ic_analysis.csv）")

    # 选用建议：Test ICIR 最大
    best_idx = -1
    best_ir = float("-inf")
    for i, mj in enumerate(args.metrics):
        m = _load_metrics(mj)
        ir = m.get("test_ICIR", float("nan"))
        if ir == ir and ir > best_ir:
            best_ir = ir
            best_idx = i
    ver_names = ["v1 全特征", "v2 基础", "v3 扩展期"]
    rec = ver_names[best_idx] if best_idx >= 0 else "（数据不足）"

    lines.extend([
        "",
        "## 生产模型建议",
        "",
        f"- **建议首选**: **{rec}**（以 `metrics_*.json` 中 **Test ICIR** 最高为准）。",
        "- v2 仅用基础因子，训练期最全、无扩展列全 NaN，**稳定性 Often 更好**。",
        "- v3 在 2021 年后数据上对齐扩展因子含义，但若 Test 仍为 2023，需注意 **训练/检验时间错位**。",
        "- v1 利用全特征 + LightGBM 处理 NaN，适合作为 **上限探索**；需监控过拟合。",
        "",
        "## 下一步调参方向",
        "",
        "- 调整 `num_leaves` / `min_child_samples` / `colsample_bytree` 压制过拟合。",
        "- 尝试更长标签 `forward_return_63d` 或更换 rank 标签分位数 `n_quantiles`。",
        "- 引入 `FactorSelector` 预筛因子（基于 `factor_ic_analysis.csv`）再做 LGBM。",
        "- Walk-Forward（`walk_forward_evaluate`）做全样本时序鲁棒性检验。",
        "",
    ])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out.resolve()}")


if __name__ == "__main__":
    main()
