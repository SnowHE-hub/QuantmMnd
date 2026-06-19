#!/usr/bin/env python
"""P3f：分桶深度模型合并 + 全1374/分桶 评估（seq vs tabular 公平对决）.

- seq 模型：3 桶预测合并为全 1374。桶内(as_of)按 score rank→[0,1] 归一化再拼接（消除三个独立
  桶模型的尺度差），算全 1374 中性化 IC；同时各桶用原始 score 报分桶 neut IC。
- tabular 模型（已在全 1374）：全 + 把预测筛到各桶报分桶 neut IC。
- 输出对比表：model × (full neut IC, 桶0/1/2 neut IC)，回答"序列模型能否超过 Ridge(full)"。
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
from scripts.bakeoff.evaluate_bakeoff import score_predictions, _load_context

PREDS = REPO / "data/bakeoff/preds"
BUCKETS = REPO / "data/bakeoff/deep_buckets"
PERIOD = "12d"
panel, lmc = _load_context()


def bucket_tickers():
    d = {}
    for f in sorted(BUCKETS.glob("bucket*of*.json")):
        j = json.loads(f.read_text()); d[int(j["bucket"])] = set(j["tickers"])
    return d


def neut_full_and_buckets(preds, bt, score_col="score"):
    """preds: as_of,ticker,<score_col>。返回 full + 各桶 neut IC。"""
    p = preds[["as_of", "ticker"]].copy()
    p["score"] = preds[score_col].values   # 避免 rename 与既有 score 列碰撞致 2D
    res = {}
    m = score_predictions(p, period=PERIOD, panel=panel, lmc=lmc)
    res["full_neut_ic"] = m["neut_ic"]; res["full_neut_icir"] = m["neut_icir"]
    res["full_net_excess"] = m.get("net_excess_cost"); res["full_dd"] = m.get("max_drawdown")
    for b, ts in sorted(bt.items()):
        sub = p[p["ticker"].isin(ts)]
        if len(sub) > 1000:
            mb = score_predictions(sub, period=PERIOD, panel=panel, lmc=lmc)
            res[f"b{b}_neut_ic"] = mb["neut_ic"]
    return res


def seq_combined(model, bt, seed=0):
    parts = []
    for b in sorted(bt):
        f = PREDS / f"{model}_alpha360_{PERIOD}_quarterly_s{seed}_b{b}of{len(bt)}.parquet"
        if not f.exists():
            return None
        x = pd.read_parquet(f); x["bucket"] = b; parts.append(x)
    allp = pd.concat(parts, ignore_index=True)
    allp["as_of"] = pd.to_datetime(allp["as_of"])
    # 桶内(as_of) rank → [0,1]，使三桶可比后拼成全 1374
    allp["score_rank"] = allp.groupby(["as_of", "bucket"])["score"].rank(pct=True)
    return allp


def main():
    bt = bucket_tickers()
    if not bt:
        print("no bucket membership files yet"); return
    rows = []
    # seq models（分桶合并）
    for model in ["gru", "lstm", "transformer"]:
        comb = seq_combined(model, bt)
        if comb is None:
            print(f"[skip] {model}: not all {len(bt)} buckets done yet"); continue
        r = neut_full_and_buckets(comb, bt, score_col="score_rank")
        r["model"] = f"{model}(seq,3桶)"; rows.append(r)
    # tabular（全 1374，已存预测）— 全 + 分桶
    for f in ["ridge_full_12d_quarterly", "lgbm_full_12d_quarterly",
              "lgbm_plus_surv_12d_quarterly", "lgbm_v2_35_12d_quarterly"]:
        fp = PREDS / f"{f}.parquet"
        if not fp.exists():
            continue
        r = neut_full_and_buckets(pd.read_parquet(fp), bt)
        r["model"] = f.replace("_12d_quarterly", ""); rows.append(r)
    if not rows:
        print("nothing to eval yet"); return
    df = pd.DataFrame(rows)
    cols = ["model", "full_neut_ic", "full_neut_icir", "b0_neut_ic", "b1_neut_ic",
            "b2_neut_ic", "full_net_excess", "full_dd"]
    df = df[[c for c in cols if c in df.columns]].sort_values("full_neut_ic", ascending=False)
    df.to_csv(REPO / "data/bakeoff/bucket_leaderboard.csv", index=False)
    print("=== 全1374 + 分桶 neut IC（桶0=最活跃…桶2=最不活跃）===")
    print("（seq=3桶合并rank；Ridge(full)基准=+0.034 on 1374）")
    print(df.to_string(index=False))
    # F-13：run manifest（溯源）
    from scripts.bakeoff.run_manifest import write_run_manifest
    write_run_manifest(REPO / "data/bakeoff", kind="eval", data_version="v5",
                       feature_set="seq-3bucket + tabular", label="forward_return_12d", horizon=PERIOD,
                       extra={"leaderboard": "bucket_leaderboard.csv", "n_models": int(len(df))})


if __name__ == "__main__":
    main()
