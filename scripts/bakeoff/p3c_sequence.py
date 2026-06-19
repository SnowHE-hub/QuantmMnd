#!/usr/bin/env python
"""P3c：序列模型（GRU/LSTM/Transformer）fold 驱动训练 + 导出（qlib env, GPU）.

制式 = Alpha360（过去 60 日 OHLCV，默认 CSZScoreNorm 逐日截面标准化）。
label = Ref($close,-H)/$close-1（与面板 forward_return_{H}d 精确一致，G1 已验）。
fold 由 PurgedWalkForwardSplit（quantmind）驱动；深度模型在 fold 的 train/val/test 日期【范围】上
训练（qlib 用范围内全日频，发挥长处），预测后只取周频 as_of test 日。方向 H-A 用 val IC 符号。
导出：data/bakeoff/preds/<model>_alpha360_<period>_<refit>_s<seed>.parquet (as_of,ticker,score)
"""
import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"   # mlflow3.13 弃用 filestore，显式 opt-in
import warnings, json, sys, time, argparse
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
import qlib
from qlib.data.dataset import DatasetH
from qlib.contrib.data.handler import Alpha360

REPO = Path("/home/lenovo/projects/quantmind"); sys.path.insert(0, str(REPO))
PREDS = REPO / "data/bakeoff/preds"; PREDS.mkdir(parents=True, exist_ok=True)
PERIOD = {"12d": (12, 20), "63d": (63, 63)}

# qlib env 无 akshare → 不走 quantmind 包 __init__；standalone 载 wf_split + 自建日历
import importlib.util
def _load_wf_split_cls():
    name = "wf_split_standalone"
    spec = importlib.util.spec_from_file_location(name, REPO / "quantmind/backtest/wf_split.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod                      # 注册：dataclass type-hint 解析需 sys.modules[__module__]
    spec.loader.exec_module(mod)
    return mod.PurgedWalkForwardSplit

def _calendar():
    td = pd.read_parquet(REPO / "data/raw/alpha_prices_panel.parquet", columns=["trade_date"])["trade_date"]
    return pd.DatetimeIndex(sorted(pd.to_datetime(td).unique()))


def qsym_map():
    m = json.loads((REPO / "data/bakeoff/ticker_qsym_map.json").read_text())  # qsym->ts
    return {k.upper(): v for k, v in m.items()}


def bucket_universe(q2t, n_buckets, bucket_idx):
    """按流动性（日均 amount 中位数）降序切 n_buckets 桶，返回第 bucket_idx 桶（覆盖全 1374、控内存）。
    桶0=最活跃…桶(n-1)=最不活跃；每桶 ~1374/n_buckets 只。保存桶成员供评估端合并/分桶报。"""
    px = pd.read_parquet(REPO / "data/raw/alpha_prices_panel.parquet", columns=["ts_code", "amount"])
    med = px.groupby("ts_code")["amount"].median().sort_values(ascending=False)
    ts2q = {v: k for k, v in q2t.items()}
    all_q = [ts2q[t] for t in med.index if t in ts2q]          # 流动性降序
    qs = list(np.array_split(all_q, n_buckets)[bucket_idx])
    bdir = REPO / "data/bakeoff/deep_buckets"; bdir.mkdir(parents=True, exist_ok=True)
    (bdir / f"bucket{bucket_idx}of{n_buckets}.json").write_text(json.dumps(
        {"bucket": bucket_idx, "n_buckets": n_buckets, "n": len(qs),
         "tickers": sorted(q2t[x] for x in qs)}, ensure_ascii=False, indent=2))
    return sorted(qs)


def make_cutoffs(cal, start, end, refit):
    freq = "QS-JAN" if refit == "quarterly" else "MS"
    cs = sorted(cal); out = []
    for qd in pd.date_range(start, end, freq=freq):
        a = [d for d in cs if d >= qd]
        if a: out.append(a[0])
    return out


def label_handler(H):
    class A360H(Alpha360):
        def get_label_config(self):
            return ([f"Ref($close,-{H})/$close-1"], ["LABEL0"])
    return A360H


def build_model(name, seed):
    from qlib.contrib.model.pytorch_gru import GRU
    from qlib.contrib.model.pytorch_lstm import LSTM
    from qlib.contrib.model.pytorch_transformer import TransformerModel
    common = dict(d_feat=6, n_epochs=30, lr=1e-3, early_stop=8, batch_size=800,
                  GPU=0, seed=seed)
    if name == "gru":
        return GRU(hidden_size=64, num_layers=2, dropout=0.3, **common)
    if name == "lstm":
        return LSTM(hidden_size=64, num_layers=2, dropout=0.3, **common)
    if name == "transformer":
        return TransformerModel(d_model=64, nhead=4, num_layers=2, dropout=0.3,
                                **{k: v for k, v in common.items() if k != "d_feat"}, d_feat=6)
    raise ValueError(name)


def run(model_name, period, refit, seed):
    PurgedWalkForwardSplit = _load_wf_split_cls()
    H, E = PERIOD[period]
    # kernels=2：限制 qlib 数据加载并行度，避免 12 worker 内存翻倍致 OOM（WSL 12GB 硬上限）
    qlib.init(provider_uri=str(REPO / "data/qlib_cn_daily"), region="cn",
              expression_cache=None, dataset_cache=None, kernels=2)
    q2t = qsym_map()
    n_buckets = int(os.environ.get("QM_N_BUCKETS", "3"))
    bucket = int(os.environ.get("QM_BUCKET", "0"))
    insts = bucket_universe(q2t, n_buckets, bucket)   # 流动性分桶：覆盖全1374、控内存
    cal = _calendar()
    asof = sorted({d for (d, t) in pd.read_parquet(
        REPO / "data/panel/alpha_panel_weekly_v5.parquet", columns=["forward_return_12d"]).index})
    asof_set = {pd.Timestamp(a).normalize() for a in asof}
    oos_start = pd.Timestamp("2022-01-01"); oos_end = asof[-1]
    sp = PurgedWalkForwardSplit(cal, horizon=H, embargo=E, mode="rolling",
                                rolling_lookback_td=756, n_val=2)
    cut = make_cutoffs(cal, oos_start, pd.Timestamp("2025-10-01"), refit)
    folds = sp.make_folds(asof, cut, oos_start=oos_start, oos_end=oos_end)
    Hnd = label_handler(H)
    import gc
    max_folds = int(os.environ.get("QM_MAX_FOLDS", "0")) or len(folds)
    valid_folds = [f for f in folds if f.test_dates and f.train_dates][:max_folds]
    rows, dirs = [], {}; t0 = time.time()
    # ProcessInf→CSZScoreNorm(逐日截面)→Fillna(0)：缺 Fillna 会让 inf/0方差日的特征→NaN→预测全NaN
    csz_f = [{"class": "ProcessInf", "kwargs": {}},
             {"class": "CSZScoreNorm", "kwargs": {"fields_group": "feature"}},
             {"class": "Fillna", "kwargs": {"fields_group": "feature"}}]
    csz_l = [{"class": "DropnaLabel"}, {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}}]
    # 全范围一次性 build 在 12GB 下 OOM（7.3年×1374×360）；改 per-fold（每折~3年窗口可放下）+ fold 间 gc 释放
    print(f"[{model_name}/alpha360/{period}/{refit}/s{seed}] folds={len(valid_folds)} insts={len(insts)} "
          f"per-fold handler (memory-safe, gc between)", flush=True)
    for i, fold in enumerate(valid_folds):
        tr0, tr1 = str(min(fold.train_dates).date()), str(max(fold.train_dates).date())
        va0, va1 = str(min(fold.val_dates).date()), str(max(fold.val_dates).date())
        te0, te1 = str(min(fold.test_dates).date()), str(max(fold.test_dates).date())
        h = Hnd(instruments=insts, start_time=tr0, end_time=te1,
                fit_start_time=tr0, fit_end_time=tr1,
                infer_processors=csz_f, learn_processors=csz_l)
        ds = DatasetH(h, segments={"train": (tr0, tr1), "valid": (va0, va1), "test": (te0, te1)})
        from qlib.workflow import R
        try:
            m = build_model(model_name, seed)
            with R.start(experiment_name=f"bakeoff_{model_name}", uri="file:///tmp/qlib_mlruns"):
                m.fit(ds)
                pr = m.predict(ds, segment="test")
                d = _val_direction(m, ds, H)
        except Exception as e:
            print(f"  fold {i} ERROR {repr(e)[:160]}"); continue
        pr = pr.reset_index(); pr.columns = ["datetime", "instrument", "score"] \
            if str(pr.columns[0]).lower().startswith("date") else ["instrument", "datetime", "score"]
        pr["datetime"] = pd.to_datetime(pr["datetime"]).dt.normalize()
        dirs[i] = int(d)   # d 已在 R.start 块内由 _val_direction 计算（H-A，仅用 val）
        prf = pr[pr["datetime"].isin(asof_set)].copy()
        prf["ticker"] = prf["instrument"].astype(str).str.upper().map(q2t)   # 防 categorical 列 .str 失效
        prf["score"] = prf["score"] * d
        rows.append(prf.dropna(subset=["ticker", "score"])[["datetime", "ticker", "score"]]
                    .rename(columns={"datetime": "as_of"}))
        print(f"  fold {i} dir={d:+d} test_rows={len(prf)} ({time.time()-t0:.0f}s)", flush=True)
        # fold 间显式释放，防内存累积 OOM（WSL 12GB 硬上限）
        try:
            import torch
            del m, ds, h, pr, prf
            torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()
    preds = pd.concat(rows, ignore_index=True)
    out = PREDS / f"{model_name}_alpha360_{period}_{refit}_s{seed}_b{bucket}of{n_buckets}.parquet"
    preds.to_parquet(out)
    (PREDS / f"{out.stem}.dirs.json").write_text(json.dumps(
        {"dirs": dirs, "n_flips": sum(1 for v in dirs.values() if v < 0)}, indent=2))
    print(f"[save] {out} shape={preds.shape} ({time.time()-t0:.0f}s)")
    # F-13：run manifest（溯源）
    from scripts.bakeoff.run_manifest import write_run_manifest
    write_run_manifest(PREDS, kind="train", data_version="v5", feature_set="alpha360",
                       label=f"forward_return_{H}d", horizon=period,
                       extra={"model": model_name, "bucket": f"{bucket}of{n_buckets}", "preds": out.name})


def _val_direction(model, ds, H):
    """val 段：预测 vs 标签 的截面 IC 均值符号（H-A，只用 val）。"""
    from scipy import stats
    try:
        pv = model.predict(ds, segment="valid")
        lab = ds.prepare("valid", col_set="label").iloc[:, 0]
        j = pd.concat([pv.rename("p"), lab.rename("y")], axis=1).dropna()
        if len(j) < 30:
            return 1
        ic = j.groupby(level=0).apply(lambda g: stats.spearmanr(g["p"], g["y"])[0] if len(g) > 5 else np.nan)
        return 1 if float(np.nanmean(ic)) >= 0 else -1
    except Exception:
        return 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["gru", "lstm", "transformer"], required=True)
    ap.add_argument("--period", default="12d"); ap.add_argument("--refit", default="quarterly")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run(a.model, a.period, a.refit, a.seed)
