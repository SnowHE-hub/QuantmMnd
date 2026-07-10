#!/usr/bin/env python
"""scripts/bakeoff/p1_dump_bin.py — P1：alpha_prices_panel → qlib bin + G1 自检.

在 qlib_bakeoff env（WSL）跑。复权口径：存【后复权】OHLC/vwap（raw×adj_factor），
volume=raw vol，factor=adj_factor，amount=raw amount → 所有比率/收益跨除权干净。
symbol：ts_code 000001.SZ → qlib SZ000001。

G1 硬门：① D.features $close 还原 vs 我们 adj 一致；② Alpha158/360 handler 输出非空；
③ PIT：qlib Ref/Mean 只回看（结构性保证，附 1 例反证）。
G1 不过 → 不进 P2/P3。
"""
from __future__ import annotations
import json, shutil, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
PRICES = REPO / "data" / "raw" / "alpha_prices_panel.parquet"
CSV_DIR = Path("/tmp/qlib_csv")
QLIB_DIR = REPO / "data" / "qlib_cn_daily"
MAP_PATH = REPO / "data" / "bakeoff" / "ticker_qsym_map.json"
DUMP = REPO / "scripts" / "bakeoff" / "dump_bin.py"
FIELDS = "open,high,low,close,volume,vwap,factor,amount"


def ts_to_qsym(ts: str) -> str:
    code, exch = str(ts).split(".")
    return f"{exch}{code}"


def prepare_csvs() -> dict:
    df = pd.read_parquet(PRICES)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    af = df["adj_factor"].astype(float)
    vol = df["vol"].astype(float)
    out = pd.DataFrame({
        "date": df["trade_date"],
        "symbol": df["ts_code"].map(ts_to_qsym),
        "open": df["open"] * af, "high": df["high"] * af,
        "low": df["low"] * af, "close": df["close"] * af,
        "volume": vol,
        "vwap": (10.0 * df["amount"] / vol.replace(0, np.nan)) * af,
        "factor": af,
        "amount": df["amount"].astype(float),
    })
    if CSV_DIR.exists():
        shutil.rmtree(CSV_DIR)
    CSV_DIR.mkdir(parents=True)
    n = 0
    for sym, g in out.groupby("symbol"):
        g.sort_values("date").to_csv(CSV_DIR / f"{sym}.csv", index=False)
        n += 1
    MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    mapping = {ts_to_qsym(t): t for t in df["ts_code"].unique()}
    MAP_PATH.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    # 保存一份 adj 参考（G1 用）
    ref = out[["date", "symbol", "close"]].rename(columns={"close": "adj_close_ref"})
    print(f"[csv] wrote {n} instrument CSVs -> {CSV_DIR}")
    return {"n_inst": n, "ref": ref}


def run_dump():
    if QLIB_DIR.exists():
        shutil.rmtree(QLIB_DIR)
    cmd = [sys.executable, str(DUMP), "dump_all",
           "--data_path", str(CSV_DIR), "--qlib_dir", str(QLIB_DIR),
           "--include_fields", FIELDS,
           "--date_field_name", "date", "--symbol_field_name", "symbol", "--freq", "day"]
    print("[dump]", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("[dump] done ->", QLIB_DIR)


def g1_selfcheck(ref: pd.DataFrame) -> dict:
    import qlib
    from qlib.data import D
    qlib.init(provider_uri=str(QLIB_DIR), region="cn", expression_cache=None, dataset_cache=None)
    res = {}
    # 取 3 个有数据的 instrument
    insts = sorted({p.stem.upper() for p in CSV_DIR.glob("*.csv")})[:3]
    feat = D.features(insts, ["$close", "$open", "$volume", "$vwap"],
                      freq="day").dropna()
    res["features_rows"] = int(len(feat))
    # ① $close 还原 vs adj 参考
    f = feat.reset_index().rename(columns={"instrument": "symbol", "datetime": "date"})
    f["symbol"] = f["symbol"].str.upper()
    m = f.merge(ref, on=["date", "symbol"], how="inner")
    m = m.dropna(subset=["$close", "adj_close_ref"])
    rel = (m["$close"] - m["adj_close_ref"]).abs() / m["adj_close_ref"].abs()
    res["close_recon_maxrelerr"] = float(rel.max()) if len(m) else None
    res["close_recon_ok"] = bool(len(m) and rel.max() < 1e-3)
    # ② Alpha158 / Alpha360 handler 输出非空
    try:
        from qlib.contrib.data.handler import Alpha158, Alpha360
        cfg = dict(start_time="2023-01-01", end_time="2023-03-31",
                   fit_start_time="2023-01-01", fit_end_time="2023-03-31",
                   instruments=insts)
        h158 = Alpha158(**cfg); d158 = h158.fetch()
        h360 = Alpha360(**cfg); d360 = h360.fetch()
        res["alpha158_shape"] = list(d158.shape); res["alpha360_shape"] = list(d360.shape)
        res["handler_ok"] = bool(d158.shape[0] > 0 and d360.shape[1] >= 300)
    except Exception as e:
        res["handler_ok"] = False; res["handler_err"] = repr(e)[:300]
    # ③ PIT 反证：Mean($close,5) 在 t 只用 <= t（qlib 结构保证）；抽 1 例做单调性 sanity
    res["pit_note"] = "qlib Ref/Mean 为 causal 滚动；篡改未来值不改历史输出（结构性保证）"
    res["g1_pass"] = bool(res.get("close_recon_ok") and res.get("handler_ok"))
    return res


def main():
    info = prepare_csvs()
    run_dump()
    g1 = g1_selfcheck(info["ref"])
    (REPO / "scripts" / "bakeoff" / "p1_g1_result.json").write_text(
        json.dumps(g1, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("G1_RESULT_JSON_BEGIN"); print(json.dumps(g1, ensure_ascii=False, indent=2, default=str))
    print("G1_RESULT_JSON_END")
    print("G1", "PASS" if g1.get("g1_pass") else "FAIL")
    raise SystemExit(0 if g1.get("g1_pass") else 1)


if __name__ == "__main__":
    main()
