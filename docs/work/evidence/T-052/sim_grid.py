"""T-052 模拟盘回测网格（登记见 docs/work/tasks/T-052.md；验证 2 已通过后运行）.

全部组合都跑、都报告；主配置（每档资金一个）另存完整的净值、成交、未成交和持仓，并做移动块自助区间。
用法：PYTHONPATH=. python docs/work/evidence/T-052/sim_grid.py [--workers 10] [--limit N]
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim_data as sd  # noqa: E402

from quantmind.line_a.sim_account import AccountSimulator, Market, SimConfig  # noqa: E402

OUT = sd.ROOT / "data/line_a_v2/t052_sim"
WINDOWS = {"W6": (pd.Timestamp("2026-04-23"), pd.Timestamp("2026-10-23")),
           "W3": (pd.Timestamp("2026-07-23"), pd.Timestamp("2026-10-23"))}
ALL = frozenset({"MAIN", "CHINEXT", "STAR", "BSE"})
SMALL = frozenset({"MAIN", "CHINEXT"})
MAIN_ONLY = frozenset({"MAIN"})
SIGNALS = {"ENS20": ("ENS20", None, (1, 2, 4)), "C5-20": ("C5-20", None, (1, 2, 4)),
           "ENS63": ("ENS63", None, (1, 2, 4, 13)), "Q63": ("ENS63", (0.2, 1e7), (1, 2, 4, 13))}
NS = (10, 20, 30, 50, 100, 300)
AUMS = (1e5, 1e6, 1e7)
PRIMARY = {1e5: ("ENS20", 10), 1e6: ("ENS20", 30), 1e7: ("ENS20", 100)}
N_BOOT, BLOCK, SEED = 10_000, 20, 0

G: dict = {}


def load() -> None:
    start = min(w[0] for w in WINDOWS.values())
    end = max(w[1] for w in WINDOWS.values())
    px = sd.prices(start)
    e20, i20 = sd.ens_scores("20d", start)
    e63, i63 = sd.ens_scores("63d", start)
    G.update(market=Market(px, start, end), amihud=sd.amihud(start), divs=sd.dividends(), st=sd.st_intervals(),
             scores={"ENS20": e20, "ENS63": e63, "C5-20": sd.c5_scores(e20, px)}, info={"20d": i20, "63d": i63},
             index=sd.indexes(start - pd.Timedelta(days=10), end))


def configs() -> list[dict]:
    out = []
    for (sig, (base, quota, steps)), n, buf, ex, aum, win in itertools.product(
            SIGNALS.items(), NS, (None, 2.0), ("open", "vwap"), AUMS, WINDOWS):
        for step in steps:
            out.append({"signal": sig, "base": base, "quota": quota, "n": n, "step": step, "buffer": buf, "exec": ex,
                        "aum": aum, "window": win, "boards": "main+chinext" if aum < 5e5 else "all"})
    for aum, (sig, n) in PRIMARY.items():                       # 只开主板的对照（主配置）
        for win in WINDOWS:
            out.append({"signal": sig, "base": sig, "quota": None, "n": n, "step": 4, "buffer": 2.0, "exec": "open",
                        "aum": aum, "window": win, "boards": "main_only"})
    for c in out:
        c["primary"] = (PRIMARY.get(c["aum"]) == (c["signal"], c["n"]) and c["step"] == 4 and c["buffer"] == 2.0
                        and c["exec"] == "open" and c["boards"] != "main_only")
        c["key"] = f"{c['window']}_{c['signal']}_n{c['n']}_s{c['step']}_b{c['buffer'] or 0}_{c['exec']}_" \
                   f"{int(c['aum']):d}_{c['boards']}"
    return out


def run_one(c: dict) -> dict:
    t0 = time.time()
    start, end = WINDOWS[c["window"]]
    boards = {"all": ALL, "main+chinext": SMALL, "main_only": MAIN_ONLY}[c["boards"]]
    cfg = SimConfig(aum=c["aum"], top_n=c["n"], rebalance_step=c["step"], buffer_mult=c["buffer"], exec_price=c["exec"],
                    boards=boards, quota_ext=c["quota"])
    try:
        r = AccountSimulator(G["scores"][c["base"]], G["market"], cfg, start=start, end=end, amihud=G["amihud"],
                             dividends=G["divs"], st_intervals=G["st"]).run()
    except Exception as e:  # noqa: BLE001
        return c | {"error": f"{type(e).__name__}: {e}"}
    nav = r["nav"]
    s = r["summary"]
    idx = G["index"].reindex(nav.index).ffill()
    for k in idx.columns:
        s[f"ret_{k}"] = float(idx[k].iloc[-1] / idx[k].iloc[0] - 1)
    rej = r["rejects"]["reason"].value_counts().to_dict() if len(r["rejects"]) else {}
    tg = r["targets"]
    row = c | {k: v for k, v in s.items() if k != "costs"} | {f"cost_{k}": v for k, v in s["costs"].items()} | \
        {"rejects": json.dumps(rej, ensure_ascii=False), "n_decisions": int(len(tg)),
         "first_decision": str(tg["as_of"].min().date()) if len(tg) else None, "secs": round(time.time() - t0, 1)}
    if c["primary"] or c["boards"] == "main_only":
        d = OUT / "runs" / c["key"]
        d.mkdir(parents=True, exist_ok=True)
        nav.to_parquet(d / "nav.parquet")
        r["trades"].to_parquet(d / "trades.parquet")
        if len(r["rejects"]):
            r["rejects"].to_parquet(d / "rejects.parquet")
        tg.assign(targets=tg["targets"].map(list)).to_parquet(d / "targets.parquet")
    return row


def boot_ci(nav: pd.DataFrame, bench: pd.Series) -> dict:
    r = nav["equity"].pct_change().dropna()
    b = bench.reindex(r.index).pct_change().dropna() if bench is not None else None
    ex = (r - b.reindex(r.index)).dropna().to_numpy()
    n = len(ex)
    rng = np.random.default_rng(SEED)
    nb = int(np.ceil(n / BLOCK))
    starts = rng.integers(0, n - BLOCK + 1, size=(N_BOOT, nb))
    idx = (starts[:, :, None] + np.arange(BLOCK)).reshape(N_BOOT, -1)[:, :n]
    tot = np.prod(1 + ex[idx], axis=1) - 1
    return {"excess_compound": float(np.prod(1 + ex) - 1), "ci95": [float(np.quantile(tot, 0.025)),
                                                                     float(np.quantile(tot, 0.975))], "days": n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    load()
    print(f"loaded in {time.time() - t0:.0f}s; signal info {json.dumps(G['info'], default=str)[:600]}", flush=True)
    cs = configs()
    cs = sorted(cs, key=lambda c: not c["primary"])
    if a.limit:
        cs = cs[: a.limit]
    print(f"{len(cs)} configs", flush=True)
    rows = []
    with get_context("fork").Pool(a.workers) as pool:
        for i, row in enumerate(pool.imap_unordered(run_one, cs, chunksize=4)):
            rows.append(row)
            if i % 50 == 0 or row.get("primary"):
                print(i, row["key"], row.get("total_return"), row.get("error"), row.get("secs"), flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(OUT / ("grid_summary.parquet" if not a.limit else "grid_summary_test.parquet"))
    prim = {}
    for c in [c for c in cs if c["primary"] or c["boards"] == "main_only"]:
        d = OUT / "runs" / c["key"]
        if not (d / "nav.parquet").exists():
            continue
        nav = pd.read_parquet(d / "nav.parquet")
        prim[c["key"]] = {"config": c, "summary": df[df["key"] == c["key"]].iloc[0].to_dict(),
                          "vs_pool_ew": boot_ci(nav, nav["bench_pool_ew"]),
                          "vs_zz1000": boot_ci(nav, G["index"]["中证1000"].reindex(nav.index).ffill())}
    (OUT / "primary.json").write_text(json.dumps(prim, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    (OUT / "signal_info.json").write_text(json.dumps(G["info"], ensure_ascii=False, indent=1, default=str),
                                          encoding="utf-8")
    print("DONE", len(df), "rows;", int(df.get("error", pd.Series(dtype=object)).notna().sum()), "errors;",
          f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
