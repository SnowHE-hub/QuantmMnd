"""quantmind.line_a.harness.evalset — 研究质量评测集（03_final_plan_v2 第 10.1 节第二条评估线；第 13 节 P0 并行项）.

第一版针对 A1（财务事实），标准答案全部来自结构化数据，可以自动生成、可以复现：
- 抽样：按固定种子在若干历史时点上抽取股票（上市满 3 年，按板块分层），只用 as_of 当天或之前已公告的记录；
- 标准事实：
  - 最新报告期的营业收入、归母净利润，以及上年同期值；
  - 最近 120 天内的业绩预告类型；
  - 最近一份年报的审计意见。
  每条都带来源证据编号，与 A1 工具返回的编号一致。
- 打分：
  - 事实召回：标准事实里，被一条"已核实"的声明以同一证据编号和同一字段覆盖的比例；
  - 已核实占比；
  - 有问题声明的比例（mismatch + missing_evidence）；
  - 弃答率。
评测题与研究期无关，只检验"是否把材料里的事实准确地取出来"；不涉及收益，不能用来证明策略有效。
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from quantmind.backtest.wf_costs import board_of


def _ymd(d) -> str:
    return pd.Timestamp(d).strftime("%Y%m%d")


def sample_cases(lake: Path, as_ofs: list[str], per_date: int = 10, seed: int = 20261021) -> pd.DataFrame:
    sb = pd.read_parquet(sorted(lake.glob("stock_basic_full_merged_2*.parquet"))[-1],
                         columns=["ts_code", "list_date", "delist_date"])
    rows = []
    for k, d in enumerate(as_ofs):
        lim = _ymd(pd.Timestamp(d) - pd.Timedelta(days=3 * 365))
        live = sb[(sb["list_date"].astype(str) <= lim) & (sb["delist_date"].isna() | (sb["delist_date"].astype(str) > _ymd(d)))]
        live = live.assign(board=live["ts_code"].map(board_of))
        live = live[live["board"].isin(["MAIN", "CHINEXT", "STAR"])]
        rng = np.random.default_rng(seed + k)
        for b, g in live.groupby("board"):
            n = max(1, round(per_date * {"MAIN": 0.6, "CHINEXT": 0.25, "STAR": 0.15}[b]))
            for t in rng.choice(g["ts_code"].to_numpy(), size=min(n, len(g)), replace=False):
                rows.append({"as_of": _ymd(d), "ticker": t, "board": b})
    out = pd.DataFrame(rows)
    out["case_id"] = [hashlib.sha256(f"{r.as_of}:{r.ticker}".encode()).hexdigest()[:10] for r in out.itertuples()]
    return out


def gold_facts_a1(lake: Path, ticker: str, as_of: str) -> list[dict]:
    from quantmind.line_a.harness.tools_data import _latest, _rows
    facts: list[dict] = []
    inc = _rows(lake / "income.parquet", ticker, ["ts_code", "ann_date", "end_date", "revenue", "n_income_attr_p"])
    inc = inc[inc["ann_date"].astype(str) <= as_of]
    if len(inc):
        inc = inc.sort_values(["end_date", "ann_date"])
        last = inc.groupby("end_date").tail(1)                       # 每个报告期取 as_of 前的最新版本
        cur = last.iloc[-1]
        prev_end = str(int(str(cur["end_date"])[:4]) - 1) + str(cur["end_date"])[4:]
        prev = last[last["end_date"].astype(str) == prev_end]
        for f in ("revenue", "n_income_attr_p"):
            if pd.notna(cur[f]):
                facts.append({"fact_id": f"income.{f}.latest", "metric": f, "period": str(cur["end_date"]),
                              "value": float(cur[f]), "unit": "元",
                              "evidence_id": f"tushare:income:{ticker}:{cur['end_date']}:{cur['ann_date']}"})
            if len(prev) and pd.notna(prev.iloc[0][f]):
                p = prev.iloc[0]
                facts.append({"fact_id": f"income.{f}.prev_year", "metric": f, "period": prev_end,
                              "value": float(p[f]), "unit": "元",
                              "evidence_id": f"tushare:income:{ticker}:{p['end_date']}:{p['ann_date']}"})
    fc = _latest(lake, "forecast")
    if fc is not None:
        df = _rows(fc, ticker)
        a = df["ann_date"].astype(str) if len(df) else pd.Series(dtype=str)
        lo = _ymd(pd.Timestamp(as_of) - pd.Timedelta(days=120))
        df = df[(a > lo) & (a <= as_of)].sort_values("ann_date") if len(df) else df
        if len(df):
            r = df.iloc[-1]
            facts.append({"fact_id": "forecast.type.latest", "metric": "type", "period": str(r["end_date"]),
                          "value": str(r["type"]), "unit": None,
                          "evidence_id": f"tushare:forecast:{ticker}:{r['end_date']}:{r['ann_date']}"})
    au = _latest(lake, "fina_audit")
    if au is not None:
        df = _rows(au, ticker)
        if len(df):
            df = df[(df["ann_date"].astype(str) <= as_of) & df["end_date"].astype(str).str.endswith("1231")]
            if len(df):
                r = df.sort_values("end_date").iloc[-1]
                facts.append({"fact_id": "audit.result.latest_annual", "metric": "audit_result",
                              "period": str(r["end_date"]), "value": str(r.get("audit_result")), "unit": None,
                              "evidence_id": f"tushare:fina_audit:{ticker}:{r['end_date']}:{r['ann_date']}"})
    return facts


def score_a1(output: dict | None, checks: list[dict], gold: list[dict], status: str) -> dict:
    if not output or status != "ok":
        return {"abstained": True, "recall": 0.0, "n_gold": len(gold)}
    st = {c["local_id"]: c["status"] for c in checks}
    hit = set()
    for c in output.get("claims", []):
        if st.get(c["local_id"]) not in ("verified", "minor_mismatch"):
            continue
        for e in c.get("evidence", []):
            for g in gold:
                if e.get("evidence_id") == g["evidence_id"] and (e.get("field") in (g["metric"], None)):
                    hit.add(g["fact_id"])
    n = max(1, len(checks))
    bad = sum(c["status"] in ("mismatch", "missing_evidence") for c in checks)
    return {"abstained": False, "n_gold": len(gold), "n_hit": len(hit), "recall": len(hit) / max(1, len(gold)),
            "n_claims": len(checks), "verified_share": sum(c["status"] == "verified" for c in checks) / n,
            "problem_share": bad / n, "missed": sorted({g["fact_id"] for g in gold} - hit)}
