"""quantmind.line_a.api_analytics — 只读接口背后的聚合计算（10-24 前端审查后新增）.

原来前端自己拼的几样东西，统一搬到后端、按系统已有的口径算：
- forward：事后收益。T 日收盘出信号 → T+1 开盘买入（复权价）→ 持有 h 个交易日按收盘计；
  入场日停牌、一字涨停（开盘即涨停价）记为"无法成交"，不算收益；对照基准：研究池（PIT 成交额前 1500）等权、
  沪深 300 / 中证 1000 / 中证 2000，超额 = 个股 − 基准。涨跌停价用 sim_account 的规则（T-052 同一套）。
- redteam：红队规则（大脑标记、重大事件、风险卡阈值）命中的股票，事后是不是真的跑输；和未命中股票的基准跑输率
  （base rate）比较，不用"跌了就算排雷成功"。
- consensus：四个模型的截面百分位（不用 CV：分数是截面 z 分，均值在 0 附近，CV 会爆炸）。
- claim_trace：F7 一条声明的真实核验记录（原声明、所引记录的字段值、校验器备注、按公式重算）。
- funnel：每一层的真实数量。
全部只读，不写任何文件；价格面板按文件修改时间缓存。
"""
from __future__ import annotations

import glob
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/lenovo/projects/quantmind")
OUT = ROOT / "data/line_a_v2"
LAKE = ROOT / "data/lake"
PANEL_START = "2026-01-01"
INDEXES = {"hs300": ("000300.SH", "沪深300"), "csi1000": ("000852.SH", "中证1000"), "csi2000": ("932000.CSI", "中证2000")}

# 红队展示规则 RT-v1：阈值沿用前端原实现（未经前瞻验证，只用来展示和事后评估，不进组合）
RT_RULES = {
    "brain_avoid_new_buy": "Agent 大脑标记 avoid_new_buy",
    "brain_caution": "Agent 大脑标记 caution / review_holding",
    "bypass_major": "全市场事件监测：重大事件",
    "beta_gt_2.5": "beta_252 > 2.5",
    "vol_gt_70pct": "60 日年化波动 > 70%",
    "limit_down_ge_2": "近 250 日跌停收盘 ≥ 2 次",
}


# ---------------------------------------------------------------- 价格
def _panel_file() -> str:
    fs = sorted(glob.glob(str(ROOT / "data/raw/alpha_prices_panel_v6_2*.parquet")))
    if not fs:
        raise FileNotFoundError("price panel")
    return fs[-1]


@lru_cache(maxsize=2)
def _market(path: str, mtime: float) -> dict:
    import pyarrow.dataset as pds
    cols = ["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "vol", "adj_factor"]
    t = pds.dataset(path).to_table(columns=cols, filter=pds.field("trade_date") >= pd.Timestamp(PANEL_START)).to_pandas()
    t["trade_date"] = pd.to_datetime(t["trade_date"]).dt.normalize()
    t = t.drop_duplicates(["trade_date", "ts_code"], keep="last")

    def wide(c):
        return t.pivot(index="trade_date", columns="ts_code", values=c).sort_index().astype("float64")
    m = {c: wide(c) for c in ("open", "high", "low", "close", "pre_close", "vol")}
    m["adj"] = wide("adj_factor").ffill()
    m["dates"] = m["close"].index
    idx = []
    for f in ("index_daily.parquet", "index_daily_extra.parquet"):
        p = LAKE / f
        if p.exists():
            x = pd.read_parquet(p, columns=["ts_code", "trade_date", "open", "close"])
            x["trade_date"] = pd.to_datetime(x["trade_date"].astype(str))
            idx.append(x[x["trade_date"] >= pd.Timestamp(PANEL_START)])
    ix = pd.concat(idx).drop_duplicates(["ts_code", "trade_date"], keep="last") if idx else pd.DataFrame()
    m["idx_open"] = ix.pivot(index="trade_date", columns="ts_code", values="open").sort_index() if len(ix) else pd.DataFrame()
    m["idx_close"] = ix.pivot(index="trade_date", columns="ts_code", values="close").sort_index() if len(ix) else pd.DataFrame()
    return m


def market() -> dict:
    p = _panel_file()
    return _market(p, Path(p).stat().st_mtime)


def _is_st(d: str) -> set[str]:
    try:
        f1 = pd.read_parquet(day_outputs(d)["f1"], columns=["is_st"])
        return set(f1.index[f1["is_st"].astype(bool)])
    except Exception:  # noqa: BLE001
        return set()


def day_outputs(d: str) -> dict:
    runs = sorted((OUT / "runs" / d).glob("day_*.json"), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise FileNotFoundError(f"no daily run for {d}")
    return json.loads(runs[-1].read_text(encoding="utf-8"))["outputs"]


def research_pool(d: str) -> list[str]:
    f2 = pd.read_parquet(day_outputs(d)["f2"])
    return sorted(f2.index[f2["domain"] == "research_pool"])


def forward(tickers: list[str], d: str, horizons=(1, 5, 10, 20), pool: list[str] | None = None) -> dict:
    """T 日（d）信号的事后收益。返回逐股票、逐持有期的收益、基准与超额，和每个基准的收益。"""
    from quantmind.line_a.sim_account import limit_pct, limit_price
    m = market()
    dates = m["dates"]
    T = pd.Timestamp(d)
    after = dates[dates > T]
    meta = {"as_of": d, "convention": "T+1 开盘买入（复权），持有 h 个交易日按第 h 日收盘计；入场日停牌或开盘即涨停记为无法成交",
            "last_bar": str(dates[-1].date()), "horizons": list(horizons)}
    if len(after) == 0:
        empty = {f"{k}_{h}": None for h in horizons for k in ("ret", "excess_pool", "excess_csi2000")}
        return {"meta": meta | {"entry_date": None, "note": "还没有 T+1 的行情"},
                "rows": [{"ticker": t, "status": "pending", "entry_price_raw": None, **empty} for t in dict.fromkeys(tickers)],
                "benchmarks": {str(h): {"exit_date": None} for h in horizons}}
    e = after[0]
    meta["entry_date"] = str(e.date())
    st = _is_st(d)
    adj = m["adj"]
    o_adj = m["open"] * adj
    c_adj = (m["close"] * adj).ffill()                       # 持有期内停牌：沿用最后一个收盘
    exits = {h: (after[h - 1] if len(after) >= h else None) for h in horizons}

    def entry_status(t: str) -> str:
        if t not in m["open"].columns or not (m["vol"].at[e, t] > 0) or not np.isfinite(m["open"].at[e, t]):
            return "suspended_at_entry"
        pre = m["pre_close"].at[e, t]
        up = limit_price(pre, limit_pct(t, e, t in st), True) if np.isfinite(pre) else math.inf
        if m["open"].at[e, t] >= up - 1e-9:
            return "limit_up_at_open"
        return "filled"

    def ret(t: str, h: int) -> float | None:
        x = exits[h]
        if x is None or t not in o_adj.columns:
            return None
        a, b = o_adj.at[e, t], c_adj.at[x, t]
        return float(b / a - 1) if np.isfinite(a) and np.isfinite(b) and a > 0 else None

    bench = {}
    pool = pool if pool is not None else research_pool(d)
    pool_ok = [t for t in pool if t in m["open"].columns and entry_status(t) == "filled"]
    for h in horizons:
        b = {}
        if exits[h] is not None:
            rs = [r for t in pool_ok if (r := ret(t, h)) is not None]
            b["pool_ew"] = float(np.mean(rs)) if rs else None
            for k, (code, _) in INDEXES.items():
                try:
                    b[k] = float(m["idx_close"].at[exits[h], code] / m["idx_open"].at[e, code] - 1)
                except (KeyError, ValueError):
                    b[k] = None
        bench[str(h)] = {"exit_date": str(exits[h].date()) if exits[h] is not None else None, **b}
    rows = []
    for t in dict.fromkeys(tickers):
        s = entry_status(t)
        r = {"ticker": t, "status": s, "entry_price_raw": None if s == "suspended_at_entry" else float(m["open"].at[e, t])}
        for h in horizons:
            v = ret(t, h) if s == "filled" else None
            bh = bench[str(h)]
            r[f"ret_{h}"] = v
            r[f"excess_pool_{h}"] = v - bh["pool_ew"] if v is not None and bh.get("pool_ew") is not None else None
            r[f"excess_csi2000_{h}"] = v - bh["csi2000"] if v is not None and bh.get("csi2000") is not None else None
        rows.append(r)
    meta["benchmark_names"] = {"pool_ew": f"研究池等权（{len(pool_ok)} 只可成交）"} | {k: v[1] for k, v in INDEXES.items()}
    return {"meta": meta, "rows": rows, "benchmarks": bench}


# ---------------------------------------------------------------- 红队
def _brain_flags(d: str) -> dict[str, str]:
    p = OUT / "research/daily" / d / "brain_memo.json"
    if not p.exists():
        return {}
    memo = json.loads(p.read_text(encoding="utf-8"))
    return {f["ticker"]: f.get("action", "") for f in memo.get("flags", [])}


RT_RULES_V2 = {
    "brain_avoid_new_buy": RT_RULES["brain_avoid_new_buy"], "brain_caution": RT_RULES["brain_caution"],
    "bypass_major": RT_RULES["bypass_major"],
    "beta_top10": "beta_252 在当日风险卡截面的前 10%",
    "vol_top10": "60 日年化波动在当日风险卡截面的前 10%",
    "limit_down_top10": "近 250 日跌停收盘次数在当日截面前 10%（且 ≥ 2 次）",
}


def redteam_rules(d: str, tickers: list[str], version: str = "v2") -> dict[str, list[str]]:
    """v1：固定阈值（前端原实现）；v2：当日截面前 10%（v1 在 ENS 选出的高波动小盘股里会标记一大半，没有区分度）。"""
    o = day_outputs(d)
    hits: dict[str, list[str]] = {t: [] for t in tickers}
    for t, a in _brain_flags(d).items():
        if t in hits:
            hits[t].append("brain_avoid_new_buy" if a == "avoid_new_buy" else "brain_caution")
    if "bypass" in o:
        b = pd.read_parquet(o["bypass"])
        for t in b.loc[b["importance"] == "major", "ticker"]:
            if t in hits:
                hits[t].append("bypass_major")
    if "f5c" in o:
        k = pd.read_parquet(o["f5c"]).set_index("ticker")
        if version == "v1":
            th = {"beta_252": (2.5, "beta_gt_2.5"), "vol_60_ann": (0.7, "vol_gt_70pct"),
                  "limit_down_close_250": (1.999, "limit_down_ge_2")}
        else:
            th = {"beta_252": (k["beta_252"].quantile(0.9), "beta_top10"),
                  "vol_60_ann": (k["vol_60_ann"].quantile(0.9), "vol_top10"),
                  "limit_down_close_250": (max(1.999, k["limit_down_close_250"].quantile(0.9)), "limit_down_top10")}
        for t in tickers:
            if t not in k.index:
                continue
            r = k.loc[t]
            for col, (lim, rule) in th.items():
                if pd.notna(r.get(col)) and r[col] > lim:
                    hits[t].append(rule)
    return hits


def subject_tickers(d: str, source: str, top: int = 100) -> list[str]:
    """红队 / 事后收益评估的对象：formal（正式组合的委托与持仓）、f4（研究候选，按优先级）、f3top（ENS 20 日前 top）。"""
    if source == "formal":
        p = OUT / "formal" / d / "report.json"
        if not p.exists():
            return []
        rep = json.loads(p.read_text(encoding="utf-8"))
        ts = []
        for tier in rep.get("tiers", {}).values():
            ts += [o["ticker"] for o in tier.get("orders_next_open", [])] + [h["ticker"] for h in tier.get("holdings", [])]
        return list(dict.fromkeys(ts))
    o = day_outputs(d)
    if source == "f4":
        c = pd.read_parquet(o["f4"])
        c = c[c["horizon"] == "20d"].sort_values(["priority", "ens20"], ascending=[True, False])
        return c["ticker"].head(top).tolist()
    s = pd.read_parquet(o["ens_20d"]).sort_values("ENS", ascending=False)
    pool = set(research_pool(d))
    return [t for t in s["ticker"] if t in pool][:top]


def redteam(d: str, source: str = "f4", horizon: int = 5, top: int = 100, version: str = "v2") -> dict:
    ts = subject_tickers(d, source, top)
    hits = redteam_rules(d, ts, version)
    rules = RT_RULES if version == "v1" else RT_RULES_V2
    fw = forward(ts, d, horizons=(horizon,))
    rows = []
    for r in fw["rows"]:
        rows.append(r | {"rules": hits.get(r["ticker"], [])})

    def stats(sub, key):
        v = [x[key] for x in sub if x[key] is not None]
        return {"n": len(v), "mean": float(np.mean(v)) if v else None,
                "share_below_0": float(np.mean([x < 0 for x in v])) if v else None}
    flagged = [r for r in rows if r["rules"]]
    clean = [r for r in rows if not r["rules"]]
    o = day_outputs(d)
    inputs = {"bypass": "bypass" in o, "f5c": "f5c" in o, "brain": bool(_brain_flags(d))}
    out = {"meta": fw["meta"] | {"source": source, "horizon": horizon, "rules": rules, "rule_version": version,
                                 "inputs_available": inputs,
                                 "n_subjects": len(rows), "flagged_share": len(flagged) / max(1, len(rows)),
                                 "definition": "命中 = 被红队规则标记的股票，事后相对基准跑输（超额 < 0）；"
                                               "和未标记股票的跑输比例（base rate）比较，差值才是规则的信息量"},
           "benchmarks": fw["benchmarks"], "rows": rows, "by_group": {}}
    for key, name in ((f"ret_{horizon}", "absolute"), (f"excess_pool_{horizon}", "excess_pool"),
                      (f"excess_csi2000_{horizon}", "excess_csi2000")):
        f, c = stats(flagged, key), stats(clean, key)
        lift = (f["share_below_0"] - c["share_below_0"]) if f["share_below_0"] is not None and c["share_below_0"] is not None else None
        out["by_group"][name] = {"flagged": f, "unflagged": c, "hit_rate": f["share_below_0"],
                                 "base_rate": c["share_below_0"], "lift": lift,
                                 "mean_gap": (f["mean"] - c["mean"]) if f["mean"] is not None and c["mean"] is not None else None}
    by_rule = {}
    for rule in rules:
        sub = [r for r in rows if rule in r["rules"]]
        by_rule[rule] = stats(sub, f"excess_pool_{horizon}")
    out["by_rule"] = by_rule
    missing = [k for k, v in inputs.items() if not v]
    out["missing_inputs"] = missing          # 这一天没有的规则输入（早期运行日没有 bypass / f5c，大脑从 10-23 起才有）
    out["warning"] = "样本少于 30 只时不下结论" if len([x for x in flagged if x[f"excess_pool_{horizon}"] is not None]) < 30 else None
    return out


# ---------------------------------------------------------------- 模型共识
MODELS = ("xgb", "cat", "tabm", "seqfusion")


@lru_cache(maxsize=8)
def _f3_pct(path: str, mtime: float, pool_key: str) -> pd.DataFrame:
    s = pd.read_parquet(path).set_index("ticker")
    pool = set(pool_key.split(",")) if pool_key else None
    if pool:
        s = s[s.index.isin(pool)]
    cols = [c for c in (*MODELS, "ENS") if c in s]
    p = s[cols].rank(pct=True) * 100
    p.columns = [f"{c}_pct" for c in cols]
    mp = p[[f"{c}_pct" for c in MODELS if f"{c}_pct" in p]]
    p["dispersion_pp"] = mp.std(axis=1, ddof=0)               # 四个模型百分位的标准差（百分点）
    p["range_pp"] = mp.max(axis=1) - mp.min(axis=1)
    p["dispersion_rank_pct"] = p["dispersion_pp"].rank(pct=True) * 100
    return pd.concat([s[cols], p], axis=1)


def consensus(d: str, ticker: str | None = None, horizon: str = "20d", universe: str = "research_pool") -> dict:
    o = day_outputs(d)
    path = o[f"ens_{horizon}"]
    pool = research_pool(d) if universe == "research_pool" else []
    t = _f3_pct(path, Path(path).stat().st_mtime, ",".join(pool))
    mp = t[[c for c in MODELS if c in t]]
    meta = {"as_of": d, "horizon": horizon, "universe": universe, "n": len(t),
            "score_scale": "F3 分数是截面 z 分（均值 0、标准差约 1），这里换成截面百分位（0–100，越高越看好）",
            "dispersion": "四个模型百分位的标准差；dispersion_rank_pct = 它在同一截面里的分位（高 = 模型分歧大）",
            "spearman": mp.corr(method="spearman").round(3).to_dict()}
    if ticker is None:
        return {"meta": meta}
    if ticker not in t.index:
        return {"meta": meta, "row": None, "note": f"{ticker} 不在 {universe} 截面里"}
    r = t.loc[ticker]
    disp = float(r["dispersion_rank_pct"])
    label = "一致看好" if r.get("ENS_pct", 0) >= 80 and disp <= 50 else ("模型分歧" if disp >= 80 else "温和一致")
    return {"meta": meta, "row": {"ticker": ticker, **{k: (None if pd.isna(v) else float(v)) for k, v in r.items()},
                                  "label": label}}


# ---------------------------------------------------------------- F7 核验溯源
def _role_file(d: str, ticker: str, role: str) -> Path | None:
    fs = sorted((OUT / "research/daily" / d / "cases").glob(f"*/{ticker}/{role}_*.json"), key=lambda p: p.stat().st_mtime)
    return fs[-1] if fs else None


def _slim_record(rec: dict, fields: set[str]) -> dict:
    keep = {"evidence_id", "end_date", "ann_date", "title", "type", "trade_date", "event_type", "stage", "available_at"}
    out = {k: rec[k] for k in keep if k in rec}
    for f in fields:
        if f and f in rec:
            out[f] = rec[f]
    if rec.get("_units"):
        out["_units"] = {k: v for k, v in rec["_units"].items() if k in fields}
    if rec.get("_display"):
        out["_display"] = {k: v for k, v in rec["_display"].items() if k in fields}
    txt = rec.get("text") or rec.get("para") or rec.get("content")
    if isinstance(txt, str):
        out["text_excerpt"] = txt[:600]
    return out


def claim_trace(d: str, ticker: str, ref: str) -> dict:
    from dataclasses import asdict

    from quantmind.line_a.harness.verifier import _clean_formula, _eval, verify
    role, _, lid = ref.partition(":")
    p = _role_file(d, ticker, role)
    if p is None:
        raise FileNotFoundError(f"{role} run file for {ticker} on {d}")
    res = json.loads(p.read_text(encoding="utf-8"))
    out = res.get("output") or {}
    claims = {c["local_id"]: c for c in out.get("claims", [])}
    if lid not in claims:
        raise KeyError(ref)
    c = claims[lid]
    ev = res.get("evidence") or {}
    stored = next((x for x in res.get("checks", []) if x.get("local_id") == lid), None)
    now = next((asdict(x) for x in verify(out, ev) if x.local_id == lid), None)     # 用现在的校验器重跑一遍
    cited = []
    for r in c.get("evidence") or []:
        rec = ev.get(r.get("evidence_id"))
        f = r.get("field")
        cited.append({"ref": r, "found": rec is not None,
                      "record": _slim_record(rec, {f}) if rec else None,
                      "record_value": rec.get(f) if rec and f else None,
                      "record_unit": ((rec or {}).get("_units") or {}).get(f)})
    inputs = [{"local_id": i, "claim_type": claims[i].get("claim_type"), "text": claims[i].get("text"),
               "value": claims[i].get("value"), "unit": claims[i].get("unit"),
               "check": next((x.get("status") for x in res.get("checks", []) if x.get("local_id") == i), None)}
              for i in c.get("inputs") or [] if i in claims]
    recompute = None
    if c.get("claim_type") == "computed" and c.get("formula"):
        env = {i["local_id"]: i["value"] for i in inputs}
        try:
            v = _eval(_clean_formula(c["formula"]), {k: float(x) for k, x in env.items() if x is not None})
            rel = abs(float(c["value"]) - v) / max(abs(v), 1e-12) if c.get("value") is not None else None
            recompute = {"formula": c["formula"], "env": env, "value": v, "stated": c.get("value"), "rel_diff": rel}
        except Exception as e:  # noqa: BLE001
            recompute = {"formula": c["formula"], "env": env, "error": f"{type(e).__name__}: {e}"}
    return {"as_of": d, "ticker": ticker, "ref": ref, "run_file": p.name, "model": res.get("model"),
            "claim": c, "check_at_run": stored, "check_now": now, "cited": cited, "inputs": inputs,
            "recompute": recompute,
            "verifier": "quantmind/line_a/harness/verifier.py（代码校验，不由模型兼任）"}


# ---------------------------------------------------------------- 漏斗
def funnel(d: str) -> dict:
    o = day_outputs(d)
    layers = []
    f1 = pd.read_parquet(o["f1"])
    layers.append({"layer": "F1", "name": "证券身份与可交易性", "n": len(f1),
                   "detail": {"ST": int(f1["is_st"].sum()), "停牌": int(f1["suspended"].sum()),
                              "上市不足": int(f1["c_listed_lt_min"].sum()), "低价": int(f1["c_low_price"].sum()),
                              "可新买（主板 + 创业板）": int(f1["new_buy_no_bse"].sum()) if "new_buy_no_bse" in f1 else None}})
    f2 = pd.read_parquet(o["f2"])
    layers.append({"layer": "F2", "name": "支持域", "n": int((f2["domain"] == "research_pool").sum()),
                   "detail": {k: int(v) for k, v in f2["domain"].value_counts().items()}})
    s = pd.read_parquet(o["ens_20d"])
    layers.append({"layer": "F3", "name": "量化打分（ENS 20 日）", "n": len(s), "detail": {}})
    c = pd.read_parquet(o["f4"])
    layers.append({"layer": "F4", "name": "研究候选", "n": int(c["ticker"].nunique()),
                   "detail": {k: int(v) for k, v in c["horizon"].value_counts().items()}})
    if "f5c" in o:
        layers.append({"layer": "F5", "name": "风险与执行卡", "n": len(pd.read_parquet(o["f5c"], columns=["ticker"])), "detail": {}})
    if "bypass" in o:
        b = pd.read_parquet(o["bypass"])
        layers.append({"layer": "旁路", "name": "全市场事件监测", "n": len(b),
                       "detail": {k: int(v) for k, v in b["importance"].value_counts().items()}})
    day = OUT / "research/daily" / d
    if (day / "f6_cases.parquet").exists():
        cases = pd.read_parquet(day / "f6_cases.parquet")
        layers.append({"layer": "F6", "name": "Agent 研究案例", "n": len(cases), "detail": {}})
    if (day / "f7_claims.parquet").exists():
        cl = pd.read_parquet(day / "f7_claims.parquet")
        vc = {k: int(v) for k, v in cl["check"].value_counts().items()}
        layers.append({"layer": "F7", "name": "代码核验的声明", "n": len(cl),
                       "detail": vc | {"每案例平均": round(len(cl) / max(1, cl["ticker"].nunique()), 1),
                                       "已核实比例": round(vc.get("verified", 0) / max(1, len(cl)), 4)}})
    p = OUT / "formal" / d / "report.json"
    if p.exists():
        rep = json.loads(p.read_text(encoding="utf-8"))
        layers.append({"layer": "正式", "name": "正式组合委托",
                       "n": sum(len(t.get("orders_next_open", [])) for t in rep.get("tiers", {}).values()),
                       "detail": {k: len(t.get("orders_next_open", [])) for k, t in rep.get("tiers", {}).items()}})
    return {"as_of": d, "layers": layers}


# ---------------------------------------------------------------- 模型事后 IC
def forward_ic(horizon: int = 5, model: str = "ens_20d") -> dict:
    """每个运行日：ENS 分数与事后 h 日收益（T+1 开盘入场）在研究池里的 Spearman IC，和前 10% 减研究池等权。"""
    rows = []
    for dd in sorted(p.name for p in (OUT / "runs").iterdir() if p.is_dir() and p.name[:2] == "20"):
        try:
            o = day_outputs(dd)
            s = pd.read_parquet(o[model]).set_index("ticker")
            col = "ENS" if "ENS" in s else "score"
            pool = research_pool(dd)
        except Exception:  # noqa: BLE001
            continue
        fw = forward([t for t in pool if t in s.index], dd, horizons=(horizon,), pool=pool)
        if fw["meta"].get("entry_date") is None or fw["benchmarks"][str(horizon)]["exit_date"] is None:
            rows.append({"as_of": dd, "ic": None, "n": 0, "note": "未到期"})
            continue
        r = pd.DataFrame(fw["rows"]).set_index("ticker")[f"ret_{horizon}"].dropna()
        sc = s.loc[r.index, col]
        top = sc[sc >= sc.quantile(0.9)].index
        rows.append({"as_of": dd, "ic": float(sc.rank().corr(r.rank())), "n": int(len(r)),
                     "top10_minus_pool": float(r.loc[top].mean() - r.mean()),
                     "exit_date": fw["benchmarks"][str(horizon)]["exit_date"]})
    v = [x["ic"] for x in rows if x.get("ic") is not None]
    return {"horizon": horizon, "model": model, "rows": rows,
            "mean_ic": float(np.mean(v)) if v else None, "n_dates": len(v),
            "note": "日期之间的持有期重叠，IC 不独立；只作跟踪，不作检验"}


def execution_rules() -> dict:
    import yaml

    from quantmind.backtest.wf_costs import stamp_duty_rate
    from quantmind.line_a.sim_account import LOT
    REPO = Path(__file__).resolve().parents[2]
    cfg = yaml.safe_load((REPO / "configs/line_a_v2/formal.yaml").read_text(encoding="utf-8"))
    return {"execution": cfg.get("execution", {}), "stamp_duty_sell": stamp_duty_rate(pd.Timestamp.today()),
            "lot": {k: {"min": v[0], "step": v[1]} for k, v in LOT.items()},
            "tiers": {k: {"aum": v["aum"], "top_n": v["top_n"], "boards": v["boards"]} for k, v in cfg["tiers"].items()},
            "source": "configs/line_a_v2/formal.yaml + quantmind/line_a/sim_account.py（T-052 模拟器同一套规则）"}
