"""quantmind.line_a.f6_research — 每日影子研究：F6 Agent 研究、F7 证据验收、F9 研究报告（03_final_plan_v2 第 5 节；T-040）.

规则在 T-040 登记（`f6-select-v1-2026-10-23`），这里只实现：
- 选谁：持仓 × 重大事件 → 其余重大事件（在宽候选里）→ 排名靠前（20 日两模型平均百分位、63 日 Ridge，2 : 1 交替）
  → 抽查样本；每天最多 max_cases 个案例；
- 跑哪些角色：10 个交易日内没研究过就跑 A1–A4 + A6；研究过只跑输入有变化的角色（见 triggers），沿用的角色交给 A6 一起审查；
- 降级：网络被解密、预算用完、墙钟到时就不再开新案例，已完成的照常写出（状态 partial）。
只读影子：研究结果只进入影子研究报告，不改变任何账本、排名或权重。
产物：<out_root>/research/daily/<as_of>/：cases/（各角色与案例 JSON）、f6_cases.parquet、f7_claims.parquet、
     f7_risk_flags.parquet、f9_research.md、status.json。重跑同一天时，已有案例文件的股票直接沿用（断点续跑）。
"""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from quantmind.line_a.atomic import atomic_write_json

RULE_VERSION = "f6-select-v1-2026-10-23"
CFG = {"max_cases": 40, "budget_rmb": 15.0, "per_case_rmb": 1.5, "workers": 4, "fresh_tdays": 10,
       "a4_refresh_tdays": 5, "max_wall_s": 9000, "cap_holding_event": 15, "cap_event": 10, "n_sample": 2}
ROLES = ("A1", "A2", "A3", "A4")
A3_TYPES = {"holder_reduction", "holder_increase", "repurchase", "unlock", "pledge", "control_change",
            "equity_incentive", "private_placement", "restructuring"}
FIN_TABLES = ("income", "balancesheet", "cashflow", "fina_indicator_full")
A1_EVENTS = ("forecast", "express", "fina_audit")
A3_EVENTS = (("stk_holdertrade", "ann_date"), ("share_float", "ann_date"), ("repurchase", "ann_date"),
             ("pledge_stat", "end_date"))
CASE_COLS = ["ticker", "category", "roles_run", "roles_reuse", "as_of", "cost_rmb", "role_paths", "role_as_of_map",
             "case_path", "error", "status"]


def ymd(d) -> str:
    return pd.Timestamp(d).strftime("%Y%m%d")


def open_days(lake: Path) -> pd.DatetimeIndex:
    cal = pd.read_parquet(Path(lake) / "trade_cal_sse.parquet")
    if "is_open" in cal:
        cal = cal[cal["is_open"].astype(int) == 1]
    return pd.DatetimeIndex(pd.to_datetime(cal["cal_date"].astype(str))).sort_values()


def tdays_between(days: pd.DatetimeIndex, a, b) -> int:
    """a 之后到 b（含）之间的交易日数。"""
    return int(((days > pd.Timestamp(a)) & (days <= pd.Timestamp(b))).sum())


# ------------------------------------------------------------------ 历史：每只股票、每个角色最近一次成功的研究
def history(out_root: Path, as_of) -> pd.DataFrame:
    """读 as_of 之前各天的 f6_cases.parquet，展开成 (ticker, role, as_of, path)。"""
    rows = []
    base = Path(out_root) / "research/daily"
    for f in sorted(base.glob("*/f6_cases.parquet")):
        if f.parent.name >= str(pd.Timestamp(as_of).date()):
            continue
        df = pd.read_parquet(f)
        for r in df[df["status"] == "ok"].itertuples():
            amap = json.loads(r.role_as_of_map or "{}")
            for role, p in json.loads(r.role_paths).items():
                rows.append({"ticker": r.ticker, "role": role, "as_of": amap.get(role, r.as_of), "path": p})
    cols = ["ticker", "role", "as_of", "path"]
    h = pd.DataFrame(rows, columns=cols)
    return h.sort_values("as_of").drop_duplicates(["ticker", "role"], keep="last") if len(h) else h


# ------------------------------------------------------------------ 输入有没有变化
def _changed(path: Path, col: str, tickers: set[str], lo: str, hi: str) -> set[str]:
    import pyarrow.dataset as ds
    if not path.exists() or not tickers:
        return set()
    dset = ds.dataset(str(path), format="parquet")
    if col not in dset.schema.names:
        return set()
    typ = str(dset.schema.field(col).type)
    conv = (lambda s: pd.Timestamp(s)) if typ.startswith(("timestamp", "date")) else \
        ((lambda s: int(s)) if typ.startswith("int") else (lambda s: s))
    t = dset.to_table(columns=["ts_code", col], filter=(ds.field(col) > conv(lo)) & (ds.field(col) <= conv(hi)))
    got = t.column("ts_code").to_pylist()
    return {x for x in got if x in tickers}


def _latest_event(lake: Path, api: str) -> Path | None:
    fs = sorted((Path(lake) / "events").glob(f"{api}_2*.parquet"))
    return fs[-1] if fs else None


def cninfo_types(lake: Path, tickers: set[str], lo: str, hi: str) -> dict[str, set[str]]:
    """(lo, hi] 自然日内的巨潮公告，按标题规则分出 major / watch 的事件类型。"""
    from quantmind.line_a.event_monitor import cninfo_ts_code
    from quantmind.line_a.evidence.ann_types import classify_title
    out: dict[str, set[str]] = {}
    parts = Path(lake) / "text/cninfo/_parts"
    days = pd.date_range(pd.Timestamp(lo) + pd.Timedelta(days=1), pd.Timestamp(hi), freq="D")
    for d in days:
        f = parts / f"{d:%Y%m%d}.parquet"
        if not f.exists():
            continue
        df = pd.read_parquet(f, columns=["secCode", "announcementTitle"])
        for code, title in zip(df["secCode"], df["announcementTitle"], strict=True):
            t = cninfo_ts_code(code)
            if t not in tickers:
                continue
            c = classify_title(str(title))
            if c.importance != "routine" and not c.is_summary_or_meta and c.event_type != "other":
                out.setdefault(t, set()).add(c.event_type)
    return out


def triggers(lake: Path, as_of, last: dict[str, dict[str, str]], days: pd.DatetimeIndex) -> dict[str, set[str]]:
    """last：{ticker: {role: 上次研究的 as_of (YYYYMMDD)}}。返回每只股票需要重跑的角色（不含 A6）。"""
    hi = ymd(as_of)
    need: dict[str, set[str]] = {t: set() for t in last}
    if not last:
        return need
    for role in ROLES:
        group: dict[str, set[str]] = {}
        for t, rs in last.items():
            if role not in rs:
                need[t].add(role)                     # 以前没跑过这个角色
            else:
                group.setdefault(rs[role], set()).add(t)
        for lo, ts in group.items():
            hit: set[str] = set()
            if role == "A1":
                for tb in FIN_TABLES:
                    hit |= _changed(Path(lake) / f"{tb}.parquet", "ann_date", ts, lo, hi)
                for api in A1_EVENTS:
                    if (p := _latest_event(lake, api)) is not None:
                        hit |= _changed(p, "ann_date", ts, lo, hi)
            elif role == "A2":
                hit |= set(cninfo_types(lake, ts, lo, hi))
            elif role == "A3":
                for api, col in A3_EVENTS:
                    if (p := _latest_event(lake, api)) is not None:
                        hit |= _changed(p, col, ts, lo, hi)
                hit |= {t for t, ty in cninfo_types(lake, ts, lo, hi).items() if ty & A3_TYPES}
            elif role == "A4":
                from quantmind.line_a.harness.tools_more import EXTENDED_RC
                hit |= _changed(Path(lake) / EXTENDED_RC, "report_date", ts, lo, hi)
                if tdays_between(days, lo, hi) >= CFG["a4_refresh_tdays"]:
                    hit |= ts
            for t in hit:
                need[t].add(role)
    return need


# ------------------------------------------------------------------ 选谁
def _consensus(g: pd.DataFrame, cols: list[str]) -> pd.Series:
    r = pd.concat([g[c].rank(pct=True) for c in cols if c in g], axis=1)
    return r.mean(axis=1, skipna=True)


def select(as_of, f4: pd.DataFrame, bypass: pd.DataFrame, holdings: set[str], hist: pd.DataFrame,
           days: pd.DatetimeIndex, cfg: dict = CFG) -> pd.DataFrame:
    """返回 ticker、category（holding_event / event / top_20d / top_63d / sample）、fresh、last（{role: as_of}）。"""
    last_any = hist.groupby("ticker")["as_of"].max().to_dict() if len(hist) else {}

    def fresh(t: str) -> bool:
        return t in last_any and tdays_between(days, last_any[t], as_of) < cfg["fresh_tdays"]

    major = bypass[bypass["importance"] == "major"].sort_values("n_events", ascending=False) if len(bypass) else bypass
    chosen: dict[str, str] = {}
    for t in [t for t in major["ticker"] if t in holdings][:cfg["cap_holding_event"]]:
        chosen.setdefault(t, "holding_event")
    in_f4 = set(f4["ticker"])
    for t in [t for t in major["ticker"] if t in in_f4 and t not in chosen][:cfg["cap_event"]]:
        chosen.setdefault(t, "event")
    g20 = f4[(f4["horizon"] == "20d") & (f4["domain"] == "research_pool")].copy()
    l20 = list(g20.assign(s=_consensus(g20, ["ens20", "ridge20"])).sort_values("s", ascending=False)["ticker"])
    g63 = f4[(f4["horizon"] == "63d") & (f4["domain"] == "research_pool")]
    l63 = list(g63.sort_values("ridge63", ascending=False)["ticker"]) if "ridge63" in g63 else []
    i20 = i63 = 0
    while len(chosen) < cfg["max_cases"] - cfg["n_sample"] and (i20 < len(l20) or i63 < len(l63)):
        for _ in range(2):
            while i20 < len(l20) and (l20[i20] in chosen or fresh(l20[i20])):
                i20 += 1
            if i20 < len(l20) and len(chosen) < cfg["max_cases"] - cfg["n_sample"]:
                chosen[l20[i20]] = "top_20d"
        while i63 < len(l63) and (l63[i63] in chosen or fresh(l63[i63])):
            i63 += 1
        if i63 < len(l63) and len(chosen) < cfg["max_cases"] - cfg["n_sample"]:
            chosen[l63[i63]] = "top_63d"
    smp = sorted({t for t, r in zip(f4["ticker"], f4["reasons"], strict=True) if "stratified_sample" in str(r)}
                 - set(chosen))
    smp = [t for t in smp if not fresh(t)]
    rng = np.random.default_rng(int(ymd(as_of)))
    for t in (rng.choice(smp, size=min(cfg["n_sample"], len(smp)), replace=False) if smp else []):
        chosen[str(t)] = "sample"
    last = {t: dict(zip(g["role"], g["as_of"], strict=True)) for t, g in hist.groupby("ticker")} if len(hist) else {}
    return pd.DataFrame([{"ticker": t, "category": c, "fresh": fresh(t), "last": json.dumps(last.get(t, {}))}
                         for t, c in chosen.items()])


def plan_roles(sel: pd.DataFrame, need: dict[str, set[str]]) -> pd.DataFrame:
    """不新鲜：A1–A4 全跑；新鲜：只跑 need 里的角色；都不用跑就记为沿用。"""
    run, reuse = [], []
    for t, fr, last in zip(sel["ticker"], sel["fresh"], sel["last"], strict=True):
        prev = json.loads(last)
        rr = sorted(need.get(t, set())) if fr else list(ROLES)
        run.append(",".join(rr))
        reuse.append(",".join(r for r in ROLES if r not in rr and r in prev))
    return sel.assign(roles_run=run, roles_reuse=reuse)


# ------------------------------------------------------------------ 运行
def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_research(data_root: Path, out_root: Path, as_of, f4: pd.DataFrame, bypass: pd.DataFrame,
                 holdings: set[str], *, cfg: dict = CFG, llm_factory=None, case_runner=None, log=print) -> dict:
    """llm_factory(ticker) → LLM；case_runner 默认 slices.run_case（测试可注入）。"""
    from quantmind.line_a.harness.llm import LLM, BudgetExceededError, TLSInterceptionError
    from quantmind.line_a.harness.slices import run_case
    case_runner = case_runner or run_case
    lake = Path(data_root) / "data/lake"
    day = Path(out_root) / "research/daily" / str(pd.Timestamp(as_of).date())
    day.mkdir(parents=True, exist_ok=True)
    days = open_days(lake)
    hist = history(out_root, as_of)
    sel = select(as_of, f4, bypass, holdings, hist, days, cfg)
    last = {t: json.loads(x) for t, x in zip(sel["ticker"], sel["last"], strict=True) if json.loads(x)}
    plan = plan_roles(sel, triggers(lake, as_of, last, days))
    plan.to_parquet(day / "f6_plan.parquet")
    if llm_factory is None:
        cache = Path(out_root) / "research/_llm_cache"

        def llm_factory(t):
            return LLM(ledger=day / "ledger" / f"{t}.jsonl", cache_dir=cache, budget_rmb=cfg["per_case_rmb"])
    t0, lock, stop = time.time(), threading.Lock(), {"reason": None}
    spent = {"rmb": 0.0}
    hist_paths = {(r.ticker, r.role): (r.path, r.as_of) for r in hist.itertuples()} if len(hist) else {}

    def one(row) -> dict:
        t = row.ticker
        base = {"ticker": t, "category": row.category, "roles_run": row.roles_run, "roles_reuse": row.roles_reuse,
                "as_of": ymd(as_of), "cost_rmb": 0.0, "role_paths": "{}", "role_as_of_map": "{}", "case_path": None,
                "error": None}
        done = sorted((day / "cases" / ymd(as_of) / t).glob("case_*.json"))
        if done:                                              # 断点续跑：今天已经完成的案例
            case = _load(str(done[-1]))
            return base | _case_fields(case, str(done[-1])) | {"status": "ok", "resumed": True}
        if not row.roles_run:
            return base | {"status": "reused"}
        with lock:
            if stop["reason"] is None and time.time() - t0 > cfg["max_wall_s"]:
                stop["reason"] = "wall clock"
            if stop["reason"] is None and spent["rmb"] + cfg["per_case_rmb"] > cfg["budget_rmb"]:
                stop["reason"] = "daily budget"
            if stop["reason"] is not None:
                return base | {"status": "not_started", "error": stop["reason"]}
        reuse = {}
        for r in row.roles_reuse.split(","):
            if r and (t, r) in hist_paths:
                p, _ = hist_paths[(t, r)]
                if Path(p).exists():
                    reuse[r] = _load(p) | {"path": p}          # 角色文件里不含自己的路径，沿用时补上
        try:
            case = case_runner(llm_factory(t), lake, ymd(as_of), t, row.roles_run.split(","), day / "cases", reuse=reuse)
        except TLSInterceptionError as e:
            with lock:
                stop["reason"] = "TLS interception"
            return base | {"status": "error", "error": str(e)[:200]}
        except BudgetExceededError as e:
            return base | {"status": "budget", "error": str(e)[:200]}
        except Exception as e:  # noqa: BLE001  单个案例失败不影响其他案例
            return base | {"status": "error", "error": f"{type(e).__name__}: {str(e)[:200]}"}
        with lock:
            spent["rmb"] += float(case.get("cost_rmb") or 0.0)
        log(f"  case {t} {row.category} run={row.roles_run} reuse={row.roles_reuse} ¥{case.get('cost_rmb', 0):.3f}")
        return base | _case_fields(case, case.get("path")) | {"status": "ok"}

    with ThreadPoolExecutor(max_workers=cfg["workers"]) as ex:
        results = list(ex.map(one, plan.itertuples(index=False)))
    cases = pd.DataFrame(results, columns=CASE_COLS) if not results else pd.DataFrame(results)
    cases.to_parquet(day / "f6_cases.parquet")
    claims, flags = f7_tables(cases)
    claims.to_parquet(day / "f7_claims.parquet")
    flags.to_parquet(day / "f7_risk_flags.parquet")
    n_ok = int((cases["status"] == "ok").sum())
    status = {"as_of": str(pd.Timestamp(as_of).date()), "rule_version": RULE_VERSION, "cfg": cfg,
              "n_selected": int(len(plan)), "n_ok": n_ok, "n_reused_only": int((cases["status"] == "reused").sum()),
              "by_status": cases["status"].value_counts().to_dict(), "cost_rmb": round(float(cases["cost_rmb"].sum()), 4),
              "stopped": stop["reason"], "wall_s": round(time.time() - t0, 1),
              "state": "ok" if stop["reason"] is None and not (cases["status"].isin(["error", "budget"])).any()
              else ("unavailable" if n_ok == 0 and len(plan) else "partial")}
    atomic_write_json(day / "status.json", status)
    return {"dir": str(day), "status": status, "cases": cases, "claims": claims, "flags": flags}


def _case_fields(case: dict, path: str | None) -> dict:
    paths = {r: u["path"] for r, u in case["upstream"].items() if u.get("path")}
    if case.get("a6", {}).get("path"):
        paths["A6"] = case["a6"]["path"]
    as_of_map = {r: (u.get("as_of") or case["as_of"]) for r, u in case["upstream"].items()} | {"A6": case["as_of"]}
    return {"cost_rmb": float(case.get("cost_rmb") or 0.0), "role_paths": json.dumps(paths),
            "role_as_of_map": json.dumps(as_of_map), "case_path": path,
            "a6_verdicts": json.dumps(case.get("a6", {}).get("verdicts", {})),
            "check_status": json.dumps({r: u.get("check_status", {}) for r, u in case["upstream"].items()})}


# ------------------------------------------------------------------ F7 证据验收
def f7_tables(cases: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    from quantmind.line_a.harness.rewrite import skeleton
    claims, flags = [], []
    for r in cases[cases["status"] == "ok"].itertuples():
        paths = json.loads(r.role_paths)
        outs = {k: _load(p) for k, p in paths.items() if k != "A6" and Path(p).exists()}
        review = _load(paths["A6"]).get("output") if "A6" in paths and Path(paths["A6"]).exists() else None
        sk = skeleton(outs, review)
        col_of = {ref: col for col, refs in sk["columns"].items() for ref in refs}
        amap = json.loads(r.role_as_of_map)
        for ref, it in sk["claims"].items():
            claims.append({"ticker": r.ticker, "ref": ref, "role_as_of": amap.get(ref.split(":")[0]), **it,
                           "column": col_of.get(ref)})
        for role, res in outs.items():
            for f in (res.get("output") or {}).get("risk_flags", []):
                flags.append({"ticker": r.ticker, "role": role, "role_as_of": amap.get(role), "flag": f})
    cc = ["ticker", "ref", "role_as_of", "type", "text", "value", "unit", "period", "check", "a6", "column"]
    return pd.DataFrame(claims, columns=cc), pd.DataFrame(flags, columns=["ticker", "role", "role_as_of", "flag"])


# ------------------------------------------------------------------ F9 研究报告
def report(res: dict, names: dict[str, str] | None = None, rewrite_llm=None) -> str:
    from quantmind.line_a.harness.rewrite import skeleton, summarize, template_text
    st, cases = res["status"], res["cases"]
    summaries: list[dict] = []
    names = names or {}
    L = [f"# 影子研究报告 · {st['as_of']}", "",
         "> 研究用只读影子运行，不是投资建议。研究声明不改变任何账本、排名或权重。资金批准状态：NOT_APPROVED。", "",
         f"- 选择规则：`{st['rule_version']}`；选中 {st['n_selected']} 只，完成 {st['n_ok']} 个案例，"
         f"全部沿用 {st['n_reused_only']} 只；估算费用 ¥{st['cost_rmb']:.2f}；用时 {st['wall_s'] / 60:.0f} 分钟",
         f"- 运行状态：{st['state']}" + (f"（停止原因：{st['stopped']}）" if st["stopped"] else ""), ""]
    cat = {"holding_event": "持仓 × 重大事件", "event": "重大事件", "top_20d": "20 日排名靠前", "top_63d": "63 日排名靠前",
           "sample": "抽查样本"}
    def js(v) -> dict:
        return json.loads(v) if isinstance(v, str) and v else {}
    rows = []
    for r in cases.itertuples():
        chk = {}
        for v in js(getattr(r, "check_status", None)).values():
            for k, n in v.items():
                chk[k] = chk.get(k, 0) + n
        rows.append({"股票": r.ticker, "名称": names.get(r.ticker, ""), "进入原因": cat.get(r.category, r.category),
                     "重跑": r.roles_run or "—", "沿用": r.roles_reuse or "—", "状态": r.status,
                     "校验": "，".join(f"{k} {n}" for k, n in sorted(chk.items())) or "—",
                     "A6": "，".join(f"{k} {n}" for k, n in sorted(js(getattr(r, "a6_verdicts", None)).items())) or "—",
                     "费用": f"¥{r.cost_rmb:.2f}"})
    from quantmind.line_a.pipeline import _md_table
    L += ["## 案例一览", "", _md_table(pd.DataFrame(rows)), ""]
    for r in cases[cases["status"] == "ok"].itertuples():
        paths = json.loads(r.role_paths)
        amap = json.loads(r.role_as_of_map)
        outs = {k: _load(p) for k, p in paths.items() if k != "A6" and Path(p).exists()}
        review = _load(paths["A6"]).get("output") if "A6" in paths and Path(paths["A6"]).exists() else None
        sk = skeleton(outs, review)
        sm = summarize(rewrite_llm, sk)                       # F9 v2：模型只写要点，逐条核对
        tt = template_text(sk)                                 # 四栏事实由代码生成，数值原样
        old = [f"{k}（{v}）" for k, v in amap.items() if k != "A6" and v != st["as_of"].replace("-", "")]
        L += [f"## {r.ticker} {names.get(r.ticker, '')}（{cat.get(r.category, r.category)}）", "",
              f"要点：{len(sm['points'])} 条通过核对" + (f"，{len(sm['dropped'])} 条未通过已去掉" if sm["dropped"] else "")
              + (f"；沿用的角色：{'，'.join(old)}" if old else ""), ""]
        L += ["**研究要点**", ""] + ([f"- {p}" for p in sm["points"]] or ["（无）"]) + [""]
        titles = {"counter_and_limits": "反证与限制", "to_verify": "待核实", "facts": "事实", "inferences": "推断"}
        for k, title in titles.items():
            L += [f"**{title}**", "", tt.get(k) or "（无）", ""]
        summaries.append({"ticker": r.ticker, "n_points": len(sm["points"]), "n_dropped": len(sm["dropped"]),
                          "dropped": sm["dropped"], "mode": sm["mode"], "errors": sm["errors"],
                          "first_pass": sm.get("first_pass", [])})
        fl = res["flags"][res["flags"]["ticker"] == r.ticker]
        if len(fl):
            L += ["**风险事实（只是事实或待确认项，不扣分）**", ""] + [f"- {f}（{ro}）" for ro, f in zip(fl["role"], fl["flag"],
                                                                                           strict=True)] + [""]
    res["summaries"] = summaries
    return "\n".join(L)
