"""quantmind.line_a.api_server — 给个人展示前端用的只读 REST 接口（FastAPI，只监听 127.0.0.1）.

只读：所有接口都是 GET，只读文件，不触发任何计算、下载或模型调用；不返回密钥和账户信息。
启动：PYTHONPATH=. python -m quantmind.line_a.api_server --port 8780   （文档：http://127.0.0.1:8780/docs）
接口与数据的说明见 docs/handoff/frontend_handoff.md。
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

ROOT = Path("/home/lenovo/projects/quantmind")
OUT = ROOT / "data/line_a_v2"
REPO = Path(__file__).resolve().parents[2]
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
app = FastAPI(title="QuantMind 只读接口", version="1.0",
              description="主线 A v2 的每日产物、正式荐股日报、Agent 大脑意见、账本、研究与实验结果。全部只读。")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["GET"], allow_headers=["*"])


def _jsonable(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records", force_ascii=False, date_format="iso"))


def _date(d: str, kind: str = "runs") -> str:
    if d == "latest":
        base = OUT / kind if kind != "research" else OUT / "research/daily"
        ds = sorted(p.name for p in base.iterdir() if p.is_dir() and DATE.match(p.name))
        if not ds:
            raise HTTPException(404, f"no {kind} dates")
        return ds[-1]
    if not DATE.match(d):
        raise HTTPException(400, "date must be YYYY-MM-DD or latest")
    return d


def _day_outputs(d: str) -> dict:
    runs = sorted((OUT / "runs" / d).glob("day_*.json"), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise HTTPException(404, f"no daily run for {d}")
    return json.loads(runs[-1].read_text(encoding="utf-8"))["outputs"]


def _output(d: str, key: str) -> str:
    """当天某一步的产物路径；早期的运行日可能没有这一步（例如 10-12 还没有 bypass / f5c），返回 404 而不是 500。"""
    o = _day_outputs(d)
    if key not in o:
        raise HTTPException(404, f"{key} not produced on {d}")
    return o[key]


def _json(p: Path):
    if not p.exists():
        raise HTTPException(404, str(p.name))
    return json.loads(p.read_text(encoding="utf-8"))


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "data_root": str(ROOT), "latest_run": _date("latest")}


@app.get("/api/dates")
def dates() -> dict:
    def ls(p: Path) -> list[str]:
        return sorted(x.name for x in p.iterdir() if x.is_dir() and DATE.match(x.name)) if p.exists() else []
    research = ls(OUT / "research/daily")
    return {"runs": ls(OUT / "runs"), "books": ls(OUT / "books"), "research": research,
            "formal": ls(OUT / "formal"),
            "brain": [d for d in research if (OUT / "research/daily" / d / "brain_memo.json").exists()],
            "claims": [d for d in research if (OUT / "research/daily" / d / "f7_claims.parquet").exists()]}


@app.get("/api/formal/{d}")
def formal(d: str) -> dict:
    return _json(OUT / "formal" / _date(d, "formal") / "report.json")


@app.get("/api/formal/{d}/markdown", response_class=PlainTextResponse)
def formal_md(d: str) -> str:
    p = OUT / "formal" / _date(d, "formal") / "report.md"
    if not p.exists():
        raise HTTPException(404, "report.md")
    return p.read_text(encoding="utf-8")


@app.get("/api/brain/{d}")
def brain(d: str) -> dict:
    d = _date(d, "research")
    memo = _json(OUT / "research/daily" / d / "brain_memo.json")
    run = OUT / "research/daily" / d / "brain_run.json"
    meta = {k: v for k, v in json.loads(run.read_text(encoding="utf-8")).items() if k != "steps"} if run.exists() else {}
    return {"memo": memo, "run": meta}


@app.get("/api/market/{d}")
def market(d: str) -> dict:
    d = _date(d)
    fs = sorted(glob.glob(str(OUT / "runs" / d / "m_market_*.json")))
    if not fs:
        raise HTTPException(404, "m_market")
    return json.loads(Path(fs[-1]).read_text(encoding="utf-8"))


@app.get("/api/candidates/{d}")
def candidates(d: str, horizon: str = Query("20d", pattern="^(20d|63d)$"), top: int = Query(50, ge=1, le=500)) -> list[dict]:
    c = pd.read_parquet(_output(_date(d), "f4"))
    return _jsonable(c[c["horizon"] == horizon].head(top))


@app.get("/api/scores/{d}")
def scores(d: str, model: str = Query("ens_20d", pattern="^(ens_20d|ens_63d|ridge_20d|ridge_63d)$"),
           top: int = Query(100, ge=1, le=6000), ticker: str | None = None) -> list[dict]:
    """ticker：只返回这一只（带全市场排名），前端不必为了找一只股票拉前 500 名。"""
    d = _date(d)
    fs = sorted(glob.glob(str(OUT / "runs" / d / f"f3_{model}_*.parquet")), key=lambda f: Path(f).stat().st_mtime)
    if not fs:
        raise HTTPException(404, f"f3_{model}")
    s = pd.read_parquet(fs[-1])
    col = "ENS" if "ENS" in s else "score"
    s = s.sort_values(col, ascending=False).reset_index(drop=True)
    s["rank"] = s.index + 1
    if ticker:
        return _jsonable(s[s["ticker"] == ticker])
    return _jsonable(s.head(top))


@app.get("/api/bypass/{d}")
def bypass(d: str, importance: str | None = Query(None, pattern="^(major|watch)$")) -> list[dict]:
    b = pd.read_parquet(_output(_date(d), "bypass"))
    if importance:
        b = b[b["importance"] == importance]
    return _jsonable(b)


@app.get("/api/risk_cards/{d}")
def risk_cards(d: str, ticker: str | None = None) -> list[dict]:
    o = _day_outputs(_date(d))
    if "f5c" not in o:
        raise HTTPException(404, "f5c")
    k = pd.read_parquet(o["f5c"])
    return _jsonable(k[k["ticker"] == ticker] if ticker else k)


@app.get("/api/research/{d}")
def research(d: str) -> dict:
    d = _date(d, "research")
    day = OUT / "research/daily" / d
    md = day / "f9_research.md"
    return {"status": _json(day / "status.json") if (day / "status.json").exists() else None,
            "summaries": _json(day / "f9_summaries.json") if (day / "f9_summaries.json").exists() else [],
            "cases": _jsonable(pd.read_parquet(day / "f6_cases.parquet")) if (day / "f6_cases.parquet").exists() else [],
            "report_markdown": md.read_text(encoding="utf-8") if md.exists() else None}


@app.get("/api/research/{d}/{ticker}")
def research_ticker(d: str, ticker: str) -> dict:
    day = OUT / "research/daily" / _date(d, "research")
    cl = pd.read_parquet(day / "f7_claims.parquet") if (day / "f7_claims.parquet").exists() else pd.DataFrame()
    rf = pd.read_parquet(day / "f7_risk_flags.parquet") if (day / "f7_risk_flags.parquet").exists() else pd.DataFrame()
    return {"claims": _jsonable(cl[cl["ticker"] == ticker]) if len(cl) else [],
            "risk_flags": _jsonable(rf[rf["ticker"] == ticker]) if len(rf) else []}


@app.get("/api/books/{d}")
def books(d: str) -> dict:
    return _json(OUT / "books" / _date(d, "books") / "books_summary.json")


@app.get("/api/books/{d}/{book}/{kind}")
def book_detail(d: str, book: str, kind: str = "nav") -> list[dict]:
    if kind not in ("nav", "targets", "capacity", "rejected") or not re.match(r"^[\w.\-+]+$", book):
        raise HTTPException(400, "bad book or kind")
    p = OUT / "books" / _date(d, "books") / f"{book}__{kind}.parquet"
    if not p.exists():
        raise HTTPException(404, p.name)
    return _jsonable(pd.read_parquet(p))


@app.get("/api/stock/{ticker}/prices")
def prices(ticker: str, start: str = "2026-01-01", end: str | None = None) -> list[dict]:
    import pyarrow.dataset as pds
    fs = sorted(glob.glob(str(ROOT / "data/raw/alpha_prices_panel_v6_2*.parquet")))
    if not fs:
        raise HTTPException(404, "price panel")
    cols = ["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "pct_chg", "vol", "amount", "adj_factor"]
    t = pds.dataset(fs[-1]).to_table(columns=cols, filter=(pds.field("ts_code") == ticker)).to_pandas()
    t["trade_date"] = pd.to_datetime(t["trade_date"])
    t = t[(t["trade_date"] >= pd.Timestamp(start)) & (t["trade_date"] <= pd.Timestamp(end or "2100-01-01"))]
    return _jsonable(t.sort_values("trade_date"))


@app.get("/api/stock/{ticker}/announcements")
def announcements(ticker: str, as_of: str, lookback_days: int = Query(90, ge=7, le=730)) -> dict:
    from quantmind.line_a.harness.tools_more import ResearchTools
    return ResearchTools(ROOT / "data/lake", as_of.replace("-", "")).list_announcements(ticker, lookback_days=lookback_days)


@app.get("/api/backtest/t052")
def backtest_t052() -> dict:
    d = OUT / "t052_sim"
    return {"primary": _json(d / "primary.json"), "report_markdown": (d / "report.md").read_text(encoding="utf-8")
            if (d / "report.md").exists() else None}


@app.get("/api/backtest/t052/nav/{key}")
def backtest_nav(key: str) -> list[dict]:
    if not re.match(r"^[\w.\-+]+$", key):
        raise HTTPException(400, "bad key")
    p = OUT / "t052_sim/runs" / key / "nav.parquet"
    if not p.exists():
        raise HTTPException(404, key)
    return _jsonable(pd.read_parquet(p).reset_index())


@app.get("/api/experiments")
def experiments() -> list[dict]:
    rows = []
    for f in sorted((REPO / "docs/work/tasks").glob("T-0*.md")):
        txt = f.read_text(encoding="utf-8")
        first = txt.splitlines()[0].lstrip("# ").strip() if txt else f.stem
        verdicts = [ln.strip() for ln in txt.splitlines() if "判定" in ln and ("采用" in ln or "通过" in ln)][-3:]
        rows.append({"id": f.stem, "title": first, "bytes": len(txt.encode()), "verdict_lines": verdicts})
    return rows


@app.get("/api/experiments/{tid}", response_class=PlainTextResponse)
def experiment(tid: str) -> str:
    if not re.match(r"^T-\d{3}[\w-]*$", tid):
        raise HTTPException(400, "bad id")
    p = REPO / "docs/work/tasks" / f"{tid}.md"
    if not p.exists():
        raise HTTPException(404, tid)
    return p.read_text(encoding="utf-8")


@app.get("/api/catalog")
def catalog() -> dict:
    return _json(OUT / "dsh_out/t55/data_catalog.json")


# ------------------------------------------------------------------ 聚合接口（10-24 前端审查后新增，见 api_analytics）
TICK = re.compile(r"^\d{6}\.(SH|SZ|BJ)$")


def _an(fn, *a, **k):
    from quantmind.line_a import api_analytics  # noqa: F401
    try:
        return fn(*a, **k)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e
    except KeyError as e:
        raise HTTPException(404, f"not found: {e}") from e


def _horizons(s: str) -> tuple[int, ...]:
    try:
        hs = tuple(sorted({int(x) for x in s.split(",") if x}))
    except ValueError as e:
        raise HTTPException(400, "horizons must be integers") from e
    if not hs or any(h < 1 or h > 60 for h in hs):
        raise HTTPException(400, "horizons in 1..60")
    return hs


@app.get("/api/forward/{d}")
def forward(d: str, tickers: str | None = None, source: str = Query("formal", pattern="^(formal|f4|f3top)$"),
            horizons: str = "1,5,10,20", top: int = Query(100, ge=1, le=500)) -> dict:
    """T 日信号的事后收益（T+1 开盘复权入场），一次请求返回整组股票和基准，替代前端逐只请求行情。"""
    from quantmind.line_a import api_analytics as A
    d = _date(d)
    ts = [t for t in (tickers or "").split(",") if t]
    if any(not TICK.match(t) for t in ts) or len(ts) > 500:
        raise HTTPException(400, "bad tickers")
    if not ts:
        ts = _an(A.subject_tickers, d, source, top)
    return _an(A.forward, ts, d, _horizons(horizons))


@app.get("/api/redteam/{d}")
def redteam(d: str, source: str = Query("f4", pattern="^(formal|f4|f3top)$"), horizon: int = Query(5, ge=1, le=60),
            version: str = Query("v2", pattern="^(v1|v2)$"), top: int = Query(100, ge=1, le=500)) -> dict:
    from quantmind.line_a import api_analytics as A
    return _an(A.redteam, _date(d), source, horizon, top, version)


@app.get("/api/consensus/{d}")
def consensus(d: str, ticker: str | None = None, horizon: str = Query("20d", pattern="^(20d|63d)$"),
              universe: str = Query("research_pool", pattern="^(research_pool|all)$")) -> dict:
    from quantmind.line_a import api_analytics as A
    if ticker and not TICK.match(ticker):
        raise HTTPException(400, "bad ticker")
    return _an(A.consensus, _date(d), ticker, horizon, universe)


@app.get("/api/claim/{d}/{ticker}/{ref}")
def claim(d: str, ticker: str, ref: str) -> dict:
    """F7 一条声明的真实核验记录：原声明、所引记录的字段值、校验器备注（运行时和现在各一份）、按公式重算。"""
    from quantmind.line_a import api_analytics as A
    if not TICK.match(ticker) or not re.match(r"^A[0-9]:[\w.\-]+$", ref):
        raise HTTPException(400, "bad ticker or ref")
    return _an(A.claim_trace, _date(d, "research"), ticker, ref)


@app.get("/api/funnel/{d}")
def funnel(d: str) -> dict:
    from quantmind.line_a import api_analytics as A
    return _an(A.funnel, _date(d))


@app.get("/api/forward_ic")
def forward_ic(horizon: int = Query(5, ge=1, le=60), model: str = Query("ens_20d", pattern="^(ens_20d|ens_63d|ridge_20d|ridge_63d)$")) -> dict:
    from quantmind.line_a import api_analytics as A
    return _an(A.forward_ic, horizon, model)


@app.get("/api/rules/execution")
def rules_execution() -> dict:
    from quantmind.line_a import api_analytics as A
    return _an(A.execution_rules)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8780)
    a = ap.parse_args()
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="info")


if __name__ == "__main__":
    main()
