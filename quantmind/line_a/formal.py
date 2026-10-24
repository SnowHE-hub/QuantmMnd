"""quantmind.line_a.formal — 正式荐股日报（用户 2026-10-24 批准恢复正式荐股、资金闸门）.

每个资金档一本"正式纸面账户"：
- 用 T-052 验证过的按股数记账模拟器（sim_account），从首次调仓日起每天重放到当天，得到当前持仓和明天要执行的委托
  （股数按当天收盘价估算，实际以开盘成交为准）；
- 打分：冻结第 18 折 ENS20 模型包的每日 F3；调仓：周频网格每 4 期一次；组合与执行规则见 configs/line_a_v2/formal.yaml；
- 首次调仓日之前，给出"按今天的打分建仓"的预览。
日报附上：持仓的研究摘要（F9）、风险提示（F7）、当天的重大事件（旁路），以及 Agent 大脑的意见（只作提示，不改持仓）。
本模块只生成报告和委托清单，不连接券商、不下单。
"""
from __future__ import annotations

import argparse
import glob
import json
import shutil
from pathlib import Path

import pandas as pd
import yaml

from quantmind.line_a.f8_books import forward_grid
from quantmind.line_a.sim_account import AccountSimulator, Market, SimConfig, board_of, lot_shares

CFG = Path(__file__).resolve().parents[2] / "configs/line_a_v2/formal.yaml"
BOARD_CN = {"MAIN": "主板", "CHINEXT": "创业板", "STAR": "科创板", "BSE": "北交所"}


def load_cfg(p: Path = CFG) -> dict:
    return yaml.safe_load(Path(p).read_text(encoding="utf-8"))


def _f3(runs: Path, d: pd.Timestamp) -> pd.DataFrame | None:
    fs = sorted(glob.glob(str(runs / str(d.date()) / "f3_ens_20d_*.parquet")), key=lambda f: Path(f).stat().st_mtime)
    if not fs:
        return None
    df = pd.read_parquet(fs[-1])
    df = df.rename(columns={"ENS": "score"}) if "ENS" in df else df
    df["as_of"] = pd.Timestamp(d)
    return df[["as_of", "ticker", "score"]]


def _latest_run_day(runs: Path, as_of: pd.Timestamp) -> pd.Timestamp | None:
    ds = sorted(pd.Timestamp(p.name) for p in runs.iterdir() if p.is_dir() and p.name <= str(as_of.date())
                and list(p.glob("f3_ens_20d_*.parquet")))
    return ds[-1] if ds else None


def _names(runs: Path, d: pd.Timestamp) -> dict:
    fs = sorted(glob.glob(str(runs / str(d.date()) / "f1_instruments_*.parquet")))
    if not fs:
        return {}
    f1 = pd.read_parquet(fs[-1])
    return f1["name"].to_dict() if "name" in f1 else {}


def build(data_root: Path, out_root: Path, as_of: str, cfg: dict | None = None, sim_cls=AccountSimulator,
          keep: dict | None = None) -> dict:
    """sim_cls / keep：账本（ledger.py）用带记录钩子的模拟器重放同一套规则，并取回每一档的模拟器和结果。"""
    cfg = cfg or load_cfg()
    as_of_t = pd.Timestamp(as_of)
    runs = out_root / "runs"
    from quantmind.research.gpu.folds import calendar_and_grid
    _, frozen = calendar_and_grid(data_root)
    cal = pd.DatetimeIndex(pd.to_datetime(pd.read_parquet(data_root / "data/lake/trade_cal_sse.parquet")["cal_date"]
                                          .astype(str))).sort_values()
    grid = [d for d in forward_grid(cal[cal <= as_of_t], frozen) if d >= pd.Timestamp(cfg["go_live"])]
    step = int(cfg["rebalance_every_grid_points"])
    rebal = grid[::step]
    tag = as_of_t.strftime("%Y%m%d")
    px_path = data_root / f"data/raw/alpha_prices_panel_v6_{tag}.parquet"
    if not px_path.exists():
        raise FileNotFoundError(f"{px_path} 不存在：当天的价格面板还没生成（每日运行第 2 步）")
    cols = ["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "vol", "amount", "adj_factor"]
    prices = pd.read_parquet(px_path, columns=cols)
    amihud = None
    months = sorted({d.strftime("%Y-%m") for d in [as_of_t, *rebal]})
    fwd = Path(f"/home/lenovo/qm_fwd_root/{tag}/main/data/gpu_v1/panel/base")
    if fwd.exists():
        amihud = pd.concat([pd.read_parquet(fwd / f"{m}.parquet", columns=["amihud_illiquidity"]) for m in months
                            if (fwd / f"{m}.parquet").exists()] or [pd.DataFrame(columns=["amihud_illiquidity"])])
    from quantmind.line_a.challengers import _st_names
    st = _st_names(data_root / "data/lake")
    st = st[st["name"].astype(str).str.contains("ST")][["ts_code", "s", "e"]]
    dv = pd.read_parquet(sorted((data_root / "data/lake/events").glob("dividend_2*.parquet"))[-1],
                         columns=["ts_code", "div_proc", "ex_date", "cash_div_tax", "stk_div"])
    ex = cfg["execution"]
    scores = [x for x in (_f3(runs, d) for d in rebal) if x is not None]
    missing = [str(d.date()) for d in rebal if _f3(runs, d) is None]
    preview = not scores
    if preview:                                     # 首次调仓之前：按最新一天的打分预览建仓
        d0 = _latest_run_day(runs, as_of_t)
        scores = [_f3(runs, d0)] if d0 is not None else []
    if not scores:
        raise ValueError("没有可用的 F3 打分")
    sc = pd.concat(scores, ignore_index=True)
    start = pd.Timestamp(sc["as_of"].min())
    market = Market(prices, start, as_of_t)
    names = _names(runs, _latest_run_day(runs, as_of_t) or as_of_t)
    close_today = market.close_ffill.loc[:as_of_t].iloc[-1]
    tiers = {}
    for key, t in cfg["tiers"].items():
        c = SimConfig(aum=float(t["aum"]), top_n=int(t["top_n"]), rebalance_step=1, buffer_mult=t.get("buffer_mult"),
                      exec_price=ex["price"], commission_rate=ex["commission_rate"], min_commission=ex["min_commission"],
                      transfer_rate=ex["transfer_rate"], participation=ex["participation"], boards=frozenset(t["boards"]))
        sim = sim_cls(sc, market, c, start=start, end=as_of_t, amihud=amihud, dividends=dv, st_intervals=st)
        r = sim.run()
        if keep is not None:
            keep[key] = (sim, r)
        h = r["holdings"]
        last = h[h["date"] == h["date"].max()] if len(h) else h
        orders = []
        if not cfg.get("kill_switch"):
            for o in sim.pending:
                tk = o["ticker"]
                px = float(close_today.get(tk, float("nan")))
                if o["side"] == "buy":
                    sh = lot_shares(o["value"], px, board_of(tk)) if px == px else 0.0
                else:
                    p = sim.pos.get(tk)
                    sh = p.shares if (p and o["sell_all"]) else (lot_shares(o["value"], px, board_of(tk)) if px == px else 0.0)
                orders.append({"side": o["side"], "ticker": tk, "name": names.get(tk, ""), "board": BOARD_CN.get(board_of(tk), ""),
                               "est_shares": float(sh), "ref_close": px, "est_value": float(sh * px) if px == px else None,
                               "rank_in_target": int(o["rank"]) + 1 if o["rank"] >= 0 else None,
                               "decided": str(pd.Timestamp(o["decided"]).date())})
        nav = r["nav"]
        tiers[key] = {"aum": float(t["aum"]), "top_n": int(t["top_n"]), "boards": t["boards"],
                      "equity": float(nav["equity"].iloc[-1]) if len(nav) else float(t["aum"]),
                      "cash": float(nav["cash"].iloc[-1]) if len(nav) else float(t["aum"]),
                      "return_since_start": float(nav["equity"].iloc[-1] / t["aum"] - 1) if len(nav) else 0.0,
                      "pool_ew_since_start": float(nav["bench_pool_ew"].iloc[-1] / nav["bench_pool_ew"].iloc[0] - 1)
                      if len(nav) else 0.0,
                      "holdings": [{"ticker": x.ticker, "name": names.get(x.ticker, ""), "board": BOARD_CN.get(x.board, ""),
                                    "shares": float(x.shares), "price": float(x.price), "value": float(x.value),
                                    "weight": float(x.weight)} for x in last.itertuples()],
                      "orders_next_open": orders, "costs": r["costs"],
                      "targets_latest": list(r["targets"]["targets"].iloc[-1]) if len(r["targets"]) else []}
    fut = data_root / "data/lake/trade_cal_sse_future.parquet"           # 含未来交易日（fetch_future_calendar.py）
    full = pd.DatetimeIndex(pd.to_datetime(pd.read_parquet(fut)["cal_date"])).sort_values() if fut.exists() else cal
    nxt = [d for d in forward_grid(full, frozen) if d > as_of_t and d >= pd.Timestamp(cfg["go_live"])]
    next_rebal = None
    if rebal and nxt:
        k = len(grid)                                     # 下一个网格点的序号；每 step 个一调
        nxt_idx = [i for i in range(len(nxt)) if (k + i) % step == 0]
        next_rebal = str(nxt[nxt_idx[0]].date()) if nxt_idx else None
    elif nxt:
        next_rebal = str(nxt[0].date())
    return {"as_of": str(as_of_t.date()), "mode": "preview" if preview else "live", "approval": cfg["approval"],
            "kill_switch": bool(cfg.get("kill_switch")), "go_live": cfg["go_live"],
            "rebalance_dates": [str(d.date()) for d in rebal], "missing_scores": missing,
            "today_is_rebalance": bool(rebal and rebal[-1] == as_of_t), "next_rebalance": next_rebal,
            "score_dates_used": sorted({str(d.date()) for d in sc["as_of"]}), "tiers": tiers}


def _research_for(out_root: Path, as_of: str, tickers: set[str]) -> tuple[dict, dict]:
    days = sorted(p.name for p in (out_root / "research/daily").iterdir() if p.is_dir() and p.name <= as_of)
    summ, flags = {}, {}
    for d in days[-10:]:                                   # 最近 10 个研究日里每只股票最新的一份
        f = out_root / "research/daily" / d / "f9_summaries.json"
        if f.exists():
            for s in json.loads(f.read_text(encoding="utf-8")):
                if s["ticker"] in tickers:
                    pts = [p["text"] for p in (s.get("points") or s.get("first_pass") or [])][:3]
                    summ[s["ticker"]] = {"date": d, "points": pts}
        rf = out_root / "research/daily" / d / "f7_risk_flags.parquet"
        if rf.exists():
            r = pd.read_parquet(rf)
            for t, g in r[r["ticker"].isin(tickers)].groupby("ticker"):
                flags[t] = {"date": d, "flags": list(g["flag"])[:4]}
    return summ, flags


def render(rep: dict, out_root: Path) -> str:
    as_of = rep["as_of"]
    held = {h["ticker"] for t in rep["tiers"].values() for h in t["holdings"]} | \
        {o["ticker"] for t in rep["tiers"].values() for o in t["orders_next_open"]}
    summ, flags = _research_for(out_root, as_of, held)
    by = {}
    runs = sorted((out_root / "runs" / as_of).glob("day_*.json"), key=lambda p: p.stat().st_mtime)
    if runs:
        o = json.loads(runs[-1].read_text(encoding="utf-8"))["outputs"]
        if "bypass" in o:
            b = pd.read_parquet(o["bypass"])
            by = {r.ticker: r for r in b[b["ticker"].isin(held)].itertuples()}
    brain = out_root / "research/daily" / as_of / "brain_memo.json"
    memo = json.loads(brain.read_text(encoding="utf-8")) if brain.exists() else None
    a = rep["approval"]
    L = [f"# 正式荐股日报 · {as_of}", "",
         f"- 状态：{'**预览**（首次调仓日之前，按最新打分模拟建仓）' if rep['mode'] == 'preview' else '正式'}；"
         f"{'**今天是调仓日**' if rep['today_is_rebalance'] else '今天不是调仓日'}；下一个调仓日：{rep['next_rebalance'] or '—'}",
         f"- 批准：研究报告 {'已批准' if a['research_report_approved'] else '未批准'}，资金闸门 {a['capital_gate']}（{a['approved_on']}，用户）",
         f"- 依据：{a['evidence_at_approval'].strip()}",
         "- 打分来自冻结的 ENS20 模型包，组合与执行规则事先固定（`configs/line_a_v2/formal.yaml`）。"
         "委托股数按今天收盘价估算，实际以明天开盘成交为准；涨停开盘买不进、跌停开盘卖不出。",
         "- 本报告由量化系统按固定规则生成，不是对任何个人的投资建议；是否下单、下多少由账户持有人自行决定。", ""]
    if rep["kill_switch"]:
        L += ["> **紧急停止开关已打开**：本日不生成委托。", ""]
    if memo:
        L += ["## Agent 大脑的意见（只作提示，不改持仓）", "", memo.get("summary", ""), ""]
        for f in memo.get("flags", [])[:12]:
            L.append(f"- {f.get('ticker')} · {f.get('action')}：{f.get('reason', '')[:160]}")
        L.append("")
    tier_cn = {"aum_1e5": "10 万元档", "aum_1e6": "100 万元档", "aum_1e7": "1000 万元档"}
    for key, t in rep["tiers"].items():
        L += [f"## {tier_cn.get(key, key)}（持股 {t['top_n']} 只；板块：{'、'.join(BOARD_CN[b] for b in t['boards'])}）", "",
              f"- 纸面账户：总资产 {t['equity']:,.0f} 元，现金 {t['cash']:,.0f} 元；"
              f"自首次调仓以来 {t['return_since_start'] * 100:+.2f}%（同期前 1500 等权 {t['pool_ew_since_start'] * 100:+.2f}%）", ""]
        if t["orders_next_open"]:
            head = (f"### 建仓预览（按今天的打分；首次调仓日 {rep['next_rebalance']}，到时按当天打分重新生成）"
                    if rep["mode"] == "preview" else "### 明日开盘委托")
            L += [head, "", "| 方向 | 代码 | 名称 | 板块 | 估计股数 | 参考价（今收） | 估计金额 | 目标排名 |",
                  "|---|---|---|---|---|---|---|---|"]
            skip = []
            for o in sorted(t["orders_next_open"], key=lambda o: (o["side"] != "sell", o["rank_in_target"] or 999)):
                if o["side"] == "buy" and o["est_shares"] <= 0:
                    skip.append(o)
                    continue
                L.append(f"| {'卖出' if o['side'] == 'sell' else '买入'} | {o['ticker']} | {o['name']} | {o['board']} | "
                         f"{o['est_shares']:,.0f} | {o['ref_close']:.2f} | {(o['est_value'] or 0):,.0f} | {o['rank_in_target'] or '—'} |")
            L.append("")
            if skip:
                per = t["aum"] / t["top_n"]
                L += [f"- 买不起一手、按规则跳过（每只目标金额约 {per:,.0f} 元，现金留着，不顺延到下一名）："
                      + "；".join(f"{o['ticker']} {o['name']}（一手约 {o['ref_close'] * 100:,.0f} 元）" for o in skip), ""]
        else:
            L += ["- 明日没有委托。", ""]
        if t["holdings"]:
            L += ["### 当前持仓", "", "| 代码 | 名称 | 板块 | 股数 | 收盘价 | 市值 | 权重 | 事件 |", "|---|---|---|---|---|---|---|---|"]
            for h in sorted(t["holdings"], key=lambda h: -h["value"]):
                ev = by.get(h["ticker"])
                L.append(f"| {h['ticker']} | {h['name']} | {h['board']} | {h['shares']:,.0f} | {h['price']:.2f} | {h['value']:,.0f} | "
                         f"{h['weight'] * 100:.1f}% | {getattr(ev, 'importance', '') if ev is not None else ''} |")
            L.append("")
    if summ or flags or by:
        L += ["## 相关股票的研究与事件", ""]
        for tk in sorted(held):
            if tk not in summ and tk not in flags and tk not in by:
                continue
            L.append(f"**{tk}**")
            if tk in by:
                L.append(f"- 今日事件（{by[tk].importance}）：{'；'.join(map(str, list(by[tk].summaries)[:3]))[:200]}")
            if tk in flags:
                L.append(f"- 风险提示（{flags[tk]['date']}）：{'；'.join(flags[tk]['flags'])[:300]}")
            if tk in summ:
                L.append(f"- 研究要点（{summ[tk]['date']}）：{' / '.join(summ[tk]['points'])[:400]}")
            L.append("")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out-root", type=Path)
    ap.add_argument("--with-brain", action="store_true", help="保存正式计划后运行 Agent 大脑，再生成日报")
    a = ap.parse_args()
    out_root = a.out_root or a.data_root / "data/line_a_v2"
    cfg = load_cfg()
    if not cfg.get("enabled"):
        print("formal disabled")
        return
    rep = build(a.data_root, out_root, a.as_of, cfg)
    d = out_root / "formal" / rep["as_of"]
    d.mkdir(parents=True, exist_ok=True)
    (d / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    if a.with_brain:
        try:
            from quantmind.line_a import brain
            b = brain.run(a.data_root, out_root, rep["as_of"], rep)
            print("brain", b["status"], round(b["cost_rmb"], 3), "flags", len(b["memo"].get("flags", [])),
                  "dropped", len(b["dropped_flags"]))
        except Exception as e:  # noqa: BLE001  大脑失败（例如网络）不影响日报，日报里就没有大脑一节
            print("brain failed:", type(e).__name__, str(e)[:200])
    md = render(rep, out_root)
    (d / "report.md").write_text(md, encoding="utf-8")
    w = Path(cfg["output"]["windows_dir"])
    try:
        w.mkdir(parents=True, exist_ok=True)
        shutil.copy(d / "report.md", w / f"正式荐股日报_{rep['as_of']}.md")
    except OSError as e:
        print("windows copy failed:", e)
    print(json.dumps({"as_of": rep["as_of"], "mode": rep["mode"], "next_rebalance": rep["next_rebalance"],
                      "orders": {k: len(t["orders_next_open"]) for k, t in rep["tiers"].items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
