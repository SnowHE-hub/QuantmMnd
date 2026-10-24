"""quantmind.line_a.ledger — 按股数记账的模拟账户账本（给前端的"模拟持仓与交易账本"用）.

账本不是另算一套：用 T-052 / 正式日报同一个 AccountSimulator 重跑，只在模拟器上挂记录钩子，
把它内部本来就在做的事逐笔记下来：
- 每笔成交的佣金（最低 5 元）、过户费、卖出印花税、滑点（模拟的冲击成本）、红利税，以及成交后的现金余额；
- 每天收盘后的持仓（股数、T+1 可用数量、收盘价、市值）；
- 分红与送转（现金红利入账、送转后股数）；
- 没成交的委托及原因（涨停、停牌、T+1、买不起一手、参与率上限、到期未成交）。
在此之上计算持仓成本（移动加权平均，含买入费用）、已实现 / 未实现盈亏、完整回合、每日与每月盈亏。

会计恒等式逐日核对，写进 meta.checks：
1. 总资产 = 现金 + Σ 股数 × 收盘价；
2. 总资产 − 本金 = 已实现盈亏 + 未实现盈亏 + 现金分红；
3. T-052 账户的每日总资产与 T-052 存档的净值逐日一致（重跑可复现）。

输出：data/line_a_v2/ledger/<account>/{meta.json, daily.parquet, positions.parquet, trades.parquet,
rejects.parquet, actions.parquet, round_trips.parquet}。只写这个新目录。
用法：PYTHONPATH=. python -m quantmind.line_a.ledger build [--t052] [--formal] [--shadow] [--as-of YYYY-MM-DD]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from quantmind.line_a.sim_account import LOT, AccountSimulator, board_of

ROOT = Path("/home/lenovo/projects/quantmind")
OUT = ROOT / "data/line_a_v2"
LEDGER = OUT / "ledger"
REPO = Path(__file__).resolve().parents[2]
BOARD_CN = {"MAIN": "主板", "CHINEXT": "创业板", "STAR": "科创板", "BSE": "北交所"}
REJECT_CN = {
    "limit_up_locked": "当日触及涨停价，保守视为买不进", "suspended_no_bar": "停牌（当日无行情）", "limit_down_locked": "跌停卖不出", "limit_down": "跌停卖不出",
    "suspended": "停牌", "no_bar": "停牌", "t_plus_1_blocked": "T+1 当日买入不可卖", "cannot_afford_one_lot": "买不起一个最小申报单位",
    "insufficient_cash": "现金不足", "participation_cap": "超过成交量 5% 参与率上限", "expired_unfilled": "到下一个调仓日仍未成交，作废",
    "below_one_lot": "部分卖出不足一个申报单位", "board_not_allowed": "账户无该板块权限",
}


class _Hook(list):
    def __init__(self, on_append):
        super().__init__()
        self._on = on_append

    def append(self, x):
        self._on(x)
        super().append(x)


class LedgerSimulator(AccountSimulator):
    """AccountSimulator + 记录钩子。不改任何交易逻辑：只在成交、日终、分红时记下当时的状态。"""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._last_fee: dict = {}
        self.trade_log: list[dict] = []
        self.day_pos: list[dict] = []
        self.actions: list[dict] = []
        self.trades = _Hook(self._on_trade)
        self.nav_rows = _Hook(self._on_day)

    def _fees(self, gross, side, d, slip_bp):
        f = super()._fees(gross, side, d, slip_bp)
        self._last_fee = f                         # 同一个字典：卖出时随后加进去的 dividend_tax 也会记到
        return f

    def _on_trade(self, t: dict) -> None:
        fee = dict(self._last_fee) if t.get("cost", 0) else {}
        self.trade_log.append(t | {k: float(fee.get(k, 0.0)) for k in ("commission", "transfer", "stamp", "slippage", "dividend_tax")}
                              | {"cash_after": float(self.cash)})
        self._last_fee = {}

    def _on_day(self, row: dict) -> None:
        d = row["date"]
        for t, p in self.pos.items():
            self.day_pos.append({"date": d, "ticker": t, "shares": float(p.shares),
                                 "bought_today": p.last_buy == d, "close": float(self.close_ffill.at[d, t])})

    def _corporate_actions(self, d) -> None:
        before = {t: (p.shares, p.div_received) for t, p in self.pos.items()}
        super()._corporate_actions(d)
        for t, p in self.pos.items():
            s0, dv0 = before.get(t, (p.shares, p.div_received))
            if p.shares != s0 or p.div_received != dv0:
                self.actions.append({"date": d, "ticker": t, "cash_dividend": float(p.div_received - dv0),
                                     "shares_before": float(s0), "shares_after": float(p.shares)})


# ------------------------------------------------------------------ 账本计算
def _names() -> dict[str, str]:
    fs = sorted((ROOT / "data/lake").glob("stock_basic_full_merged_2*.parquet"))
    if not fs:
        return {}
    s = pd.read_parquet(fs[-1], columns=["ts_code", "name"])
    return dict(zip(s["ts_code"], s["name"]))


def _index_close(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    parts = []
    for f in ("index_daily.parquet", "index_daily_extra.parquet"):
        p = ROOT / "data/lake" / f
        if p.exists():
            x = pd.read_parquet(p, columns=["ts_code", "trade_date", "close"])
            x["trade_date"] = pd.to_datetime(x["trade_date"].astype(str).str.replace("-", "").str.slice(0, 8))
            parts.append(x)
    i = pd.concat(parts).drop_duplicates(["ts_code", "trade_date"], keep="last")
    i = i[i["ts_code"].isin(["000300.SH", "932000.CSI"]) & (i["trade_date"] >= start) & (i["trade_date"] <= end)]
    w = i.pivot_table(index="trade_date", columns="ts_code", values="close").sort_index()
    return w.rename(columns={"000300.SH": "hs300", "932000.CSI": "csi2000"})


def _reason(t: dict, held_before: bool, targets_at: dict) -> str:
    dec = t.get("decided")
    tg = targets_at.get(pd.Timestamp(dec)) if dec is not None and pd.notna(dec) else None
    if t["side"] == "writeoff":
        return "停牌 / 退市超过 20 个交易日，按最后价格清算"
    if t["side"] == "buy":
        if tg is not None and t["ticker"] in tg:
            r = tg.index(t["ticker"]) + 1
            return f"调仓加仓（目标第 {r} 名，回到等权）" if held_before else f"调仓新进目标（第 {r} 名）"
        return "调仓买入"
    if tg is not None and t["ticker"] in tg:
        return "调仓减仓（回到等权）"
    return "跌出目标（不在前 2N 缓冲内）清仓"


def assemble(sim: LedgerSimulator, r: dict, meta: dict) -> dict[str, pd.DataFrame | dict]:
    """把一次模拟的记录整理成账本表格，并做会计核对。"""
    aum = float(sim.cfg.aum)
    names = _names()
    nav = r["nav"].copy()
    nav.index = pd.to_datetime(nav.index)
    tr = pd.DataFrame(sim.trade_log)
    targets_at = {pd.Timestamp(x["as_of"]): list(x["targets"]) for x in sim.targets}
    acts = pd.DataFrame(sim.actions, columns=["date", "ticker", "cash_dividend", "shares_before", "shares_after"])

    # 逐笔：成本（移动加权平均，含买入费用；送转后总成本不变）、已实现盈亏、回合
    cost, shares, first = {}, {}, {}
    rt_open: dict[str, dict] = {}
    round_trips, rows = [], []
    ev = []
    if len(tr):
        tr["date"] = pd.to_datetime(tr["date"])
        tr["_k"] = range(len(tr))
        ev += [("trade", x) for x in tr.to_dict("records")]
    for a in acts.to_dict("records"):
        ev.append(("action", a))
    order = {"action": 0, "trade": 1}                       # 同一天：先除权（开盘前），再成交
    ev.sort(key=lambda e: (pd.Timestamp(e[1]["date"]), order[e[0]], e[1].get("_k", 0)))
    realized_by_day: dict[pd.Timestamp, float] = {}
    for kind, x in ev:
        t, d = x["ticker"], pd.Timestamp(x["date"])
        if kind == "action":
            shares[t] = x["shares_after"]
            if t in rt_open:
                rt_open[t]["dividends"] += x["cash_dividend"]
            continue
        fee = x["commission"] + x["transfer"] + x["stamp"] + x["slippage"] + x["dividend_tax"]
        held = shares.get(t, 0.0) > 0
        row = dict(x)
        row.update(name=names.get(t, ""), board=BOARD_CN.get(board_of(t), ""), fee_total=fee,
                   reason=_reason(x, held, targets_at))
        if x["side"] == "buy":
            cost[t] = cost.get(t, 0.0) + x["gross"] + fee
            shares[t] = shares.get(t, 0.0) + x["shares"]
            first.setdefault(t, d)
            if t not in rt_open:
                rt_open[t] = {"ticker": t, "open_date": d, "buy_shares": 0.0, "buy_gross": 0.0, "buy_cost": 0.0,
                              "sell_shares": 0.0, "sell_gross": 0.0, "sell_net": 0.0, "dividends": 0.0}
            o = rt_open[t]
            o["buy_shares"] += x["shares"]
            o["buy_gross"] += x["gross"]
            o["buy_cost"] += x["gross"] + fee
            row.update(net_cash=-(x["gross"] + fee), realized_pnl=None)
        else:                                               # sell / writeoff
            avg = cost[t] / shares[t] if shares.get(t) else 0.0
            removed = avg * x["shares"]
            net = x["gross"] - fee
            pnl = net - removed
            cost[t] -= removed
            shares[t] -= x["shares"]
            realized_by_day[d] = realized_by_day.get(d, 0.0) + pnl
            o = rt_open[t]
            o["sell_shares"] += x["shares"]
            o["sell_gross"] += x["gross"]
            o["sell_net"] += net
            row.update(net_cash=net, realized_pnl=pnl, avg_cost=avg)
            if shares[t] <= 1e-9:
                shares.pop(t)
                cost.pop(t, None)
                first.pop(t, None)
                o = rt_open.pop(t)
                pnl_rt = o["sell_net"] + o["dividends"] - o["buy_cost"]
                round_trips.append(o | {
                    "name": names.get(t, ""), "board": BOARD_CN.get(board_of(t), ""), "close_date": d,
                    "avg_buy": o["buy_gross"] / o["buy_shares"] if o["buy_shares"] else None,
                    "avg_sell": o["sell_gross"] / o["sell_shares"] if o["sell_shares"] else None,
                    "days_held": int((d - o["open_date"]).days), "net_pnl": pnl_rt,
                    "return": pnl_rt / o["buy_cost"] if o["buy_cost"] else None, "writeoff": x["side"] == "writeoff"})
        row["_cost_after"] = cost.get(t, 0.0)
        row["_shares_after"] = shares.get(t, 0.0)
        rows.append(row)
    trades = pd.DataFrame(rows).drop(columns=["_k"], errors="ignore")

    # 每日持仓：股数来自模拟器日终快照；成本按当天收盘前的最后一笔逐笔记录
    pos = pd.DataFrame(sim.day_pos, columns=["date", "ticker", "shares", "bought_today", "close"])
    pos["date"] = pd.to_datetime(pos["date"])
    if len(pos):
        cost_track = {}
        last_cost = []
        tr_by_day = {d: g for d, g in trades.groupby("date")} if len(trades) else {}
        dates = sorted(pos["date"].unique())
        first_buy: dict[str, pd.Timestamp] = {}
        div_by: dict[str, float] = {}
        act_by = {d: g for d, g in acts.assign(date=pd.to_datetime(acts["date"])).groupby("date")} if len(acts) else {}
        bought_qty = {}
        for d in dates:
            for a in act_by.get(d, pd.DataFrame()).to_dict("records"):
                div_by[a["ticker"]] = div_by.get(a["ticker"], 0.0) + a["cash_dividend"]
            bought_qty = {}
            for x in tr_by_day.get(d, pd.DataFrame()).to_dict("records"):
                cost_track[x["ticker"]] = x["_cost_after"]
                if x["side"] == "buy":
                    first_buy.setdefault(x["ticker"], d)
                    bought_qty[x["ticker"]] = bought_qty.get(x["ticker"], 0.0) + x["shares"]
                elif x["_shares_after"] <= 1e-9:
                    first_buy.pop(x["ticker"], None)
                    div_by.pop(x["ticker"], None)
            for t in pos.loc[pos["date"] == d, "ticker"]:
                last_cost.append((d, t, cost_track.get(t, np.nan), first_buy.get(t), div_by.get(t, 0.0), bought_qty.get(t, 0.0)))
        lc = pd.DataFrame(last_cost, columns=["date", "ticker", "cost_basis", "open_date", "dividends", "bought_qty"])
        pos = pos.merge(lc, on=["date", "ticker"], how="left")
        pos["value"] = pos["shares"] * pos["close"]
        pos["available"] = pos["shares"] - pos["bought_qty"]              # T+1：当天买入的部分不可卖
        pos["avg_cost"] = pos["cost_basis"] / pos["shares"]
        pos["diluted_cost"] = (pos["cost_basis"] - pos["dividends"]) / pos["shares"]   # 摊薄成本：扣掉持有期间收到的现金分红
        pos["unrealized"] = pos["value"] - pos["cost_basis"]
        pos["unrealized_pct"] = pos["unrealized"] / pos["cost_basis"]
        pos["days_held"] = (pos["date"] - pd.to_datetime(pos["open_date"])).dt.days
        prev = pos.sort_values("date").groupby("ticker")["close"].shift()
        pre = sim.pre.stack().rename("pre_close").reset_index().rename(columns={"level_0": "date", "trade_date": "date", "level_1": "ticker", "ts_code": "ticker"})
        pre.columns = ["date", "ticker", "pre_close"]
        pos = pos.merge(pre, on=["date", "ticker"], how="left")
        pos["day_ret"] = pos["close"] / pos["pre_close"] - 1
        # 当日盈亏：昨天就有的股数按（收盘 − 昨收）计；今天买入的按（收盘 − 今天的成交均价）计
        if len(trades):
            b = trades[trades["side"] == "buy"].groupby(["date", "ticker"]).agg(bq=("shares", "sum"), bg=("gross", "sum")).reset_index()
            pos = pos.merge(b, on=["date", "ticker"], how="left")
        else:
            pos["bq"], pos["bg"] = np.nan, np.nan
        bq, bg = pos["bq"].fillna(0.0), pos["bg"].fillna(0.0)
        old_sh = (pos["shares"] - bq).clip(lower=0)
        pos["day_pnl"] = old_sh * (pos["close"] - pos["pre_close"]).fillna(0.0) + (bq * pos["close"] - bg)
        pos = pos.drop(columns=["bq", "bg"])
        pos["name"] = pos["ticker"].map(names).fillna("")
        pos["board"] = pos["ticker"].map(lambda t: BOARD_CN.get(board_of(t), ""))
        eq = nav["equity"]
        pos["weight"] = pos["value"] / pos["date"].map(eq)
        del prev

    # 每日：盈亏结构
    daily = nav[["equity", "cash"]].copy()
    daily["market_value"] = pos.groupby("date")["value"].sum().reindex(daily.index).fillna(0.0) if len(pos) else 0.0
    daily["n_pos"] = nav["n_pos"]
    daily["daily_pnl"] = daily["equity"].diff().fillna(daily["equity"].iloc[0] - aum)
    daily["daily_ret"] = daily["equity"] / daily["equity"].shift().fillna(aum) - 1
    daily["cum_ret"] = daily["equity"] / aum - 1
    daily["drawdown"] = daily["equity"] / daily["equity"].cummax() - 1
    daily["realized_cum"] = pd.Series(realized_by_day).reindex(daily.index).fillna(0.0).cumsum()
    daily["unrealized"] = pos.groupby("date")["unrealized"].sum().reindex(daily.index).fillna(0.0) if len(pos) else 0.0
    dv = acts.assign(date=pd.to_datetime(acts["date"])).groupby("date")["cash_dividend"].sum() if len(acts) else pd.Series(dtype=float)
    daily["dividends_cum"] = dv.reindex(daily.index).fillna(0.0).cumsum()
    for k in ("commission", "transfer", "stamp", "slippage", "dividend_tax"):
        daily[f"{k}_cum"] = (trades.groupby("date")[k].sum().reindex(daily.index).fillna(0.0).cumsum() if len(trades) else 0.0)
    daily["fees_cum"] = daily[[f"{k}_cum" for k in ("commission", "transfer", "stamp", "slippage", "dividend_tax")]].sum(axis=1)
    daily["bench_pool_ew"] = nav["bench_pool_ew"] / nav["bench_pool_ew"].iloc[0] - 1
    ix = _index_close(daily.index[0] - pd.Timedelta(days=10), daily.index[-1]).reindex(daily.index).ffill()
    for k in ix.columns:
        daily[f"bench_{k}"] = ix[k] / ix[k].iloc[0] - 1
    rebal = {pd.Timestamp(x["as_of"]) for x in sim.targets}
    daily["is_rebalance"] = daily.index.isin(list(rebal))
    daily["n_trades"] = trades.groupby("date").size().reindex(daily.index).fillna(0).astype(int) if len(trades) else 0

    # 会计核对
    checks = {}
    mv_check = (daily["cash"] + daily["market_value"] - daily["equity"]).abs().max()
    checks["equity_eq_cash_plus_mv_max_abs"] = float(mv_check)
    pnl_check = (daily["equity"] - aum - daily["realized_cum"] - daily["unrealized"] - daily["dividends_cum"]).abs().max()
    checks["pnl_decomposition_max_abs"] = float(pnl_check)
    if len(trades):
        flow = trades["net_cash"].sum() + (acts["cash_dividend"].sum() if len(acts) else 0.0)
        checks["cash_flow_identity_abs"] = float(abs(aum + flow - daily["cash"].iloc[-1]))
    checks["passed"] = bool(mv_check < 0.01 and pnl_check < 0.01 and checks.get("cash_flow_identity_abs", 0.0) < 0.01)

    rj = pd.DataFrame(sim.rejects)
    if len(rj):
        rj["date"] = pd.to_datetime(rj["date"])
        rj["reason_cn"] = rj["reason"].map(REJECT_CN).fillna(rj["reason"])
        rj["name"] = rj["ticker"].map(names).fillna("")
    rts = pd.DataFrame(round_trips)
    month = daily.assign(m=daily.index.strftime("%Y-%m")).groupby("m").agg(
        pnl=("daily_pnl", "sum"), days=("daily_pnl", "size"), win_days=("daily_pnl", lambda s: int((s > 0).sum())),
        start_equity=("equity", lambda s: float(s.iloc[0])), end_equity=("equity", "last"))
    month["win_rate"] = month["win_days"] / month["days"]
    last = daily.iloc[-1]
    meta = meta | {
        "aum": aum, "start": str(daily.index[0].date()), "end": str(daily.index[-1].date()),
        "summary": {"equity": float(last["equity"]), "cash": float(last["cash"]), "market_value": float(last["market_value"]),
                    "total_pnl": float(last["equity"] - aum), "total_ret": float(last["cum_ret"]),
                    "realized": float(last["realized_cum"]), "unrealized": float(last["unrealized"]),
                    "dividends": float(last["dividends_cum"]), "daily_pnl": float(last["daily_pnl"]),
                    "fees": {k: float(last[f"{k}_cum"]) for k in ("commission", "transfer", "stamp", "slippage", "dividend_tax")},
                    "fees_total": float(last["fees_cum"]), "max_drawdown": float(daily["drawdown"].min()),
                    "bench": {k: float(last[f"bench_{k}"]) for k in ("pool_ew", "hs300", "csi2000") if f"bench_{k}" in daily},
                    "n_trades": int(len(trades)), "n_rejects": int(len(rj)), "n_round_trips": int(len(rts))},
        "monthly": [{"month": m, **{k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v) for k, v in r.items()}}
                    for m, r in month.iterrows()],
        "rebalance_dates": sorted(str(d.date()) for d in rebal),
        "pending_orders": [{"side": o["side"], "ticker": o["ticker"], "name": names.get(o["ticker"], ""),
                            "value": float(o["value"]), "decided": str(pd.Timestamp(o["decided"]).date())} for o in sim.pending],
        "checks": checks,
        "rules": {"commission_rate": sim.cfg.commission_rate, "min_commission": sim.cfg.min_commission,
                  "transfer_rate": sim.cfg.transfer_rate, "stamp_duty": "卖出 0.05%（2023-08-28 起）", "participation": sim.cfg.participation,
                  "exec_price": "次日开盘价（未复权，真实账户口径）", "lot": {k: list(v) for k, v in LOT.items()},
                  "slippage": "按 Amihud 流动性分档的冲击成本（模拟值，不是券商收费）"}}
    return {"meta": meta, "daily": daily.reset_index().rename(columns={"index": "date"}), "positions": pos,
            "trades": trades.drop(columns=["_cost_after", "_shares_after"], errors="ignore"), "rejects": rj,
            "actions": acts, "round_trips": rts}


def save(account: str, led: dict) -> Path:
    d = LEDGER / account
    d.mkdir(parents=True, exist_ok=True)
    for k, v in led.items():
        if k == "meta":
            (d / "meta.json").write_text(json.dumps(v, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        elif isinstance(v, pd.DataFrame):
            v.to_parquet(d / f"{k}.parquet")
    return d


# ------------------------------------------------------------------ 账户来源
T052_PRIMARY = [("W6", 1e5, 10, "main+chinext"), ("W6", 1e6, 30, "all"), ("W6", 1e7, 100, "all"),
                ("W3", 1e5, 10, "main+chinext"), ("W3", 1e6, 30, "all"), ("W3", 1e7, 100, "all")]


def build_t052() -> list[str]:
    sys.path.insert(0, str(REPO / "docs/work/evidence/T-052"))
    import sim_grid as sg  # noqa: E402

    from quantmind.line_a.sim_account import SimConfig
    sg.load()
    done = []
    for win, aum, n, boards in T052_PRIMARY:
        key = f"{win}_ENS20_n{n}_s4_b2.0_open_{int(aum):d}_{boards}"
        start, end = sg.WINDOWS[win]
        bset = {"all": sg.ALL, "main+chinext": sg.SMALL, "main_only": sg.MAIN_ONLY}[boards]
        cfg = SimConfig(aum=aum, top_n=n, rebalance_step=4, buffer_mult=2.0, exec_price="open", boards=bset, drift_band=0.0)
        sim = LedgerSimulator(sg.G["scores"]["ENS20"], sg.G["market"], cfg, start=start, end=end, amihud=sg.G["amihud"],
                              dividends=sg.G["divs"], st_intervals=sg.G["st"])
        r = sim.run()
        meta = {"account": f"t052_{win}_{int(aum):d}", "kind": "t052", "t052_key": key,
                "title": f"T-052 全程模拟 · {win}（{'6' if win == 'W6' else '3'} 个月）· {int(aum / 1e4)} 万元",
                "description": "T-052 主配置（ENS20、每 4 周调仓、前 2N 缓冲、次日开盘成交），历史回放，不是实盘。"}
        led = assemble(sim, r, meta)
        stored = OUT / "t052_sim/runs" / key / "nav.parquet"
        if stored.exists():                          # 重跑可复现：与 T-052 存档的净值逐日比较
            s = pd.read_parquet(stored)["equity"]
            s.index = pd.to_datetime(s.index)
            e = led["daily"].set_index("date")["equity"]
            led["meta"]["checks"]["t052_nav_reproduced_max_abs"] = float((e - s.reindex(e.index)).abs().max())
            led["meta"]["checks"]["passed"] = led["meta"]["checks"]["passed"] and led["meta"]["checks"]["t052_nav_reproduced_max_abs"] < 0.01
        save(meta["account"], led)
        print(meta["account"], json.dumps(led["meta"]["checks"]), flush=True)
        done.append(meta["account"])
    return done


def build_formal(as_of: str) -> list[str]:
    from quantmind.line_a import formal
    keep: dict = {}
    rep = formal.build(ROOT, OUT, as_of, sim_cls=LedgerSimulator, keep=keep)
    done = []
    for key, (sim, r) in keep.items():
        acc = f"formal_{key}"
        meta = {"account": acc, "kind": "formal", "tier": key, "as_of": as_of, "mode": rep["mode"],
                "go_live": rep["go_live"], "next_rebalance": rep["next_rebalance"],
                "title": f"正式荐股跟踪账户 · {int(sim.cfg.aum / 1e4)} 万元",
                "description": ("正式日报的纸面账户（10-24 批准，go_live 2026-10-26）；每天用当天收盘数据重放。"
                                + ("当前是建仓预览：首次调仓之前账户全是现金，下面的委托是预览。" if rep["mode"] == "preview" else ""))}
        if len(r["nav"]) == 0:
            continue
        led = assemble(sim, r, meta)
        led["meta"]["preview_orders"] = rep["tiers"][key]["orders_next_open"]
        save(acc, led)
        print(acc, rep["mode"], json.dumps(led["meta"]["checks"]), flush=True)
        done.append(acc)
    return done


def build_shadow(as_of: str) -> list[str]:
    """L9 影子账本：冻结引擎按权重记账（研究口径，没有股数），这里只整理成同样的日序列，交割单为空。"""
    day = OUT / "books" / as_of
    s = json.loads((day / "books_summary.json").read_text(encoding="utf-8"))
    names = _names()
    done = []
    for book in s.get("books", {}):
        nav = pd.read_parquet(day / f"{book}__nav.parquet")
        if not len(nav):
            continue
        nav["date"] = pd.to_datetime(nav["date"])
        tg = pd.read_parquet(day / f"{book}__targets.parquet") if (day / f"{book}__targets.parquet").exists() else pd.DataFrame()
        acc = f"shadow_{book}"
        daily = nav.rename(columns={"nav_net": "nav"})[["date", "nav", "nav_gross", "nav_bench", "cash", "n_positions", "cum_cost",
                                                          "daily_ret_net", "excess_net"]]
        daily["cum_ret"] = daily["nav"] - 1
        daily["bench_pool_ew"] = daily["nav_bench"] - 1
        daily["drawdown"] = daily["nav"] / daily["nav"].cummax() - 1
        daily["is_rebalance"] = daily["date"].isin(pd.to_datetime(tg["as_of"]) if len(tg) else [])
        ix = _index_close(daily["date"].iloc[0] - pd.Timedelta(days=10), daily["date"].iloc[-1]).reindex(daily["date"]).ffill()
        for k in ix.columns:
            daily[f"bench_{k}"] = (ix[k] / ix[k].iloc[0] - 1).to_numpy()
        targets = []
        for x in tg.to_dict("records"):
            for t in list(x["targets"]):
                targets.append({"as_of": pd.Timestamp(x["as_of"]), "ticker": t, "name": names.get(t, ""),
                                "board": BOARD_CN.get(board_of(t), ""), "weight": float(x["target_weight_each"])})
        last = daily.iloc[-1]
        meta = {"account": acc, "kind": "shadow", "book": book, "as_of": as_of, "title": f"L9 影子账本 · {book}",
                "description": "冻结引擎按权重记账（研究口径）：没有股数、没有逐笔交割单；净值已扣交易成本。用于和正式账户对照。",
                "start": str(daily["date"].iloc[0].date()), "end": str(daily["date"].iloc[-1].date()),
                "summary": {"total_ret": float(last["cum_ret"]), "max_drawdown": float(daily["drawdown"].min()),
                            "cum_cost": float(last["cum_cost"]),
                            "bench": {k: float(last[f"bench_{k}"]) for k in ("pool_ew", "hs300", "csi2000") if f"bench_{k}" in daily}},
                "rebalance_dates": [str(pd.Timestamp(x).date()) for x in (tg["as_of"] if len(tg) else [])],
                "checks": {"passed": True, "note": "权重账本，不做股数级核对"}}
        save(acc, {"meta": meta, "daily": daily, "targets": pd.DataFrame(targets)})
        done.append(acc)
    return done


def accounts() -> list[dict]:
    out = []
    for d in sorted(LEDGER.glob("*/meta.json")):
        m = json.loads(d.read_text(encoding="utf-8"))
        out.append({k: m.get(k) for k in ("account", "kind", "title", "description", "start", "end", "aum", "mode", "tier")}
                   | {"summary": m.get("summary"), "checks_passed": m.get("checks", {}).get("passed")})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build"])
    ap.add_argument("--t052", action="store_true")
    ap.add_argument("--formal", action="store_true")
    ap.add_argument("--shadow", action="store_true")
    ap.add_argument("--as-of", default=None)
    a = ap.parse_args()
    as_of = a.as_of or sorted(p.name for p in (OUT / "runs").iterdir() if p.is_dir() and p.name[:2] == "20")[-1]
    if a.t052:
        build_t052()
    if a.formal:
        build_formal(as_of)
    if a.shadow:
        build_shadow(sorted(p.name for p in (OUT / "books").iterdir() if p.is_dir() and p.name <= as_of)[-1])


if __name__ == "__main__":
    main()
