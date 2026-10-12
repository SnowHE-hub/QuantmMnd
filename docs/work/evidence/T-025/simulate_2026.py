"""T-025：10 万元散户账户 2026-01 → 2026-09-30 的**总收益近似**模拟（各策略样本外分数 + 成交约束；只读数据，写到 <out_dir>）.

性质（第三轮验收后明确，v2）：这是总收益近似，不是可执行的真实整数股账户。买入受整手与现金约束；持有期间按复权价计值，
除权除息相当于分红再投资、送转按比例增加持仓——所以除权后卖出的数量可以不是整数股。没有分红 / 送转数据，无法做真实整数股账本。

在 TurnoverControlledNavEngine（T-020）之上加散户约束（RetailNavEngine）：
  - 初始资金 100,000 元；净值单位 = 元
  - 买入按 100 股一手取整（科创板 688/689：最少 200 股，之后按 1 股）；钱不够一手则不买（记 lot_too_small）
  - 部分卖出按手取整（复权后的持仓可能有零头）；清仓卖全部
  - 佣金万 3、每笔最低 5 元（买入可行性按 max(最低佣金, 比例佣金) 只算一次，v2 修正）；过户费万 0.2；印花税按卖出日规则；滑点按 amihud 分档（同引擎）
其余同引擎：决策日收盘出信号、T+1 开盘成交、涨跌停 / 停牌不可成交、现金约束、退市强制清仓。

策略（事先固定，见 docs/work/tasks/T-025.md）：
  A_LGBM_63d  主线 A 的 LGBM（T-010 样本外分数，1374 票池），季度决策（2025-12-31、2026-03-31、2026-06-30）
  B_12d / B_20d / B_40d / B_63d   主线 B 的 Ridge（T-018 / T-023 方案 C 样本外分数），周频网格上每 3 / 4 / 8 / 13 期决策一次
  共同：选股池 = PIT 成交额前 1500；Top 20 等权；持仓偏离等权目标 ≤ 50% 不调（drift_band 0.5）；不设缓冲带
对照：沪深 300、中证 500、创业板指（指数收盘，无成本）；引擎自带的选股池等权基准（无成本）。
用法：PYTHONPATH=. python simulate_2026.py <data_root> <t010_oos_scores.csv> <out_dir> [--affordable]
--research（对照）：研究口径（原引擎、零碎股、B Top 300 / A Top 30），同一期间、同一 10 万起点——"大资金按系统设计执行"的近似。
--affordable（运行后增加的修正版，见卡片"运行后发现的问题"）：只在一手买得起的候选里取 Top N，按最接近的整手买入。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from quantmind.execution.nav_engine import (
    R_EXPIRED,
    R_INSUFFICIENT_CASH,
    R_T1_BLOCKED,
    ExecutableNavEngine,
    NavConfig,
)
from quantmind.execution.nav_turnover import TurnoverControlledNavEngine

CAPITAL = 100_000.0
TOP_N = 20
DRIFT = 0.5
MIN_COMMISSION = 5.0
START, END = pd.Timestamp("2025-12-31"), pd.Timestamp("2026-09-30")
R_LOT = "lot_too_small"


def lot_rule(ticker: str) -> tuple[int, int]:
    """(最小股数, 步长)。"""
    return (200, 1) if ticker[:3] in ("688", "689") else (100, 100)


def buy_total(n: float, p_raw: float, slip_bp: float, commission_bp: float, transfer_bp: float) -> float:
    """买 n 股的现金总支出：成交额 + 滑点 + 过户费 + max(最低佣金, 比例佣金)——佣金只算一次（v2 修正）。"""
    g = n * p_raw
    return g + g * (slip_bp + transfer_bp) / 1e4 + max(MIN_COMMISSION, g * commission_bp / 1e4)


def max_affordable(limit: float, p_raw: float, mn: int, step: int, slip_bp: float, commission_bp: float,
                   transfer_bp: float) -> int:
    """总支出 ≤ limit 的最大合法股数（≥ mn、按 step 取整）；买不起一手返回 0。

    v1 把比例佣金算进单价后又减了一次最低佣金（重复预留），现金恰好够一手时会误判买不起（Codex 第三轮反例）。"""
    if limit <= 0 or p_raw <= 0:
        return 0
    per = p_raw * (1 + (slip_bp + transfer_bp) / 1e4)
    hi = max(limit / (per + p_raw * commission_bp / 1e4), (limit - MIN_COMMISSION) / per)   # 两种佣金情形的上界
    n = int(np.floor(hi / step) * step) if step > 1 else int(np.floor(hi))
    if n >= mn:
        n = mn + (n - mn) // step * step
    while n >= mn and buy_total(n, p_raw, slip_bp, commission_bp, transfer_bp) > limit + 1e-9:
        n -= step
    return n if n >= mn else 0


class RetailNavEngine(TurnoverControlledNavEngine):
    """affordable=True（修正版）：决策日只在"一手 ≤ 每只预算"的候选里取 Top N（用决策日收盘价，时点安全），
    买入按最接近的整手（现金允许时向上取整）。affordable=False 为预先写定的原版（不过滤、向下取整）。"""

    affordable: bool = False

    def _select(self, ranked: list[str], as_of: pd.Timestamp) -> list[str]:
        if self.affordable:
            budget = self._nav(as_of) / self.cfg.top_n
            keep = []
            for t in ranked:
                if t in self.positions:
                    keep.append(t)
                    continue
                try:
                    close_raw = float(self.px.adj_close.at[as_of, t] / self.px.adj_factor.at[as_of, t])
                except KeyError:
                    continue
                if np.isfinite(close_raw) and lot_rule(t)[0] * close_raw <= budget:
                    keep.append(t)
            ranked = keep
        return super()._select(ranked, as_of)

    def _cost(self, gross: float, slip_bp: float, d: pd.Timestamp, side: str) -> float:
        cfg = self.cfg
        c = gross * (slip_bp + cfg.transfer_bp) / 1e4 + max(MIN_COMMISSION, gross * cfg.commission_bp / 1e4)
        if side == "sell":
            c += gross * (cfg.sell_cost_rate(0.0, d) - (cfg.commission_bp + cfg.transfer_bp) / 1e4)   # = 印花税
        return c

    def _execute(self, d: pd.Timestamp) -> None:
        sells = sorted((o for o in self.pending if o["side"] == "sell"), key=lambda o: o["ticker"])
        buys = sorted((o for o in self.pending if o["side"] == "buy"), key=lambda o: (o["rank"], o["ticker"]))
        remaining: list[dict] = []
        for o in sells:
            t = o["ticker"]
            pos = self.positions.get(t)
            if pos is None:
                continue
            if pos.buy_date == d:
                remaining.append(o)
                self.rejected.append({**self._order_row(o), "fill_date": None, "reason_code": R_T1_BLOCKED,
                                      "reject_date": str(d.date())})
                continue
            blocked = self.px.sell_blocked(d, t)
            if blocked:
                remaining.append(o)
                self.rejected.append({**self._order_row(o), "fill_date": None, "reason_code": blocked,
                                      "reject_date": str(d.date())})
                continue
            p_adj = float(self.px.open_adj.at[d, t])
            p_raw = float(self.px.open_raw.at[d, t])
            raw_held = pos.shares * p_adj / p_raw
            if o.get("sell_all"):
                raw = raw_held
            else:
                _, step = lot_rule(t)
                raw = min(raw_held, np.floor(float(o["sell_value"]) / p_raw / step) * step)
                if raw <= 0:
                    continue
            gross = raw * p_raw
            cost = self._cost(gross, o["slippage_bp"], d, "sell")
            shares_adj = gross / p_adj
            self.cash += gross - cost
            self.cum_cost += cost
            pos.shares -= shares_adj
            if pos.shares <= 1e-9 or o.get("sell_all"):
                del self.positions[t]
            self.trades.append({**self._order_row(o), "fill_date": str(d.date()), "fill_price_adj": p_adj,
                                "shares": float(shares_adj), "raw_shares": float(raw), "gross_value": float(gross),
                                "cost": float(cost), "cost_rate": float(cost / gross), "status": "filled"})
        for o in buys:
            t = o["ticker"]
            if o["expiry"] is not None and d >= o["expiry"]:
                self.rejected.append({**self._order_row(o), "fill_date": None, "reason_code": R_EXPIRED,
                                      "reject_date": str(d.date())})
                continue
            blocked = self.px.buy_blocked(d, t)
            if blocked:
                remaining.append(o)
                self.rejected.append({**self._order_row(o), "fill_date": None, "reason_code": blocked,
                                      "reject_date": str(d.date())})
                continue
            p_adj = float(self.px.open_adj.at[d, t])
            p_raw = float(self.px.open_raw.at[d, t])
            budget = min(float(o["target_value"]), self.cash)
            mn, step = lot_rule(t)
            slip, cb, tb = o["slippage_bp"], self.cfg.commission_bp, self.cfg.transfer_bp
            if self.affordable:   # 最接近的整手；现金不够时取现金允许的最大整手
                unit_cost = p_raw * (1 + (slip + cb + tb) / 1e4)
                raw = max(np.round(float(o["target_value"]) / unit_cost / step) * step, mn)
                if buy_total(raw, p_raw, slip, cb, tb) > self.cash:
                    raw = max_affordable(self.cash, p_raw, mn, step, slip, cb, tb)
            else:
                raw = max_affordable(budget, p_raw, mn, step, slip, cb, tb)
            if raw < mn:
                reason = R_INSUFFICIENT_CASH if buy_total(mn, p_raw, slip, cb, tb) > self.cash else R_LOT
                if reason == R_INSUFFICIENT_CASH:
                    remaining.append(o)
                self.rejected.append({**self._order_row(o), "fill_date": None, "reason_code": reason,
                                      "reject_date": str(d.date())})
                continue
            gross = raw * p_raw
            cost = self._cost(gross, o["slippage_bp"], d, "buy")
            if gross + cost > self.cash:
                raw -= step
                gross = raw * p_raw
                cost = self._cost(gross, o["slippage_bp"], d, "buy")
                if raw < mn:
                    continue
            self.cash -= gross + cost
            self.cum_cost += cost
            shares_adj = gross / p_adj
            pos = self.positions.get(t)
            if pos:
                pos.shares += shares_adj
                pos.buy_date = d
            else:
                from quantmind.execution.nav_engine import _Position
                self.positions[t] = _Position(shares=shares_adj, buy_date=d)
            self.trades.append({**self._order_row(o), "fill_date": str(d.date()), "fill_price_adj": p_adj,
                                "shares": float(shares_adj), "raw_shares": float(raw), "gross_value": float(gross),
                                "cost": float(cost), "cost_rate": float(cost / gross), "status": "filled"})
        self.pending = remaining


def monthly(nav: pd.Series) -> pd.Series:
    m = nav.resample("ME").last()
    first = nav.iloc[0]
    return m.pct_change().fillna(m.iloc[0] / first - 1)


def main() -> None:
    root, a_scores, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    affordable = "--affordable" in sys.argv[4:]
    research = "--research" in sys.argv[4:]
    out_dir.mkdir(parents=True, exist_ok=True)
    D = root / "data"
    cols = ["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "adj_factor", "amount"]
    prices = pd.read_parquet(D / "raw/alpha_prices_panel_v6_20261009.parquet", columns=cols)
    prices = prices[pd.to_datetime(prices["trade_date"]) <= END]
    am = pd.read_parquet(D / "panel/alpha_panel_weekly_v6_1.parquet", columns=["amihud_illiquidity"])
    grid = am.index.get_level_values("as_of").unique().sort_values()

    strategies = {}
    a = pd.read_csv(a_scores, parse_dates=["as_of"])
    a = a[(a.as_of >= START) & (a.as_of <= END)]
    strategies["A_LGBM_63d"] = (a, 63, 1)
    for s, step in (("12d", 3), ("20d", 4), ("40d", 8), ("63d", 13)):
        p = pd.read_parquet(D / f"bakeoff/preds_v6_1/ridge_{s}_v6_1_C.parquet")
        p["as_of"] = pd.to_datetime(p["as_of"])
        p = p[(p.as_of >= START) & (p.as_of <= END)]
        strategies[f"B_{s}"] = (p, int(s[:-1]), step)

    # A 的季度决策日不在周频网格上：用 ≤ 该日的最近网格日的 amihud（时点安全）
    a_dates = sorted(a.as_of.unique())
    am_a = []
    for dt in a_dates:
        g = grid[grid <= dt][-1]
        x = am.xs(g, level="as_of").copy()
        x.index = pd.MultiIndex.from_product([[pd.Timestamp(dt)], x.index], names=["as_of", "ticker"])
        am_a.append(x)
    am_a = pd.concat(am_a)

    navs, summary, monthly_tab = {}, {}, {}
    for name, (preds, h, step) in strategies.items():
        if research:   # 研究口径：原引擎、零碎股、B Top 300 / A Top 30（T-010 口径），每期调回等权
            cfg = NavConfig(horizon_td=h, rebalance_step=step, top_n=30 if name.startswith("A_") else 300,
                            universe_size=1500, initial_nav=CAPITAL)
            eng = ExecutableNavEngine(preds, prices, cfg, amihud=am_a if name.startswith("A_") else am)
            eng.cum_cost = 0.0
        else:
            cfg = NavConfig(horizon_td=h, rebalance_step=step, top_n=TOP_N, universe_size=1500, initial_nav=CAPITAL)
            eng = RetailNavEngine(preds, prices, cfg, amihud=am_a if name.startswith("A_") else am, drift_band=DRIFT)
            eng.affordable = affordable
        r = eng.run()
        nav = r["nav_daily"].set_index("date")
        nav = nav[nav.index <= END]
        navs[name] = nav
        tr = r["trades"]
        rj = r["rejected_trades"]
        s = nav["nav_net"]
        summary[name] = {
            "decisions": [str(pd.Timestamp(x).date()) for x in eng.rebalance_dates],
            "start_nav_date": str(s.index[0].date()), "final_value": round(float(s.iloc[-1]), 2),
            "total_return": round(float(s.iloc[-1] / CAPITAL - 1), 4),
            "max_drawdown": round(float((s / s.cummax() - 1).min()), 4),
            "total_cost_rmb": round(float(eng.cum_cost), 2), "n_trades": int(len(tr)),
            "rejections": rj["reason_code"].value_counts().to_dict() if len(rj) else {},
            "avg_positions": round(float(nav["n_positions"].mean()), 1),
            "avg_cash_share": round(float((nav["cash"] / nav["nav_net"]).mean()), 4),
            "bench_pool_ew_return": round(float(nav["nav_bench"].iloc[-1] / nav["nav_bench"].iloc[0] - 1), 4)}
        monthly_tab[name] = monthly(s)
        last = r["holdings_daily"]
        last = last[last["date"] == last["date"].max()].sort_values("value", ascending=False)
        last[["ticker", "value"]].round(2).to_csv(out_dir / f"holdings_{name}_last.csv", index=False)
        tr.to_csv(out_dir / f"trades_{name}.csv", index=False)
        nav[["nav_net", "nav_bench", "cash", "n_positions", "cum_cost"]].to_csv(out_dir / f"nav_{name}.csv")
        print(name, json.dumps({k: summary[name][k] for k in ("final_value", "total_return", "max_drawdown",
                                                              "total_cost_rmb", "n_trades")}), flush=True)
    # 对照指数（无成本，2025-12-31 收盘 = 起点）
    idx = pd.read_parquet(D / "lake/index_daily.parquet", columns=["ts_code", "trade_date", "close"])
    idx["trade_date"] = pd.to_datetime(idx["trade_date"].astype(str))
    names = {"000300.SH": "沪深300", "000905.SH": "中证500", "399006.SZ": "创业板指"}
    for code, nm in names.items():
        s = idx[(idx.ts_code == code) & (idx.trade_date >= START) & (idx.trade_date <= END)].set_index("trade_date")["close"].sort_index()
        monthly_tab[nm] = monthly(s)
        summary[nm] = {"total_return": round(float(s.iloc[-1] / s.iloc[0] - 1), 4),
                       "max_drawdown": round(float((s / s.cummax() - 1).min()), 4)}
    pool = navs["B_63d"]["nav_bench"]
    monthly_tab["选股池等权(无成本)"] = monthly(pool)
    tab = pd.DataFrame(monthly_tab)
    tab.index = tab.index.strftime("%Y-%m")
    tab.round(4).to_csv(out_dir / "monthly_returns.csv")
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print((tab * 100).round(2).to_string())


if __name__ == "__main__":
    main()
