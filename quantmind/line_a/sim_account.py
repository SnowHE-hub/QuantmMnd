"""quantmind.line_a.sim_account — 按股数记账的模拟账户（T-052；不改冻结的 nav_engine）.

和 `ExecutableNavEngine` 的决策与成交顺序相同（决策日收盘出单、次日起成交、先卖后买、买单按排名、买单到下一调仓日失效、
卖单直到卖完），另外按真实账户记账：
- 整数手：主板 / 创业板 100 股整数倍；科创板买入至少 200 股、超出按 1 股递增；北交所至少 100 股、按 1 股递增；卖出零股一次卖完；
- 佣金（可设单笔最低）、过户费、印花税（按成交日）、滑点（与引擎同样按决策截面 amihud 分位三档）；
- 涨跌停：主板 10%、创业板 / 科创板 20%、北交所 30%，主板风险警示股 2026-07-06 前 5%、之后 10%；
- 参与率：单只股票当日成交股数 ≤ 当日成交量 × participation；
- 分红送转：除权日现金红利入账、送转增加股数，红利税在卖出时按持有期补扣（≤ 1 个月 20%，1–12 个月 10%）；
- 账户权限：没开通的板块不新买；
- 研究口径（mode = "adj"）：用复权价、不取整、不设最低佣金、不处理分红（复权已含），用来和冻结引擎对照。
所有决策只用决策日及之前的数据。只读研究，不形成交易指令。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
import pandas as pd

from quantmind.backtest.wf_costs import CHINEXT_20PCT_DATE, SlippageTiers, amihud_to_quantile, stamp_duty_rate
from quantmind.backtest.wf_costs import board_of as _board_of_frozen

ST_LIMIT_10PCT_FROM = pd.Timestamp("2026-07-06")     # 沪深主板风险警示股涨跌幅 5% → 10%（2026 年修订交易规则）
LOT = {"MAIN": (100, 100), "CHINEXT": (100, 100), "STAR": (200, 1), "BSE": (100, 1)}


def board_of(ticker: str) -> str:
    """在冻结的 wf_costs.board_of 之上补两类代码（T-053 审查）：689 开头是科创板 CDR，92 开头 / .BJ 后缀是北交所。
    冻结的函数不改（引擎在用）。"""
    t = str(ticker)
    if t.startswith("689"):
        return "STAR"
    if t.startswith("92") or t.endswith(".BJ"):
        return "BSE"
    return _board_of_frozen(t)


def limit_price(pre: float, pct: float, up: bool) -> float:
    """涨跌停价：昨收 × (1 ± 幅度)，十进制四舍五入到分（T-053：Python 的 round 对 10.25 × 0.9 给 9.22，交易所是 9.23）。"""
    k = Decimal(repr(pct))
    v = Decimal(repr(float(pre))) * (Decimal(1) + k if up else Decimal(1) - k)
    return float(v.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


@dataclass
class SimConfig:
    aum: float = 1e6
    top_n: int = 30
    rebalance_step: int = 4                  # 每几个决策网格点调一次仓
    universe_size: int | None = 1500         # PIT adv20 前 N；None = 不限
    buffer_mult: float | None = 2.0          # 持仓仍在前 buffer_mult × N 就不卖；None = 不缓冲
    drift_band: float = 0.0                  # 留在目标里的持仓，偏离目标金额超过这个比例才调；0 = 每次都调平（登记口径）
    exec_price: str = "open"                 # open / vwap
    commission_rate: float = 2.5e-4
    min_commission: float = 5.0
    transfer_rate: float = 1e-5
    tiers: SlippageTiers = field(default_factory=SlippageTiers)
    participation: float | None = 0.05
    boards: frozenset = frozenset({"MAIN", "CHINEXT", "STAR", "BSE"})
    mode: str = "raw"                        # raw：真实账户；adj：研究口径（与冻结引擎对照）
    quota_ext: tuple[float, float] | None = None   # (池外占比, adv20 下限元)：T-042 配额臂
    min_recent_bar_td: int = 10
    delist_writeoff_td: int = 20
    dividend_tax: bool = True
    weight_fn: object = None                 # (决策日, 目标列表) → {股票: 权重}；None = 等权（T-055 X3 v2 的逆波动加权）


def lot_shares(value: float, price: float, board: str, *, integer: bool = True) -> float:
    """value 元最多能买的股数（按板块的最小申报单位取整）；integer = False 时不取整（研究口径）。"""
    if not (price > 0) or value <= 0:
        return 0.0
    if not integer:
        return value / price
    lo, step = LOT.get(board, (100, 100))
    n = math.floor(value / price + 1e-9)
    if n < lo:
        return 0.0
    return float(lo + ((n - lo) // step) * step)


def limit_pct(ticker: str, d: pd.Timestamp, is_st: bool) -> float:
    b = board_of(ticker)
    if b == "STAR":
        return 0.20
    if b == "BSE":
        return 0.30
    if b == "CHINEXT":
        return 0.20 if d >= pd.Timestamp(CHINEXT_20PCT_DATE) else 0.10
    if is_st and d < ST_LIMIT_10PCT_FROM:
        return 0.05
    return 0.10


def dividend_tax_rate(buy_date: pd.Timestamp, sell_date: pd.Timestamp) -> float:
    days = (pd.Timestamp(sell_date) - pd.Timestamp(buy_date)).days
    return 0.20 if days <= 30 else 0.10 if days <= 365 else 0.0


@dataclass
class _Pos:
    shares: float
    first_buy: pd.Timestamp
    last_buy: pd.Timestamp
    div_received: float = 0.0               # 尚未扣税的现金红利（元）


class Market:
    """宽表价格（float64，避免涨跌停价比较时的精度误差），一次构建、多次模拟共用。"""

    def __init__(self, prices: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp):
        p = prices.copy()
        p["trade_date"] = pd.to_datetime(p["trade_date"]).dt.normalize()
        p = p[(p["trade_date"] >= start - pd.Timedelta(days=90)) & (p["trade_date"] <= end)]
        p = p.drop_duplicates(["trade_date", "ts_code"], keep="last")

        def wide(c):
            return p.pivot(index="trade_date", columns="ts_code", values=c).sort_index().astype("float64")
        self.open, self.high, self.low, self.close = wide("open"), wide("high"), wide("low"), wide("close")
        self.pre = wide("pre_close")
        self.adj = wide("adj_factor").ffill()
        self.vol = wide("vol") * 100.0                           # 手 → 股
        self.amount = wide("amount") * 1000.0                    # 千元 → 元
        self.adv20 = self.amount.rolling(20, min_periods=10).mean()
        self.vwap = (self.amount / self.vol).where(self.vol > 0)
        self.cal = self.close.index
        self.close_ffill = self.close.ffill()
        self.adjc_ffill = (self.close * self.adj).ffill()
        self.has_bar = self.open.notna()
        self.last_bar = self.close.apply(lambda c: c.last_valid_index())
        self.ret_adj = (self.close * self.adj).pct_change(fill_method=None)


class AccountSimulator:
    def __init__(self, scores: pd.DataFrame, prices: pd.DataFrame | Market, cfg: SimConfig, *, start: pd.Timestamp,
                 end: pd.Timestamp, amihud: pd.DataFrame | None = None, dividends: pd.DataFrame | None = None,
                 st_intervals: pd.DataFrame | None = None):
        self.cfg = cfg
        s = scores.copy()
        s["as_of"] = pd.to_datetime(s["as_of"]).dt.normalize()
        s = s[(s["as_of"] >= start) & (s["as_of"] <= end)]
        self.scores = {d: g.set_index("ticker")["score"] for d, g in s.groupby("as_of")}
        grid = sorted(self.scores)
        self.rebal = list(grid[:: cfg.rebalance_step])
        m = prices if isinstance(prices, Market) else Market(prices, start, end)
        for k in ("open", "high", "low", "close", "pre", "adj", "vol", "amount", "adv20", "vwap", "cal", "close_ffill",
                  "adjc_ffill", "has_bar", "last_bar", "ret_adj"):
            setattr(self, k, getattr(m, k))
        self.amihud = amihud
        self.divs = {}
        if dividends is not None and len(dividends) and cfg.mode == "raw":
            dv = dividends[dividends["div_proc"].astype(str) == "实施"].drop_duplicates(["ts_code", "ex_date"]).copy()
            dv["ex_date"] = pd.to_datetime(dv["ex_date"].astype(str), errors="coerce")
            dv = dv[dv["ex_date"].notna()]
            for d, g in dv.groupby("ex_date"):
                self.divs[pd.Timestamp(d)] = {r.ts_code: (float(r.cash_div_tax or 0.0), float(r.stk_div or 0.0))
                                              for r in g.itertuples()}
        self.st: dict[str, list] = {}                # ts_code → [(起, 止)]：名称含 ST 的区间（namechange，按时点）
        if st_intervals is not None:
            for r in st_intervals.itertuples():
                self.st.setdefault(r.ts_code, []).append((r.s, r.e))
        self.start, self.end = start, end
        # 状态
        self.cash = float(cfg.aum)
        self.pos: dict[str, _Pos] = {}
        self.pending: list[dict] = []
        self.trades, self.rejects, self.nav_rows, self.targets, self.hold_rows = [], [], [], [], []
        self.costs = {"commission": 0.0, "transfer": 0.0, "stamp": 0.0, "slippage": 0.0, "dividend_tax": 0.0}
        self.bench_nav, self.bench_members = 1.0, []

    # ------------------------------------------------------------------ 价格与限制
    def _px(self, kind: str, d, t) -> float:
        if self.cfg.mode == "adj":
            base = self.open if kind == "open" else self.vwap
            v = base.at[d, t] * self.adj.at[d, t]
        else:
            v = (self.open if kind == "open" else self.vwap).at[d, t]
        return float(v)

    def _is_st(self, t: str, d: pd.Timestamp) -> bool:
        return any(s <= d <= e for s, e in self.st.get(t, ()))

    def _blocked(self, side: str, d, t) -> str | None:
        if not bool(self.has_bar.at[d, t]) if t in self.has_bar.columns else True:
            return "suspended_no_bar"
        pre, hi, lo = float(self.pre.at[d, t]), float(self.high.at[d, t]), float(self.low.at[d, t])
        pct = limit_pct(t, d, self.cfg.mode == "raw" and self._is_st(t, d))
        if side == "buy" and (hi >= limit_price(pre, pct, True) - 1e-9 or (hi == lo and hi > pre)):
            return "limit_up_locked"
        if side == "sell" and (lo <= limit_price(pre, pct, False) + 1e-9 or (hi == lo and hi < pre)):
            return "limit_down_locked"
        return None

    def _value(self, d) -> float:
        px = self.adjc_ffill if self.cfg.mode == "adj" else self.close_ffill
        row = px.loc[d]
        return self.cash + sum(p.shares * float(row.get(t, np.nan)) for t, p in self.pos.items()
                               if pd.notna(row.get(t, np.nan)))

    def _slip(self, d, names) -> dict[str, float]:
        tiers = self.cfg.tiers
        if self.amihud is None:
            return dict.fromkeys(names, tiers.small_bp)
        try:
            cs = self.amihud.xs(d, level="as_of")["amihud_illiquidity"]
        except KeyError:
            return dict.fromkeys(names, tiers.small_bp)
        q = amihud_to_quantile(cs.reindex(names))
        return {t: tiers.bp_for_quantile(q.get(t, np.nan)) for t in names}

    def _fees(self, gross: float, side: str, d, slip_bp: float) -> dict[str, float]:
        c = self.cfg
        comm = gross * c.commission_rate
        if c.mode == "raw":
            comm = max(comm, c.min_commission)
        return {"commission": comm, "transfer": gross * c.transfer_rate, "slippage": gross * slip_bp / 1e4,
                "stamp": gross * stamp_duty_rate(d) if side == "sell" else 0.0}

    # ------------------------------------------------------------------ 决策
    def _universe(self, d) -> set[str]:
        if self.cfg.universe_size is None:
            return set(self.close.columns)
        row = self.adv20.loc[:d].iloc[-1].dropna()
        return set(row.sort_values(ascending=False).head(self.cfg.universe_size).index)

    def _decide(self, d) -> None:
        c = self.cfg
        sc = self.scores[d]
        i = self.cal.get_loc(d)
        recent = self.has_bar.iloc[max(0, i - c.min_recent_bar_td + 1): i + 1].any()
        ok = [t for t in sc.index if t in recent.index and bool(recent[t]) and board_of(t) in c.boards]
        uni = self._universe(d)
        ranked = sc.loc[ok].sort_values(ascending=False, kind="mergesort")
        if c.quota_ext is None:
            segments = [(ranked[ranked.index.isin(uni)], c.top_n)]
        else:
            share, floor = c.quota_ext
            adv = self.adv20.loc[:d].iloc[-1]
            n_ext = int(round(c.top_n * share))
            inside = ranked[ranked.index.isin(uni)]
            outside = ranked[~ranked.index.isin(uni) & ranked.index.map(lambda t: adv.get(t, 0) >= floor)]
            segments = [(inside, c.top_n - n_ext), (outside, n_ext)]
        target: list[str] = []
        for seg, n in segments:
            names = list(seg.index)
            keep = []
            if c.buffer_mult:
                top_buf = set(names[: int(math.ceil(n * c.buffer_mult))])
                keep = [t for t in names if t in self.pos and t in top_buf][:n]
            fill = [t for t in names if t not in keep][: max(0, n - len(keep))]
            target += keep + fill
        self.bench_members = sorted(uni)
        px = self.adjc_ffill if c.mode == "adj" else self.close_ffill
        equity = self._value(d)
        per = equity / max(c.top_n, 1)
        wts = c.weight_fn(d, target) if c.weight_fn is not None else None
        tv = {t: equity * wts.get(t, 0.0) for t in target} if wts else {t: per for t in target}
        later = [r for r in self.rebal if r > d]
        expiry = later[0] if later else None
        self.pending = [o for o in self.pending if not (o["side"] == "buy" and o["ticker"] not in target)
                        and not (o["side"] == "sell" and o["ticker"] in target)]
        pend = {(o["side"], o["ticker"]) for o in self.pending}
        slip = self._slip(d, sorted(set(target) | set(self.pos)))
        for o in self.pending:                      # 仍在目标里的未成交买单：刷新失效日和金额（与引擎相同）
            if o["side"] == "buy":
                t = o["ticker"]
                cur = self.pos[t].shares * float(px.at[d, t]) if t in self.pos else 0.0
                o.update(expiry=expiry, rank=target.index(t), value=max(tv[t] - cur, 0.0), decided=d)
        for t in sorted(set(target) | set(self.pos)):
            cur = self.pos[t].shares * float(px.at[d, t]) if t in self.pos else 0.0
            want = tv[t] if t in target else 0.0
            delta = want - cur
            if t in target and t in self.pos and abs(delta) <= c.drift_band * tv[t]:
                continue
            if abs(delta) < 1e-6 * equity:
                continue
            side = "buy" if delta > 0 else "sell"
            if (side, t) in pend:
                continue
            self.pending.append({"side": side, "ticker": t, "decided": d, "value": abs(delta),
                                 "sell_all": side == "sell" and want == 0.0, "slip_bp": slip.get(t, c.tiers.small_bp),
                                 "expiry": expiry if side == "buy" else None,
                                 "rank": target.index(t) if t in target else -1})
        self.targets.append({"as_of": d, "targets": target, "equity": equity, "per_name": per})
        self._snapshot(d)

    # ------------------------------------------------------------------ 成交
    def _part(self, d, t, shares: float) -> float | None:
        v = self.vol.at[d, t] if t in self.vol.columns else np.nan
        return float(shares / v) if pd.notna(v) and v > 0 else None

    def _cap(self, d, t, shares: float) -> float:
        p = self.cfg.participation
        if p is None or self.cfg.mode == "adj":
            return shares
        v = self.vol.at[d, t]
        return min(shares, math.floor(p * float(v))) if pd.notna(v) else 0.0

    def _execute(self, d) -> None:
        c = self.cfg
        integer = c.mode == "raw"
        sells = sorted((o for o in self.pending if o["side"] == "sell"), key=lambda o: o["ticker"])
        buys = sorted((o for o in self.pending if o["side"] == "buy"), key=lambda o: (o["rank"], o["ticker"]))
        keep = []
        for o in sells:
            t = o["ticker"]
            p = self.pos.get(t)
            if p is None:
                continue
            if p.last_buy == d:
                keep.append(o)
                self.rejects.append({"date": d, "ticker": t, "side": "sell", "reason": "t_plus_1_blocked"})
                continue
            why = self._blocked("sell", d, t)
            if why:
                keep.append(o)
                self.rejects.append({"date": d, "ticker": t, "side": "sell", "reason": why})
                continue
            price = self._px(c.exec_price, d, t)
            if not price > 0:
                keep.append(o)
                continue
            want = p.shares if o["sell_all"] else min(p.shares, o["value"] / price)
            step = LOT.get(board_of(t), (100, 100))[1]
            if integer and not o["sell_all"]:
                want = math.floor(want / step) * step
                if want <= 0:                          # 部分卖出不足一个申报单位：没有可卖数量，订单终止（T-053）
                    self.rejects.append({"date": d, "ticker": t, "side": "sell", "reason": "below_one_lot"})
                    continue
            qty = self._cap(d, t, want)
            if integer and qty < want:                 # 被参与率截断时按申报单位取整；零股留到最后一次卖完
                qty = math.floor(qty / step) * step
            if qty <= 0:
                keep.append(o)
                self.rejects.append({"date": d, "ticker": t, "side": "sell", "reason": "participation_cap"})
                continue
            gross = qty * price
            fee = self._fees(gross, "sell", d, o["slip_bp"])
            tax = 0.0
            if c.mode == "raw" and c.dividend_tax and p.div_received > 0:
                frac = qty / p.shares
                tax = p.div_received * frac * dividend_tax_rate(p.first_buy, d)
                tax = min(tax, max(gross - sum(fee.values()), 0.0))     # 不让税把现金扣成负数（T-053）
                p.div_received -= p.div_received * frac
            fee["dividend_tax"] = tax
            cost = sum(fee.values())
            self.cash += gross - cost
            for k, v in fee.items():
                self.costs[k] += v
            p.shares -= qty
            self.trades.append({"date": d, "ticker": t, "side": "sell", "shares": qty, "price": price, "gross": gross,
                                "cost": cost, "decided": o["decided"], "participation": self._part(d, t, qty)})
            if p.shares <= 1e-9:
                del self.pos[t]
            elif o["sell_all"] or qty < want:
                keep.append(o | {"value": (o["value"] - gross) if not o["sell_all"] else o["value"]})
        for o in buys:
            t = o["ticker"]
            if o["expiry"] is not None and d >= o["expiry"]:
                self.rejects.append({"date": d, "ticker": t, "side": "buy", "reason": "expired_unfilled"})
                continue
            why = self._blocked("buy", d, t)
            if why:
                keep.append(o)
                self.rejects.append({"date": d, "ticker": t, "side": "buy", "reason": why})
                continue
            price = self._px(c.exec_price, d, t)
            if not price > 0:
                keep.append(o)
                continue
            rate = c.commission_rate + c.transfer_rate + o["slip_bp"] / 1e4
            budget = min(o["value"], self.cash)
            qty = lot_shares(budget / (1 + rate), price, board_of(t), integer=integer)
            if integer:                                     # 最低佣金也要够钱
                while qty > 0 and qty * price + sum(self._fees(qty * price, "buy", d, o["slip_bp"]).values()) > self.cash:
                    lo, step = LOT.get(board_of(t), (100, 100))
                    qty = qty - step if qty - step >= lo else 0.0
            capped = self._cap(d, t, qty)
            if integer and capped < qty:
                capped = lot_shares(capped * price, price, board_of(t))
            if capped <= 0:
                reason = "participation_cap" if qty > 0 else ("insufficient_cash" if budget < o["value"] * 0.999
                                                               else "cannot_afford_one_lot")
                self.rejects.append({"date": d, "ticker": t, "side": "buy", "reason": reason})
                # 真实账户里资金不够的委托会被拒，不留到第二天（T-053）；研究口径保持引擎的做法（现金为 0 时留着）
                if reason == "participation_cap" or (reason == "insufficient_cash" and not integer):
                    keep.append(o)
                continue
            gross = capped * price
            fee = self._fees(gross, "buy", d, o["slip_bp"])
            cost = sum(fee.values())
            self.cash -= gross + cost
            for k, v in fee.items():
                self.costs[k] += v
            p = self.pos.get(t)
            if p:
                p.shares += capped
                p.last_buy = d
            else:
                self.pos[t] = _Pos(capped, d, d)
            self.trades.append({"date": d, "ticker": t, "side": "buy", "shares": capped, "price": price, "gross": gross,
                                "cost": cost, "decided": o["decided"], "participation": self._part(d, t, capped)})
            left = o["value"] - gross - cost
            if capped < qty and left > price * LOT.get(board_of(t), (100, 100))[0]:   # 参与率限制：剩余部分明天接着买
                keep.append(o | {"value": left})
        self.pending = keep

    def _snapshot(self, d) -> None:
        px = self.adjc_ffill if self.cfg.mode == "adj" else self.close_ffill
        eq = self._value(d)
        for t, p in sorted(self.pos.items()):
            v = p.shares * float(px.at[d, t])
            self.hold_rows.append({"date": d, "ticker": t, "board": board_of(t), "shares": p.shares,
                                   "price": float(px.at[d, t]), "value": v, "weight": v / eq if eq else np.nan})

    def _corporate_actions(self, d) -> None:
        ev = self.divs.get(d)
        if not ev:
            return
        for t, p in self.pos.items():
            if t in ev:
                cash, stk = ev[t]
                if cash > 0:
                    amt = p.shares * cash
                    self.cash += amt
                    p.div_received += amt
                if stk > 0:
                    p.shares *= 1 + stk

    def _writeoff(self, d) -> None:
        i = self.cal.get_loc(d)
        for t in list(self.pos):
            lb = self.last_bar.get(t)
            if lb is not None and pd.notna(lb) and lb < d and (i - self.cal.get_loc(lb)) >= self.cfg.delist_writeoff_td:
                px = float((self.adjc_ffill if self.cfg.mode == "adj" else self.close_ffill).at[d, t])
                gross = self.pos[t].shares * px                       # 与引擎相同：按末价清仓，计卖出成本（小盘档滑点）
                fee = self._fees(gross, "sell", d, self.cfg.tiers.small_bp) if gross > 0 else {}
                p = self.pos[t]
                if self.cfg.mode == "raw" and self.cfg.dividend_tax and p.div_received > 0 and fee:   # 清仓也算卖出（T-053）
                    fee["dividend_tax"] = min(p.div_received * dividend_tax_rate(p.first_buy, d),
                                              max(gross - sum(fee.values()), 0.0))
                for k, v in fee.items():
                    self.costs[k] += v
                self.cash += gross - sum(fee.values())
                self.trades.append({"date": d, "ticker": t, "side": "writeoff", "shares": self.pos[t].shares,
                                    "price": px, "gross": gross, "cost": sum(fee.values()), "decided": None})
                del self.pos[t]
                self.pending = [o for o in self.pending if not (o["side"] == "sell" and o["ticker"] == t)]

    # ------------------------------------------------------------------ 主循环
    def run(self) -> dict:
        if not self.rebal:
            raise ValueError("no decision dates in window")
        cal = self.cal[(self.cal >= self.rebal[0]) & (self.cal <= self.end)]
        rebal = set(self.rebal)
        ret = self.ret_adj
        prev = None
        for d in cal:
            if self.cfg.mode == "raw":
                self._corporate_actions(d)
            if self.pending:
                self._execute(d)
            if self.pos:
                self._writeoff(d)
            if d in rebal:
                self._decide(d)
            if prev is not None and self.bench_members:
                r = ret.loc[d, [t for t in self.bench_members if t in ret.columns]].dropna()
                if len(r):
                    self.bench_nav *= 1 + float(r.mean())
            eq = self._value(d)
            px = self.adjc_ffill if self.cfg.mode == "adj" else self.close_ffill
            invested = eq - self.cash
            self.nav_rows.append({"date": d, "equity": eq, "cash": self.cash, "invested": invested,
                                  "n_pos": len(self.pos), "bench_pool_ew": self.bench_nav,
                                  "max_weight": max((p.shares * float(px.at[d, t]) for t, p in self.pos.items()
                                                     if pd.notna(px.at[d, t])), default=0.0) / eq if eq else 0.0})
            prev = d
        self._snapshot(cal[-1])
        nav = pd.DataFrame(self.nav_rows).set_index("date")
        rej = pd.DataFrame(self.rejects)
        summ = summarize(nav, self.cfg.aum, pd.DataFrame(self.trades), self.costs)
        summ["rejects_by_reason"] = rej["reason"].value_counts().to_dict() if len(rej) else {}
        summ["n_cannot_afford_one_lot"] = int(summ["rejects_by_reason"].get("cannot_afford_one_lot", 0))
        return {"nav": nav, "trades": pd.DataFrame(self.trades), "rejects": rej, "holdings": pd.DataFrame(self.hold_rows),
                "targets": pd.DataFrame(self.targets), "costs": dict(self.costs), "summary": summ}


def summarize(nav: pd.DataFrame, aum: float, trades: pd.DataFrame, costs: dict) -> dict:
    eq = nav["equity"]
    r = eq.pct_change().dropna()
    b = nav["bench_pool_ew"].pct_change().dropna()
    yrs = max(len(nav) - 1, 1) / 252
    tot = float(eq.iloc[-1] / aum - 1)
    btot = float(nav["bench_pool_ew"].iloc[-1] / nav["bench_pool_ew"].iloc[0] - 1)
    q = (eq / aum) / (nav["bench_pool_ew"] / nav["bench_pool_ew"].iloc[0])
    gross_traded = float(trades.loc[trades["side"].isin(["buy", "sell"]), "gross"].sum()) if len(trades) else 0.0
    return {"days": int(len(nav)), "total_return": tot, "ann_return": (1 + tot) ** (1 / yrs) - 1,
            "bench_pool_ew_return": btot, "excess_vs_pool": tot - btot,
            "max_drawdown": float((eq / eq.cummax() - 1).min()),
            "max_rel_drawdown": float((1 - q / q.cummax()).max()),
            "vol_ann": float(r.std() * np.sqrt(252)), "ir_vs_pool": float((r - b).mean() / (r - b).std() * np.sqrt(252))
            if (r - b).std() > 0 else 0.0,
            "turnover_x": gross_traded / 2 / aum, "cost_total": float(sum(costs.values())),
            "cost_share_of_aum": float(sum(costs.values()) / aum), "costs": {k: float(v) for k, v in costs.items()},
            "avg_cash_share": float((nav["cash"] / nav["equity"]).mean()),
            "max_participation": float(trades["participation"].max()) if len(trades) and "participation" in trades
            and trades["participation"].notna().any() else None,
            "n_trades": int(trades["side"].isin(["buy", "sell"]).sum()) if len(trades) else 0}
