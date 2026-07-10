"""quantmind.execution.nav_engine — Executable NAV 组合回测引擎（阶段 1）.

实现 ``docs/plans/executable_nav_design.md`` §2 的可执行 NAV：
**真实 holdings 驱动**——每个再平衡日基于「上一期真实持仓 + 当日可成交集 + Top-N 目标」
解出真实成交；NAV 由真实成交价与持仓逐日 mark-to-market 推进；换手/成本按真实持仓变化计。

复用（§4）：
- ``wf_costs.price_limit_pct``  板块×时变涨跌停（2.1）
- ``wf_costs.SlippageTiers`` + amihud 分位 → 逐笔滑点（2.4）
- ``wf_costs.stamp_duty_rate``  时变印花税（卖出侧）
- T+1 next-open 语义（2.3）：as_of 收盘决策 → 次一交易日开盘成交

新写（§4 明确"新写"项）：可成交集过滤（停牌/涨跌停/退市）、Top-N 组合状态机、真实 NAV 推进。

关键口径（设计缺口的保守可配置实现，见 gate 报告"设计缺口"节）：
- Top-N 默认 300 = PIT top-1500 选股池的 top-quintile，与研究层 net_excess 口径对齐；
- 基准 = PIT top-1500 选股池等权（无成本），与研究层 bench 对齐；
- 无 PIT ST 标记 → 涨跌停按板块阈值 + 一字板（high==low）兜底检测；
- 持仓票 >delist_writeoff_td 个交易日无 bar 且数据集中无未来 bar → 按末价强制清仓（计卖出成本）；
- gross NAV = net NAV + 累计成本（NAV 单位），成本影响不改变可成交性判定。

所有价格统一用后复权口径（open_adj = open×adj_factor；mtm 用 adj_close），
涨跌停检测用原始价（交易所口径，四舍五入 2 位）。
纯 pandas/numpy 确定性计算，无随机源；重复运行结果一致。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from quantmind.backtest.wf_costs import (
    SlippageTiers,
    amihud_to_quantile,
    price_limit_pct,
    stamp_duty_rate,
)

# ============================================================================
# 配置
# ============================================================================


@dataclass(frozen=True)
class NavConfig:
    """一次 executable NAV 回测的全部参数（事前定，事后不调）。"""

    horizon_td: int                    # 12 | 63
    rebalance_step: int                # as_of 网格步长：12d→3, 63d→13（设计 §2.5）
    top_n: int = 300                   # 组合股票数（默认=top-quintile of 1500）
    universe_size: int = 1500          # PIT adv20 选股池大小
    commission_bp: float = 3.0         # 佣金 万3（双边）
    transfer_bp: float = 0.2           # 过户费 万0.2（双边，从研究口径：全票适用）
    tiers: SlippageTiers = field(default_factory=SlippageTiers)
    initial_nav: float = 1.0
    delist_writeoff_td: int = 20       # 无 bar 且无未来数据 → 强制清仓阈值（交易日）
    min_recent_bar_td: int = 10        # 决策日筛选：近 N 交易日内须有 bar（否则视为停牌/退市中）
    adv20_min_periods: int = 10        # 与 p4d _adv20_top1500_mask 口径一致

    def buy_cost_rate(self, slip_bp: float) -> float:
        return (slip_bp + self.commission_bp + self.transfer_bp) / 1e4

    def sell_cost_rate(self, slip_bp: float, fill_date) -> float:
        return (slip_bp + self.commission_bp + self.transfer_bp) / 1e4 + stamp_duty_rate(fill_date)


# ============================================================================
# 拒单 / 事件 reason codes（每笔未成交必须有 code）
# ============================================================================

R_SUSPENDED = "suspended_no_bar"          # 当日无 bar（停牌）
R_LIMIT_UP = "limit_up_locked"            # high 触及涨停价，不可买（设计 2.1）
R_LIMIT_DOWN = "limit_down_locked"        # low 触及跌停价，不可卖
R_ONE_LINE = "one_line_board"             # 一字板兜底（无 ST 标记的保守检测）
R_INSUFFICIENT_CASH = "insufficient_cash" # 现金不足，当日未成交（次日重试）
R_EXPIRED = "expired_unfilled"            # 买单到期（下一再平衡日）仍未成交 → 撤单
R_CANCELLED_REBAL = "cancelled_rebalance" # 新再平衡目标不含该票 → 撤未成交买单
R_DELIST_WRITEOFF = "delisted_writeoff"   # 长期无价格 → 按末价强制清仓
R_T1_BLOCKED = "t_plus_1_blocked"         # 当日买入不可当日卖出
R_FILTER_NO_RECENT = "no_recent_price"    # 决策日过滤：近期无 bar
R_FILTER_NOT_UNIVERSE = "not_in_universe" # 决策日过滤：不在 PIT top-N 选股池


# ============================================================================
# 价格仓库（宽表，一次构建）
# ============================================================================


class PriceStore:
    """v6 价格长表 → 宽表集合。所有查询严格按 trade_date 行取，无前视。"""

    def __init__(self, prices: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp):
        p = prices.copy()
        p["trade_date"] = pd.to_datetime(p["trade_date"]).dt.normalize()
        p = p[(p["trade_date"] >= start) & (p["trade_date"] <= end)]
        p = p.drop_duplicates(["trade_date", "ts_code"], keep="last")

        def wide(col: str) -> pd.DataFrame:
            return (p.pivot(index="trade_date", columns="ts_code", values=col)
                    .sort_index().astype("float32"))

        self.open_raw = wide("open")
        self.high_raw = wide("high")
        self.low_raw = wide("low")
        self.pre_close_raw = wide("pre_close")
        self.adj_factor = wide("adj_factor")
        self.adj_close = wide("close") * self.adj_factor
        self.open_adj = self.open_raw * self.adj_factor
        self.amount = wide("amount")
        self.adv20 = self.amount.rolling(20, min_periods=10).mean()
        # 前向填充只用过去数据 → PIT 安全；用于"≤d 末价"快速查询
        self.adj_close_ffill = self.adj_close.ffill()
        self.calendar: pd.DatetimeIndex = self.adj_close.index
        # 每 (d, ticker) 的"≤d 最近有 bar 的日期"（决策日停牌/退市过滤用，PIT 安全）
        dates_i8 = pd.DataFrame(
            np.where(self.open_raw.notna(),
                     self.calendar.asi8[:, None].repeat(self.open_raw.shape[1], axis=1),
                     np.nan),
            index=self.calendar, columns=self.open_raw.columns)
        self.last_bar_upto = dates_i8.ffill()
        # 每票最后一个有 bar 的日期（退市判定用；对"未来是否还有数据"这一模拟器
        # 全知信息只用于 write-off 判断，不用于选股/成交决策）
        self._last_bar = self.adj_close.apply(lambda s: s.last_valid_index())

    def has_bar(self, d: pd.Timestamp, ticker: str) -> bool:
        try:
            return bool(pd.notna(self.open_raw.at[d, ticker]))
        except KeyError:
            return False

    def last_bar_date(self, ticker: str) -> pd.Timestamp | None:
        v = self._last_bar.get(ticker)
        return None if v is None or pd.isna(v) else pd.Timestamp(v)

    def last_price_upto(self, d: pd.Timestamp, ticker: str) -> float | None:
        """≤d 的最后已知 adj_close（严格 PIT：ffill 只带过去值）。"""
        if ticker not in self.adj_close_ffill.columns:
            return None
        try:
            v = self.adj_close_ffill.at[d, ticker]
        except KeyError:
            s = self.adj_close_ffill[ticker].loc[:d]
            v = s.iloc[-1] if len(s) else np.nan
        return None if pd.isna(v) else float(v)

    def universe_topn(self, as_of: pd.Timestamp, n: int) -> set[str]:
        """PIT adv20 top-N（与 p4d._adv20_top1500_mask 同口径：只用 ≤as_of）。"""
        prior = self.adv20.index[self.adv20.index <= as_of]
        if len(prior) == 0:
            return set()
        row = self.adv20.loc[prior[-1]].dropna()
        return set(row.sort_values(ascending=False).head(n).index)

    def buy_blocked(self, d: pd.Timestamp, ticker: str) -> str | None:
        """买入是否受限。返回 reason code 或 None（可买）。"""
        if not self.has_bar(d, ticker):
            return R_SUSPENDED
        pre = float(self.pre_close_raw.at[d, ticker])
        hi = float(self.high_raw.at[d, ticker])
        lo = float(self.low_raw.at[d, ticker])
        limit_up = round(pre * (1 + price_limit_pct(ticker, d)), 2)
        if hi >= limit_up - 1e-9:
            return R_LIMIT_UP
        if hi == lo and hi > pre:          # 一字涨停兜底（ST ±5% 无标记时也能捕获）
            return R_ONE_LINE
        return None

    def sell_blocked(self, d: pd.Timestamp, ticker: str) -> str | None:
        """卖出是否受限。返回 reason code 或 None（可卖）。"""
        if not self.has_bar(d, ticker):
            return R_SUSPENDED
        pre = float(self.pre_close_raw.at[d, ticker])
        hi = float(self.high_raw.at[d, ticker])
        lo = float(self.low_raw.at[d, ticker])
        limit_dn = round(pre * (1 - price_limit_pct(ticker, d)), 2)
        if lo <= limit_dn + 1e-9:
            return R_LIMIT_DOWN
        if hi == lo and hi < pre:          # 一字跌停兜底
            return R_ONE_LINE
        return None


# ============================================================================
# 引擎
# ============================================================================


@dataclass
class _Position:
    shares: float          # 后复权口径股数（value = shares × adj_close）
    buy_date: pd.Timestamp # 最近一次买入成交日（T+1 校验）


class ExecutableNavEngine:
    """Top-N 组合状态机 + 真实 NAV 推进（设计 2.5–2.7）。"""

    def __init__(
        self,
        preds: pd.DataFrame,          # columns: as_of, ticker, score
        prices: pd.DataFrame,         # v6 长表：ts_code/trade_date/OHLC/pre_close/adj_factor/…
        config: NavConfig,
        amihud: pd.DataFrame | None = None,  # index=(as_of,ticker), col=amihud_illiquidity
    ) -> None:
        self.cfg = config
        p = preds.copy()
        p["as_of"] = pd.to_datetime(p["as_of"]).dt.normalize()
        self.preds = p.sort_values(["as_of", "ticker"]).reset_index(drop=True)
        self.as_of_list = sorted(self.preds["as_of"].unique())
        # 再平衡日历 = as_of[::step]（设计 2.5）
        self.rebalance_dates = list(pd.DatetimeIndex(self.as_of_list)[:: config.rebalance_step])

        start = pd.Timestamp(self.as_of_list[0]) - pd.Timedelta(days=60)  # adv20 回看缓冲
        end = pd.to_datetime(prices["trade_date"]).max()
        self.px = PriceStore(prices, start, end)
        self.amihud = amihud

        # 运行状态
        self.cash: float = config.initial_nav
        self.positions: dict[str, _Position] = {}
        self.pending: list[dict] = []      # 未成交订单
        self.trades: list[dict] = []
        self.rejected: list[dict] = []
        self.targets_log: list[dict] = []
        self.filtered_log: list[dict] = []
        self.nav_rows: list[dict] = []
        self.holdings_rows: list[dict] = []
        self.cum_cost: float = 0.0
        self.turnover_rows: list[dict] = []
        self._bench_members: set[str] = set()
        self._bench_nav: float = config.initial_nav
        self._missing_bar_days: int = 0

    # ── 滑点：决策截面 amihud 分位 → tier bp ─────────────────────────────────
    def _slippage_bp_map(self, as_of: pd.Timestamp, tickers: list[str]) -> dict[str, float]:
        tiers = self.cfg.tiers
        if self.amihud is None:
            return dict.fromkeys(tickers, tiers.small_bp)   # 无流动性信息 → 最保守档
        try:
            cs = self.amihud.xs(as_of, level="as_of")["amihud_illiquidity"]
        except KeyError:
            return dict.fromkeys(tickers, tiers.small_bp)
        q = amihud_to_quantile(cs.reindex(tickers))
        return {t: tiers.bp_for_quantile(q.get(t, np.nan)) for t in tickers}

    # ── 组合估值（≤d 末价，严格 PIT）───────────────────────────────────────
    def _portfolio_value(self, d: pd.Timestamp) -> float:
        total = 0.0
        for t, pos in self.positions.items():
            px = self.px.last_price_upto(d, t)
            if px is not None:
                total += pos.shares * px
        return total

    def _nav(self, d: pd.Timestamp) -> float:
        return self.cash + self._portfolio_value(d)

    # ── 决策日：生成目标与订单（as_of 收盘，只用 ≤as_of 信息）────────────────
    def _decide(self, as_of: pd.Timestamp) -> None:
        cfg = self.cfg
        universe = self.px.universe_topn(as_of, cfg.universe_size)
        cross = self.preds[self.preds["as_of"] == as_of]
        scores = cross.set_index("ticker")["score"]

        candidates, filtered = [], []
        recent_floor_idx = self.px.calendar.searchsorted(as_of, side="right") - cfg.min_recent_bar_td
        recent_floor_i8 = self.px.calendar[max(recent_floor_idx, 0)].value
        try:
            lb_row = self.px.last_bar_upto.loc[:as_of].iloc[-1]  # 严格 PIT：只看 ≤as_of
        except IndexError:
            lb_row = pd.Series(dtype=float)
        for t in scores.index:
            if t not in universe:
                filtered.append((t, R_FILTER_NOT_UNIVERSE))
                continue
            lb = lb_row.get(t, np.nan)
            if pd.isna(lb) or lb < recent_floor_i8:
                filtered.append((t, R_FILTER_NO_RECENT))
                continue
            candidates.append(t)

        target = (scores.loc[candidates].sort_values(ascending=False, kind="mergesort")
                  .head(cfg.top_n))
        target_names = list(target.index)
        nav_now = self._nav(as_of)
        tgt_value = nav_now / max(len(target_names), 1)

        # 撤掉与新目标冲突的未成交单：目标外的买单、重回目标的卖单
        still = []
        for o in self.pending:
            conflict = ((o["side"] == "buy" and o["ticker"] not in target_names)
                        or (o["side"] == "sell" and o["ticker"] in target_names))
            if conflict:
                self.rejected.append({**self._order_row(o), "fill_date": None,
                                      "reason_code": R_CANCELLED_REBAL, "reject_date": str(as_of.date())})
            else:
                still.append(o)
        self.pending = still

        slip = self._slippage_bp_map(as_of, sorted(set(target_names) | set(self.positions)))

        expiry = None
        later = [r for r in self.rebalance_dates if r > as_of]
        if later:
            expiry = later[0]

        # 仍在目标中的未成交买单：刷新 expiry 与目标金额（否则次日即过期，永不重试）
        for o in self.pending:
            if o["side"] == "buy" and o["ticker"] in target_names:
                o["expiry"] = expiry
                o["target_value"] = tgt_value
                o["decision_asof"] = as_of
                o["rank"] = target_names.index(o["ticker"])

        # 目标内：调平到等权（部分买/卖）；目标外持仓：全清
        pend_keys = {(o["side"], o["ticker"]) for o in self.pending}
        for t in sorted(set(target_names) | set(self.positions)):
            cur_px = self.px.last_price_upto(as_of, t)
            cur_val = (self.positions[t].shares * cur_px
                       if t in self.positions and cur_px is not None else 0.0)
            want = tgt_value if t in target_names else 0.0
            delta = want - cur_val
            if abs(delta) < nav_now * 1e-6:
                continue
            side = "buy" if delta > 0 else "sell"
            if (side, t) in pend_keys:
                continue
            order = {
                "side": side, "ticker": t, "decision_asof": as_of,
                "target_value": abs(delta) if side == "buy" else None,
                "sell_value": abs(delta) if side == "sell" else None,
                "sell_all": (side == "sell" and want == 0.0),
                "slippage_bp": float(slip.get(t, self.cfg.tiers.small_bp)),
                "expiry": expiry if side == "buy" else None,   # 卖单不会过期（强制持有直到能卖）
                "rank": (target_names.index(t) if t in target_names else -1),
            }
            self.pending.append(order)

        self.targets_log.append({
            "as_of": str(as_of.date()), "n_scored": int(len(scores)),
            "n_universe": int(len(universe)), "n_candidates": int(len(candidates)),
            "n_target": int(len(target_names)), "nav_at_decision": float(nav_now),
            "target_weight_each": float(1.0 / max(len(target_names), 1)),
            "targets": target_names,
        })
        for t, reason in filtered:
            self.filtered_log.append({"as_of": str(as_of.date()), "ticker": t, "reason_code": reason})

        # 基准成员更新（同一日历、同一选股池、无成本）
        self._bench_members = universe

    def _order_row(self, o: dict) -> dict:
        return {"side": o["side"], "ticker": o["ticker"],
                "decision_asof": str(pd.Timestamp(o["decision_asof"]).date()),
                "target_value": o.get("target_value"), "sell_value": o.get("sell_value"),
                "slippage_bp": o.get("slippage_bp")}

    # ── 成交日：先卖后买（同日先释放现金）────────────────────────────────────
    def _execute(self, d: pd.Timestamp) -> None:
        cfg = self.cfg
        sells = [o for o in self.pending if o["side"] == "sell"]
        buys = [o for o in self.pending if o["side"] == "buy"]
        sells.sort(key=lambda o: o["ticker"])
        buys.sort(key=lambda o: (o["rank"], o["ticker"]))
        remaining: list[dict] = []

        for o in sells:
            t = o["ticker"]
            pos = self.positions.get(t)
            if pos is None:
                continue  # 已被 write-off 等移除
            if pos.buy_date == d:
                remaining.append(o)      # T+1：当日买不可当日卖
                self.rejected.append({**self._order_row(o), "fill_date": None,
                                      "reason_code": R_T1_BLOCKED, "reject_date": str(d.date())})
                continue
            blocked = self.px.sell_blocked(d, t)
            if blocked:
                remaining.append(o)
                self.rejected.append({**self._order_row(o), "fill_date": None,
                                      "reason_code": blocked, "reject_date": str(d.date())})
                continue
            price = float(self.px.open_adj.at[d, t])
            if o.get("sell_all"):
                shares = pos.shares
            else:
                shares = min(pos.shares, float(o["sell_value"]) / price)
            gross = shares * price
            rate = cfg.sell_cost_rate(o["slippage_bp"], d)
            cost = gross * rate
            self.cash += gross - cost
            self.cum_cost += cost
            pos.shares -= shares
            if pos.shares <= 1e-12:
                del self.positions[t]
            self.trades.append({**self._order_row(o), "fill_date": str(d.date()),
                                "fill_price_adj": price, "shares": float(shares),
                                "gross_value": float(gross), "cost": float(cost),
                                "cost_rate": float(rate), "status": "filled"})

        for o in buys:
            t = o["ticker"]
            if o["expiry"] is not None and d >= o["expiry"]:
                self.rejected.append({**self._order_row(o), "fill_date": None,
                                      "reason_code": R_EXPIRED, "reject_date": str(d.date())})
                continue
            blocked = self.px.buy_blocked(d, t)
            if blocked:
                remaining.append(o)
                self.rejected.append({**self._order_row(o), "fill_date": None,
                                      "reason_code": blocked, "reject_date": str(d.date())})
                continue
            if self.cash <= 1e-12:
                remaining.append(o)
                self.rejected.append({**self._order_row(o), "fill_date": None,
                                      "reason_code": R_INSUFFICIENT_CASH, "reject_date": str(d.date())})
                continue
            price = float(self.px.open_adj.at[d, t])
            rate = cfg.buy_cost_rate(o["slippage_bp"])
            spend = min(float(o["target_value"]), self.cash)   # 现金守恒：绝不透支
            shares = spend / (price * (1.0 + rate))
            gross = shares * price
            cost = gross * rate
            self.cash -= gross + cost
            self.cum_cost += cost
            pos = self.positions.get(t)
            if pos:
                pos.shares += shares
                pos.buy_date = d
            else:
                self.positions[t] = _Position(shares=shares, buy_date=d)
            self.trades.append({**self._order_row(o), "fill_date": str(d.date()),
                                "fill_price_adj": price, "shares": float(shares),
                                "gross_value": float(gross), "cost": float(cost),
                                "cost_rate": float(rate), "status": "filled"})

        self.pending = remaining

    # ── 退市/长期无数据强制清仓 ──────────────────────────────────────────────
    def _writeoff_check(self, d: pd.Timestamp) -> None:
        cfg = self.cfg
        idx = self.px.calendar.searchsorted(d, side="right")
        for t in sorted(self.positions):
            last = self.px.last_bar_date(t)
            if last is None or last >= d:
                continue
            gap = idx - 1 - self.px.calendar.searchsorted(last, side="right") + 1
            if gap < cfg.delist_writeoff_td:
                continue
            # 数据集中无未来 bar → 视为退市：按末价强制清仓（计卖出成本，最保守小盘档滑点）
            pos = self.positions[t]
            px = self.px.last_price_upto(d, t)
            if px is None:
                px = 0.0
            rate = cfg.sell_cost_rate(cfg.tiers.small_bp, d)
            gross = pos.shares * px
            cost = gross * rate
            self.cash += gross - cost
            self.cum_cost += cost
            del self.positions[t]
            self.pending = [o for o in self.pending if not (o["side"] == "sell" and o["ticker"] == t)]
            self.trades.append({"side": "sell", "ticker": t,
                                "decision_asof": None, "target_value": None,
                                "sell_value": float(gross), "slippage_bp": cfg.tiers.small_bp,
                                "fill_date": str(d.date()), "fill_price_adj": float(px),
                                "shares": float(pos.shares), "gross_value": float(gross),
                                "cost": float(cost), "cost_rate": float(rate),
                                "status": R_DELIST_WRITEOFF})

    # ── 基准（同池等权，无成本）─────────────────────────────────────────────
    def _bench_step(self, d: pd.Timestamp, prev_d: pd.Timestamp | None) -> float:
        if prev_d is None or not self._bench_members:
            return self._bench_nav
        # sorted：set 迭代顺序随 PYTHONHASHSEED 变化，浮点求和顺序必须固定（跨进程确定性）
        cols = sorted(t for t in self._bench_members if t in self.adj_ret.columns)
        if not cols:
            return self._bench_nav
        r = self.adj_ret.loc[d, cols].dropna()
        if len(r) == 0:
            return self._bench_nav
        self._bench_nav *= float(1.0 + r.mean())
        return self._bench_nav

    # ── 主循环 ───────────────────────────────────────────────────────────────
    def run(self) -> dict[str, Any]:
        self.adj_ret = self.px.adj_close.pct_change(fill_method=None)
        rebal_set = set(pd.DatetimeIndex(self.rebalance_dates))
        start_d = self.rebalance_dates[0]
        cal = self.px.calendar[self.px.calendar >= start_d]
        prev_d = None
        prev_nav = self.cfg.initial_nav

        for d in cal:
            # 1) 成交（处理昨日决策与所有 pending；首日 d==start 无 pending）
            if self.pending:
                self._execute(d)
            # 2) 退市 write-off
            if self.positions:
                self._writeoff_check(d)
            # 3) 决策（as_of 收盘）
            if d in rebal_set:
                self._decide(d)
            # 4) 收盘 mtm
            nav = self._nav(d)
            bench = self._bench_step(d, prev_d)
            self.nav_rows.append({
                "date": d, "nav_net": float(nav),
                "nav_gross": float(nav + self.cum_cost),
                "nav_bench": float(bench),
                "cash": float(self.cash), "n_positions": int(len(self.positions)),
                "cum_cost": float(self.cum_cost),
                "daily_ret_net": float(nav / prev_nav - 1.0) if prev_d is not None else 0.0,
            })
            for t, pos in sorted(self.positions.items()):
                px = self.px.last_price_upto(d, t)
                self.holdings_rows.append({
                    "date": d, "ticker": t, "shares": float(pos.shares),
                    "price_adj": float(px) if px is not None else np.nan,
                    "value": float(pos.shares * px) if px is not None else np.nan,
                })
            prev_d, prev_nav = d, nav

        return self._finalize()

    # ── 汇总 ────────────────────────────────────────────────────────────────
    def _finalize(self) -> dict[str, Any]:
        nav = pd.DataFrame(self.nav_rows).set_index("date")
        nav["excess_net"] = nav["nav_net"] / nav["nav_bench"] - 1.0
        nav["cum_return_net"] = nav["nav_net"] / self.cfg.initial_nav - 1.0
        holdings = pd.DataFrame(self.holdings_rows)
        if not holdings.empty:
            nav_by_date = nav["nav_net"]
            holdings["weight"] = holdings["value"] / holdings["date"].map(nav_by_date)
        trades = pd.DataFrame(self.trades)
        rejected = pd.DataFrame(self.rejected)

        n_days = len(nav)
        years = n_days / 252.0

        def ann(total_growth: float) -> float:
            return float(total_growth ** (1.0 / years) - 1.0) if years > 0 else 0.0

        ann_net = ann(nav["nav_net"].iloc[-1] / self.cfg.initial_nav)
        ann_gross = ann(nav["nav_gross"].iloc[-1] / self.cfg.initial_nav)
        ann_bench = ann(nav["nav_bench"].iloc[-1] / self.cfg.initial_nav)

        ret_net = nav["nav_net"].pct_change().dropna()
        ret_bench = nav["nav_bench"].pct_change().dropna()
        ex = (ret_net - ret_bench).dropna()
        ir = float(ex.mean() / ex.std() * np.sqrt(252)) if ex.std() > 0 else 0.0

        roll_max = nav["nav_net"].cummax()
        maxdd = float(((nav["nav_net"] - roll_max) / roll_max).min())

        # 真实换手：每再平衡窗口 (买+卖成交额)/2 / 决策日 NAV
        turnover = []
        if not trades.empty:
            tr = trades[trades["decision_asof"].notna()].copy()
            for a, sub in tr.groupby("decision_asof"):
                navd = next((x["nav_at_decision"] for x in self.targets_log if x["as_of"] == a), None)
                if navd:
                    oneway = float(sub["gross_value"].sum()) / 2.0 / navd
                    turnover.append({"as_of": a, "oneway_turnover": oneway})
        to_df = pd.DataFrame(turnover)
        avg_oneway = float(to_df["oneway_turnover"].mean()) if not to_df.empty else 0.0
        ann_oneway = float(to_df["oneway_turnover"].sum() / years) if years > 0 and not to_df.empty else 0.0

        # 分年净超额（gate §5：每年>0）
        yearly = {}
        for y, sub in nav.groupby(nav.index.year):
            e0 = sub["nav_net"].iloc[0] / sub["nav_bench"].iloc[0]
            e1 = sub["nav_net"].iloc[-1] / sub["nav_bench"].iloc[-1]
            yearly[str(int(y))] = float(e1 / e0 - 1.0)

        n_rej = len(rejected)
        rej_counts = (rejected["reason_code"].value_counts().to_dict()
                      if not rejected.empty else {})
        summary = {
            "config": {
                "horizon_td": self.cfg.horizon_td, "rebalance_step": self.cfg.rebalance_step,
                "top_n": self.cfg.top_n, "universe_size": self.cfg.universe_size,
                "commission_bp": self.cfg.commission_bp, "transfer_bp": self.cfg.transfer_bp,
                "slippage_tiers_bp": [self.cfg.tiers.large_bp, self.cfg.tiers.mid_bp,
                                      self.cfg.tiers.small_bp],
                "delist_writeoff_td": self.cfg.delist_writeoff_td,
            },
            "period": {"start": str(nav.index[0].date()), "end": str(nav.index[-1].date()),
                       "n_trading_days": int(n_days), "n_rebalances": len(self.rebalance_dates)},
            "nav": {"final_net": float(nav["nav_net"].iloc[-1]),
                    "final_gross": float(nav["nav_gross"].iloc[-1]),
                    "final_bench": float(nav["nav_bench"].iloc[-1]),
                    "ann_return_net": ann_net, "ann_return_gross": ann_gross,
                    "ann_return_bench": ann_bench,
                    "ann_net_excess": float(ann_net - ann_bench),
                    "information_ratio": ir, "max_drawdown_net": maxdd,
                    "cum_return_net": float(nav["cum_return_net"].iloc[-1]),
                    "yearly_net_excess": yearly},
            "costs": {"total_cost_nav_units": float(self.cum_cost),
                      "cost_drag_ann": float(ann_gross - ann_net)},
            "turnover": {"avg_oneway_per_rebalance": avg_oneway,
                         "annualized_oneway": ann_oneway,
                         "annualized_twoway": ann_oneway * 2.0},
            "execution": {"n_trades_filled": int(len(trades)),
                          "n_rejection_events": int(n_rej),
                          "rejection_reason_counts": rej_counts,
                          "n_writeoffs": int((trades["status"] == R_DELIST_WRITEOFF).sum())
                          if not trades.empty else 0},
            "coverage": {"n_rebalance_targets_logged": len(self.targets_log),
                         "n_filtered_events": len(self.filtered_log),
                         "filtered_reason_counts":
                             pd.DataFrame(self.filtered_log)["reason_code"].value_counts().to_dict()
                             if self.filtered_log else {}},
        }
        return {"nav_daily": nav.reset_index(), "holdings_daily": holdings,
                "trades": trades, "rejected_trades": rejected,
                "targets": pd.DataFrame(self.targets_log),
                "filtered": pd.DataFrame(self.filtered_log),
                "turnover": to_df, "summary": summary}


__all__ = ["NavConfig", "ExecutableNavEngine", "PriceStore",
           "R_SUSPENDED", "R_LIMIT_UP", "R_LIMIT_DOWN", "R_ONE_LINE",
           "R_INSUFFICIENT_CASH", "R_EXPIRED", "R_CANCELLED_REBAL",
           "R_DELIST_WRITEOFF", "R_T1_BLOCKED",
           "R_FILTER_NO_RECENT", "R_FILTER_NOT_UNIVERSE"]
