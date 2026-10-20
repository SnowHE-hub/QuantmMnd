"""quantmind.line_a.f1_instruments — F1 证券身份与交易状态（03_final_plan_v2 第 5 节 F1）.

三类规则分开（原则 P4）：
- A 法律与账户权限：账户没开通的板块不能新买入（configs/line_a_v2/account_profiles.yaml 的三条线）；退市整理期不新买。
- B 决策时已知的执行状态：当日停牌、收盘封涨停 / 跌停、一字板。只是标记——次日能否成交由次日状态和
  冻结的执行规则决定，这里不用任何次日信息。
- C 研究 / 投资政策：ST、上市不满 120 个交易日、低价等。只作为可登记的挑战臂选项，参考臂不使用。
持仓股始终保留在输出里（不受任何截断影响）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import yaml

from quantmind.backtest.wf_costs import board_of


def _ymd(d) -> str:
    return pd.Timestamp(d).strftime("%Y%m%d")


def _latest(lake: Path, pattern: str) -> Path:
    files = sorted(lake.glob(pattern))
    if not files:
        raise FileNotFoundError(f"{pattern} not found in {lake}")
    return files[-1]


def _rows_on(path: Path, date_col: str, ymd: str, columns: list[str]) -> pd.DataFrame:
    """只读某一天的行（数据集过滤，不把大表整个读进内存）。日期列可能是字符串或整数。"""
    dset = ds.dataset(str(path), format="parquet")
    typ = str(dset.schema.field(date_col).type)
    if typ.startswith("timestamp") or typ.startswith("date"):
        val = pd.Timestamp(ymd).to_datetime64()
    elif typ.startswith("int"):
        val = int(ymd)
    else:
        val = ymd
    return dset.to_table(filter=ds.field(date_col) == val, columns=columns).to_pandas()


def _dates(s: pd.Series) -> pd.Series:
    """YYYYMMDD 字符串或时间戳 → Timestamp；空值和非法值为 NaT。"""
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    return pd.to_datetime(s.astype("string").str.slice(0, 8), format="%Y%m%d", errors="coerce")


def load_profiles(path: Path) -> dict[str, dict[str, bool]]:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return {k: v["boards"] for k, v in cfg["profiles"].items()}


def instrument_state(data_root: Path, as_of, profiles: dict[str, dict[str, bool]], holdings: set[str] = frozenset(),
                     *, low_price: float = 2.0, min_listed_td: int = 120) -> pd.DataFrame:
    D = Path(data_root) / "data"
    lake = D / "lake"
    ymd = _ymd(as_of)
    d = pd.Timestamp(as_of)
    cal = pd.to_datetime(pd.read_parquet(lake / "trade_cal_sse.parquet")["cal_date"].astype(str)).sort_values()
    cal = pd.DatetimeIndex(cal[cal <= d])
    sb = pd.read_parquet(_latest(lake, "stock_basic_full_merged_2*.parquet"),
                         columns=["ts_code", "list_date", "delist_date", "name"])
    sb["list_date"] = _dates(sb["list_date"])
    sb["delist_date"] = _dates(sb["delist_date"])
    live = sb[(sb["list_date"] <= d) & (sb["delist_date"].isna() | (sb["delist_date"] > d))]
    tickers = sorted(set(live["ts_code"]) | set(holdings))
    out = pd.DataFrame(index=pd.Index(tickers, name="ticker"))
    out["board"] = [board_of(t) for t in out.index]
    lst = live.set_index("ts_code")["list_date"].reindex(out.index)
    out["listed_td"] = [int(np.searchsorted(cal, ld, side="left")) if pd.notna(ld) else -1 for ld in lst]
    out["listed_td"] = len(cal) - out["listed_td"]
    # 名称（按名称变更表的生效区间），用来识别 ST 与退市整理
    nc = pd.read_parquet(_latest(lake / "events", "namechange_2*.parquet"),
                         columns=["ts_code", "name", "start_date", "end_date"])
    nc["start_date"] = _dates(nc["start_date"])
    nc["end_date"] = _dates(nc["end_date"])
    cur = nc[(nc["start_date"] <= d) & (nc["end_date"].isna() | (nc["end_date"] >= d))]
    cur = cur.sort_values("start_date").groupby("ts_code")["name"].last()
    name = cur.reindex(out.index).fillna(live.set_index("ts_code")["name"].reindex(out.index)).astype("string")
    out["name"] = name
    out["is_st"] = name.str.contains("ST", na=False).astype(bool)
    out["delisting_consolidation"] = name.str.contains("退", na=False).astype(bool)
    # 当日状态
    sus = _rows_on(_latest(lake / "events", "suspend_d_2*.parquet"), "trade_date", ymd, ["ts_code", "suspend_type"])
    out["suspended"] = out.index.isin(sus.loc[sus["suspend_type"].astype(str) == "S", "ts_code"])
    bar = _rows_on(lake / "daily.parquet", "trade_date", ymd, ["ts_code", "close", "high", "low", "pre_close"])
    lim = _rows_on(_latest(lake / "events", "stk_limit_2*.parquet"), "trade_date", ymd, ["ts_code", "up_limit", "down_limit"])
    px = bar.set_index("ts_code").join(lim.set_index("ts_code"), how="left").reindex(out.index)
    out["close"] = px["close"]
    out["has_bar"] = px["close"].notna()
    out["limit_up_close"] = (px["close"] >= px["up_limit"] - 1e-9).fillna(False).astype(bool)
    out["limit_down_close"] = (px["close"] <= px["down_limit"] + 1e-9).fillna(False).astype(bool)
    flat = (px["high"] == px["low"]).fillna(False)
    out["one_line_up"] = (flat & out["limit_up_close"]).astype(bool)
    out["one_line_down"] = (flat & out["limit_down_close"]).astype(bool)
    # C 类：只标记
    out["c_st"] = out["is_st"]
    out["c_listed_lt_min"] = out["listed_td"] < min_listed_td
    out["c_low_price"] = (out["close"] < low_price).fillna(False).astype(bool)
    # A 类：每条账户权限线能否新买
    for prof, boards in profiles.items():
        out[f"new_buy_{prof}"] = out["board"].map(lambda b, bd=boards: bool(bd.get(b, False))) & ~out["delisting_consolidation"]
    out["is_holding"] = out.index.isin(list(holdings))
    out["sell_execution_risk"] = out["suspended"] | out["limit_down_close"] | ~out["has_bar"]
    out["buy_execution_risk"] = out["suspended"] | out["limit_up_close"] | ~out["has_bar"]
    out.attrs.update({"as_of": str(d.date()), "low_price": low_price, "min_listed_td": min_listed_td})
    return out
