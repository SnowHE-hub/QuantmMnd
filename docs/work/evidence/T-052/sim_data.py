"""T-052 模拟盘回测的输入：信号（历史样本外 + 10-09 之后冻结模型包的每日 F3）、价格、滑点分位、分红、ST 区间、指数.

信号与研究期口径相同：
- ENS20 / ENS63：≤ 2026-10-09 用 T-030 各折样本外预测做四族等权截面排名平均（rank_average_frames）；
  之后用每日运行的 f3_ens_20d / f3_ens_63d（冻结第 18 折 / 第 8 折模型包）；
- C5-20：ENS20 去掉 challengers.c5_flags 命中的股票（T-043）；
- Q63：由模拟器的 quota_ext 处理（池内 80% + 池外 adv20 ≥ 1,000 万元 20%）。
决策日：冻结的 v6_1 周频网格，10-09 之后按同一相位每 5 个交易日延伸（f8_books.extended_grid）。
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "docs/work/evidence/T-030"))

from eval_primary import ENS_MEMBERS, load_family  # noqa: E402

from quantmind.line_a import challengers  # noqa: E402
from quantmind.line_a.f8_books import FREEZE, extended_grid  # noqa: E402
from quantmind.research.gpu.ensemble import rank_average_frames  # noqa: E402
from quantmind.research.gpu.folds import calendar_and_grid  # noqa: E402

ROOT = Path("/home/lenovo/projects/quantmind")
LAKE = ROOT / "data/lake"
RUNS = ROOT / "data/line_a_v2/runs"
FWD = Path("/home/lenovo/qm_fwd_root/20261023/main")
PRICES = ROOT / "data/raw/alpha_prices_panel_v6_20261023.parquet"
INDEX = {"沪深300": "000300.SH", "中证500": "000905.SH", "中证1000": "000852.SH", "中证2000": "932000.CSI"}


def grid() -> list[pd.Timestamp]:
    _, frozen = calendar_and_grid(ROOT)
    full = pd.DatetimeIndex(pd.to_datetime(pd.read_parquet(LAKE / "trade_cal_sse.parquet")["cal_date"].astype(str)))
    cal = full[full <= pd.Timestamp("2026-10-23")]
    return [d for d in extended_grid(cal, frozen) if d <= pd.Timestamp("2026-10-23")]


def _forward(stage: str, dates: list[pd.Timestamp]) -> tuple[pd.DataFrame, list[str]]:
    out, missing = [], []
    for d in dates:
        fs = sorted(glob.glob(str(RUNS / str(d.date()) / f"{stage}_*.parquet")), key=lambda f: Path(f).stat().st_mtime)
        if not fs:
            missing.append(str(d.date()))
            continue
        df = pd.read_parquet(fs[-1])
        col = "ENS" if "ENS" in df.columns else "score"
        df = df.rename(columns={col: "score"})
        df["as_of"] = pd.Timestamp(d)
        out.append(df[["as_of", "ticker", "score"]])
    return (pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["as_of", "ticker", "score"])), missing


def ens_scores(h: str, start: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    g = [d for d in grid() if d >= start]
    fams = {m: load_family(ROOT, m, h) for m in ENS_MEMBERS}
    fams = {m: s[s.index.get_level_values("as_of") >= start] for m, s in fams.items()}
    hist = rank_average_frames(fams).reset_index()
    hist = hist[hist["as_of"].isin([d for d in g if d <= FREEZE])]
    fwd, missing = _forward(f"f3_ens_{h}", [d for d in g if d > FREEZE])
    df = pd.concat([hist[["as_of", "ticker", "score"]], fwd], ignore_index=True)
    info = {"grid": [str(d.date()) for d in g], "hist_dates": int(hist["as_of"].nunique()),
            "forward_dates": int(fwd["as_of"].nunique()), "forward_missing": missing,
            "dates_without_scores": [str(d.date()) for d in g if d not in set(df["as_of"])]}
    return df, info


def c5_scores(base: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    out = []
    for d, g in base.groupby("as_of"):
        fl = challengers.c5_flags(LAKE, d, prices)
        out.append(g[~g["ticker"].isin(fl)])
    return pd.concat(out, ignore_index=True)


def prices(start: pd.Timestamp) -> pd.DataFrame:
    import pyarrow.dataset as pds
    cols = ["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "vol", "amount", "adj_factor"]
    t = pds.dataset(str(PRICES)).to_table(columns=cols, filter=pds.field("trade_date") >= start - pd.Timedelta(days=150))
    return t.to_pandas()


def amihud(start: pd.Timestamp) -> pd.DataFrame:
    months = pd.period_range(start - pd.Timedelta(days=31), "2026-10", freq="M")
    return pd.concat([pd.read_parquet(FWD / f"data/gpu_v1/panel/base/{m}.parquet", columns=["amihud_illiquidity"])
                      for m in months.astype(str)])


def dividends() -> pd.DataFrame:
    f = sorted((LAKE / "events").glob("dividend_2*.parquet"))[-1]
    return pd.read_parquet(f, columns=["ts_code", "div_proc", "ex_date", "cash_div_tax", "stk_div"])


def st_intervals() -> pd.DataFrame:
    n = challengers._st_names(LAKE)
    n = n[n["name"].astype(str).str.contains("ST")]
    return n[["ts_code", "s", "e"]].reset_index(drop=True)


def indexes(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    parts = []
    for f in ("index_daily.parquet", "index_daily_extra.parquet"):          # 两个文件的日期列类型不同
        x = pd.read_parquet(LAKE / f, columns=["ts_code", "trade_date", "close"])
        x["trade_date"] = pd.to_datetime(x["trade_date"].astype(str).str.replace("-", "").str.slice(0, 8))
        parts.append(x)
    i = pd.concat(parts, ignore_index=True).drop_duplicates(["ts_code", "trade_date"], keep="last")
    i = i[i["ts_code"].isin(INDEX.values()) & (i["trade_date"] >= start) & (i["trade_date"] <= end)]
    w = i.pivot_table(index="trade_date", columns="ts_code", values="close").sort_index()
    return w.rename(columns={v: k for k, v in INDEX.items()})
