"""T-019：事件 / 执行约束 / 行业基准类 Tushare 数据下载（只写新文件，可续跑）。

产物（都在 <data_root>/data/lake/events/ 下，不碰已有 lake 表）：
  _parts/<api>/<chunk>.parquet     每个分块一个文件（续跑以检查点为准）
  _checkpoints/<api>.json          每个分块的终态：done / empty / failed / truncated
  <api>_<END>.parquet              --compact 合并后的单表（拒绝覆盖已有文件）
  <api>_<END>.manifest.json        行数、日期范围、各终态分块数、sha256

分块方式（由接口的参数限制决定，见 T-019 卡片的探测记录）：
  trade_day  每个交易日一次（trade_date=）
  cal_day    每个自然日一次（ann_date=；接口不支持区间）
  month      每个自然月一次（start_date / end_date）
  window     固定天数窗口（report_rc 单次上限 5000 行、每分钟 1 次）
  period     每个报告期一次（end_date=）
  index_month 每个指数 × 每个月（index_weight）
  sw_l1      每个申万一级行业 × 是否当前成分（index_member_all）

分页：单次返回行数是 100 的整数倍且 > 0 时（Tushare 的单次上限都是整百数），带 offset 继续取下一页，
直到返回不足一页；如果下一页与已取内容完全重复（接口不支持 offset），或已到页数上限而最后一页仍是满页，
该分块记为 truncated（日期区间分块会对半拆分重取；不能再拆的保留 truncated，续跑时重试）。
--audit-page-cap：列出检查点里页数达到上限却记为 done 的分块（修复前的代码会这样记）；加 --mark 时改记为 truncated，续跑会重取。
空表：接口正常返回 0 行记为 empty；MUST_HAVE 接口在交易日返回 0 行记为 failed（重跑时会重试）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

DONE, EMPTY, FAILED, TRUNCATED = "done", "empty", "failed", "truncated"
TERMINAL_OK = {DONE, EMPTY}

SW_L1 = None  # 运行时从 index_classify 取


@dataclass
class Spec:
    mode: str
    date_col: str
    must_have: bool = False          # 交易日不应为空
    min_interval: float = 0.0        # 该接口的最小调用间隔（秒）
    window_days: int = 0
    extra: dict = field(default_factory=dict)


SPECS: dict[str, Spec] = {
    # 执行约束（T+1 可执行 NAV）
    "stk_limit": Spec("trade_day", "trade_date", must_have=True),
    "suspend_d": Spec("month", "trade_date"),
    # 行业 / 基准 / 状态
    "index_member_all": Spec("sw_l1", "in_date"),
    "index_weight": Spec("index_month", "trade_date",
                         extra={"indexes": ["000300.SH", "000905.SH", "000852.SH", "932000.CSI"]}),
    "namechange": Spec("month", "ann_date"),
    # 财报事件
    "forecast": Spec("cal_day", "ann_date"),
    "express": Spec("month", "ann_date"),
    "disclosure_date": Spec("period", "end_date"),
    # 股东 / 资本运作
    "stk_holdernumber": Spec("month", "ann_date"),
    "stk_holdertrade": Spec("month", "ann_date"),
    "repurchase": Spec("month", "ann_date"),
    "share_float": Spec("cal_day", "ann_date"),
    # 交易行为
    "moneyflow": Spec("trade_day", "trade_date", must_have=True),
    "top_list": Spec("trade_day", "trade_date"),
    "top_inst": Spec("trade_day", "trade_date"),
    "block_trade": Spec("trade_day", "trade_date"),
    # 分析师
    # 实测限额（2000 积分）：每分钟 1 次、每天 10 次（报错"频率超限(10次/天)"）；单次上限 5,000 行
    # （2026-09 约 22 天到上限）→ 15 天窗口留出余量、间隔 65 秒；每天由定时任务带 --max-calls 10 慢慢补（T-031）
    "report_rc": Spec("window", "report_date", min_interval=65.0, window_days=15),
    # 治理与审计（T-035，2026-10-21 实测 2000 积分可访问）：
    # pledge_stat 按周统计日（周五）横截面查询；fina_audit、dividend 只能按股票查询（全历史，一只一个分块）
    "pledge_stat": Spec("week_fri", "end_date"),
    "fina_audit": Spec("ticker", "ann_date"),
    "dividend": Spec("ticker", "ann_date"),
}


class QuotaExhaustedError(Exception):
    """今日额度用完：接口报"次/天"，或本次运行的调用预算（--max-calls）用完。
    整个运行停止；正在取的分块不记终态、不写分块文件（半截数据丢弃），下次运行从它重新开始。"""


def budgeted(call, max_calls: int | None):
    """给 call 加调用次数上限：第 max_calls + 1 次调用前抛 QuotaExhaustedError（不发出这次请求）。"""
    if not max_calls:
        return call
    used = {"n": 0}

    def wrapped(api, **kw):
        if used["n"] >= max_calls:
            raise QuotaExhaustedError(f"call budget {max_calls} used")
        used["n"] += 1
        return call(api, **kw)
    return wrapped


def rolling_quota(call, log_path: Path, limit: int, window_s: float, max_wait_s: float,
                  clock=time.time, sleep=time.sleep, log=print):
    """滚动窗口限额：任意 window_s 秒内最多 limit 次调用（被拒的也算）。
    实测（2026-10-19）：report_rc 的"10次/天"按滚动 24 小时计，不是北京时间零点重置——
    10-18 12:10–12:22 调满 10 次后，10-19 12:10 的第一次调用即被拒。
    每次调用前读调用记录：已满就等最早一次过期（+5 秒），需要等的时间超过 max_wait_s 就抛 QuotaExhaustedError。"""
    log_path = Path(log_path)

    def _load() -> list[float]:
        try:
            return [float(t) for t in json.loads(log_path.read_text(encoding="utf-8"))]
        except FileNotFoundError:
            return []

    def wrapped(api, **kw):
        now = clock()
        recent = sorted(t for t in _load() if t > now - window_s)
        if len(recent) >= limit:
            wait = recent[len(recent) - limit] + window_s - now + 5
            if wait > max_wait_s:
                raise QuotaExhaustedError(f"rolling quota {limit}/{window_s:.0f}s full; next slot in {wait:.0f}s")
            log(f"[quota] {len(recent)} calls in window, waiting {wait:.0f}s for the oldest to expire")
            sleep(wait)
            now = clock()
            recent = [t for t in recent if t > now - window_s]
        recent.append(now)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = log_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(recent), encoding="utf-8")
        os.replace(tmp, log_path)
        return call(api, **kw)
    return wrapped


# ---------------------------------------------------------------- 分块
def ymd(d) -> str:
    return pd.Timestamp(d).strftime("%Y%m%d")


def chunks_for(api: str, spec: Spec, start: str, end: str, trade_days: pd.DatetimeIndex,
               sw_l1: list[str] | None = None, tickers: list[str] | None = None) -> list[tuple[str, dict]]:
    """返回 [(chunk_key, params)]，chunk_key 可作文件名。"""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    out: list[tuple[str, dict]] = []
    if spec.mode == "trade_day":
        for d in trade_days[(trade_days >= s) & (trade_days <= e)]:
            out.append((ymd(d), {"trade_date": ymd(d)}))
    elif spec.mode == "cal_day":
        for d in pd.date_range(s, e, freq="D"):
            out.append((ymd(d), {"ann_date": ymd(d)}))
    elif spec.mode == "month":
        for m in pd.period_range(s, e, freq="M"):
            a, b = max(m.start_time.normalize(), s), min(m.end_time.normalize(), e)
            out.append((m.strftime("%Y%m"), {"start_date": ymd(a), "end_date": ymd(b)}))
    elif spec.mode == "window":
        a = s
        while a <= e:
            b = min(a + pd.Timedelta(days=spec.window_days - 1), e)
            out.append((f"{ymd(a)}_{ymd(b)}", {"start_date": ymd(a), "end_date": ymd(b)}))
            a = b + pd.Timedelta(days=1)
    elif spec.mode == "period":
        for q in pd.period_range(s, e, freq="Q"):
            out.append((ymd(q.end_time), {"end_date": ymd(q.end_time)}))
    elif spec.mode == "index_month":
        for idx in spec.extra["indexes"]:
            for m in pd.period_range(s, e, freq="M"):
                a, b = max(m.start_time.normalize(), s), min(m.end_time.normalize(), e)
                out.append((f"{idx}_{m.strftime('%Y%m')}",
                            {"index_code": idx, "start_date": ymd(a), "end_date": ymd(b)}))
    elif spec.mode == "sw_l1":
        for code in sw_l1 or []:
            for is_new in ("Y", "N"):
                out.append((f"{code}_{is_new}", {"l1_code": code, "is_new": is_new}))
    elif spec.mode == "week_fri":
        for d in pd.date_range(s, e, freq="W-FRI"):
            out.append((ymd(d), {"end_date": ymd(d)}))
    elif spec.mode == "ticker":
        for t in tickers or []:
            out.append((t, {"ts_code": t}))
    else:
        raise ValueError(f"unknown mode {spec.mode}")
    return out


# ---------------------------------------------------------------- 调用与分页
def _row_keys(df: pd.DataFrame) -> set:
    return set(map(tuple, df.astype(str).itertuples(index=False, name=None)))


MAX_PAGES = 200


def fetch_chunk(call, api: str, params: dict, spec: Spec, sleep=time.sleep,
                max_pages: int = MAX_PAGES) -> tuple[str, pd.DataFrame, int]:
    """取一个分块（含分页）。返回 (state, df, pages)。call(api, **params) 失败时抛异常。"""
    pages, frames, seen = 0, [], set()
    offset = 0
    while True:
        kw = dict(params)
        if offset:
            kw["offset"] = offset
        df = call(api, **kw)
        pages += 1
        if spec.min_interval:
            sleep(spec.min_interval)
        n = 0 if df is None else len(df)
        if n == 0:
            break
        keys = _row_keys(df)
        if offset and keys <= seen:          # 下一页与已取内容完全重复：接口不支持 offset
            out = pd.concat(frames, ignore_index=True)
            return TRUNCATED, out, pages
        seen |= keys
        frames.append(df)
        if n % 100 != 0:                     # 不是整百数：不是被截断的满页
            break
        if pages >= max_pages:               # 到页数上限且最后一页仍是满页：可能还有数据，不能记为完成
            return TRUNCATED, pd.concat(frames, ignore_index=True), pages
        offset += n
    if not frames:
        return EMPTY, pd.DataFrame(), pages
    out = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    return DONE, out, pages


def fetch_adaptive(call, api: str, params: dict, spec: Spec, sleep=time.sleep,
                   max_pages: int = MAX_PAGES) -> tuple[str, pd.DataFrame, int]:
    """fetch_chunk；若被截断且分块是日期区间，则对半拆分区间递归重取（直到单日）。"""
    state, df, pages = fetch_chunk(call, api, params, spec, sleep=sleep, max_pages=max_pages)
    if state != TRUNCATED or "start_date" not in params or params["start_date"] == params["end_date"]:
        return state, df, pages
    s, e = pd.Timestamp(params["start_date"]), pd.Timestamp(params["end_date"])
    mid = s + (e - s) // 2
    s1, d1, p1 = fetch_adaptive(call, api, dict(params, end_date=ymd(mid)), spec, sleep, max_pages)
    s2, d2, p2 = fetch_adaptive(call, api, dict(params, start_date=ymd(mid + pd.Timedelta(days=1))), spec, sleep,
                                max_pages)
    frames = [d for d in (d1, d2) if len(d)]
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    state = TRUNCATED if TRUNCATED in (s1, s2) else (DONE if len(out) else EMPTY)
    return state, out, pages + p1 + p2


# ---------------------------------------------------------------- 检查点
def load_ckpt(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"version": 1, "chunks": {}}


def save_ckpt(path: Path, ck: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(ck, ensure_ascii=False, indent=0), encoding="utf-8")
    os.replace(tmp, path)


def run_api(api: str, spec: Spec, chunks: list[tuple[str, dict]], events_dir: Path, call,
            is_trade_day=lambda k: True, sleep=time.sleep, retries: int = 3,
            log=print, max_pages: int = MAX_PAGES, force_keys: set[str] | frozenset = frozenset()) -> dict:
    parts = events_dir / "_parts" / api
    parts.mkdir(parents=True, exist_ok=True)
    ck_path = events_dir / "_checkpoints" / f"{api}.json"
    ck = load_ckpt(ck_path)
    # 分块已是终态但查询参数变了（例如当月分块上次只取到月中，这次 --end 延后），也要重取：
    # 否则"当月 / 当季 / 最后一个窗口"这类被 --end 截断的分块，在延长截止日后永远停在旧的截止日（T-035 发现）。
    # force_keys：迟到与修订队列（--refresh-recent），不论状态都重取
    todo = [(k, p) for k, p in chunks
            if k in force_keys or ck["chunks"].get(k, {}).get("state") not in TERMINAL_OK
            or ck["chunks"].get(k, {}).get("params", p) != p]
    log(f"[{api}] chunks={len(chunks)} todo={len(todo)}")
    t0 = time.time()
    for i, (key, params) in enumerate(todo, 1):
        rec = {"params": params}
        for attempt in range(retries):
            try:
                state, df, pages = fetch_adaptive(call, api, params, spec, sleep=sleep, max_pages=max_pages)
                if state == EMPTY and spec.must_have and is_trade_day(key):
                    state = FAILED
                    rec["error"] = "empty on a trading day"
                rec.update(state=state, rows=int(len(df)), pages=pages)
                if len(df):
                    df.to_parquet(parts / f"{key}.parquet", index=False)
                if state != FAILED:
                    rec.pop("error", None)
                    break
            except QuotaExhaustedError:
                save_ckpt(ck_path, ck)       # 当前分块保持原状态（不记 failed），下次运行重取
                log(f"[{api}] stop for today at {key} ({i - 1}/{len(todo)} chunks finished this run)")
                raise
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                if "次/天" in msg:            # 每日额度：再试只会继续被拒（被拒的调用也可能计入额度）
                    save_ckpt(ck_path, ck)
                    log(f"[{api}] daily quota hit at {key} ({i - 1}/{len(todo)} chunks finished this run)")
                    raise QuotaExhaustedError(msg[:160]) from e
                rec.update(state=FAILED, error=f"{type(e).__name__}: {msg[:160]}")
                if "次/小时" in msg:
                    sleep(3700)          # 失败的调用也可能计入额度：等满一小时再试
                elif "频率超限" in msg or "每分钟" in msg:
                    sleep(65)
                else:
                    sleep(5 * (attempt + 1))
        ck["chunks"][key] = rec
        save_ckpt(ck_path, ck)
        if i % 50 == 0 or i == len(todo):
            log(f"[{api}] {i}/{len(todo)} last={key} state={rec['state']} rows={rec.get('rows')} "
                f"({time.time() - t0:.0f}s)")
    states = pd.Series([c["state"] for c in ck["chunks"].values()]).value_counts().to_dict()
    log(f"[{api}] states={states}")
    return states


def audit_page_cap(events_dir: Path, apis: list[str], max_pages: int = MAX_PAGES, mark: bool = False) -> dict:
    """检查点里 pages ≥ max_pages 却记为 done 的分块（修复前的 fetch_chunk 会这样记，可能漏行）。

    pages 是该分块所有调用（含对半拆分后的子区间）的总页数，任何一次子调用到上限都会使总数 ≥ 上限，所以这个筛选只会多列、不会漏列。
    mark=True 时把这些分块改记为 truncated（保留原记录在 page_cap_audit 字段），续跑会重取；不删除已有分块文件（重取成功会覆盖）。
    """
    out = {}
    for api in apis:
        ck_path = events_dir / "_checkpoints" / f"{api}.json"
        if not ck_path.exists():
            continue
        ck = load_ckpt(ck_path)
        hit = {k: {"rows": v.get("rows"), "pages": v.get("pages")} for k, v in ck["chunks"].items()
               if v.get("state") == DONE and (v.get("pages") or 0) >= max_pages}
        out[api] = hit
        if mark and hit:
            for k in hit:
                rec = ck["chunks"][k]
                rec["page_cap_audit"] = {"previous_state": rec["state"], "rows": rec.get("rows"), "pages": rec.get("pages")}
                rec["state"] = TRUNCATED
            save_ckpt(ck_path, ck)
    return out


# ---------------------------------------------------------------- 合并
def _coerce(df: pd.DataFrame, numeric_cols: set[str]) -> pd.DataFrame:
    for c in df.columns:
        if c in numeric_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
        else:
            df[c] = df[c].astype("string")
    return df


def compact(api: str, events_dir: Path, end: str, spec: Spec) -> dict:
    out = events_dir / f"{api}_{end.replace('-', '')}.parquet"
    if out.exists():
        raise FileExistsError(f"refuse to overwrite {out}")
    ck = load_ckpt(events_dir / "_checkpoints" / f"{api}.json")
    files = sorted((events_dir / "_parts" / api).glob("*.parquet"))
    frames, numeric = [], set()
    for f in files:
        d = pd.read_parquet(f)
        numeric |= {c for c in d.columns if pd.api.types.is_numeric_dtype(d[c]) and not pd.api.types.is_bool_dtype(d[c])}
        frames.append(d)
    if not frames:
        raise RuntimeError(f"{api}: no parts")
    df = pd.concat(frames, ignore_index=True)
    df = _coerce(df, numeric)
    n_before = len(df)
    df = df.drop_duplicates(ignore_index=True)
    df.to_parquet(out, index=False)
    states = pd.Series([c["state"] for c in ck["chunks"].values()]).value_counts().to_dict()
    dc = spec.date_col if spec.date_col in df.columns else None
    man = {"api": api, "file": out.name, "rows": int(len(df)), "exact_duplicate_rows_dropped": int(n_before - len(df)),
           "columns": {c: str(df[c].dtype) for c in df.columns},
           "date_col": dc, "date_min": str(df[dc].min()) if dc else None, "date_max": str(df[dc].max()) if dc else None,
           "chunk_states": states,
           "failed_or_truncated": {k: v for k, v in ck["chunks"].items() if v["state"] not in TERMINAL_OK},
           "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}
    (events_dir / f"{api}_{end.replace('-', '')}.manifest.json").write_text(
        json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    return man


# ---------------------------------------------------------------- CLI
def _tushare_call():
    from dotenv import load_dotenv
    load_dotenv(os.getenv("QM_ENV_FILE", "/home/lenovo/projects/quantmind/.env"), override=False)
    from quantmind.data import tushare_provider as tp
    try:
        from loguru import logger
        logger.remove()
    except Exception:  # noqa: BLE001
        pass
    pro = tp._get_tushare_pro()

    def call(api, **kw):
        tp._rate_limit()
        return pro.query(api, **kw)
    return call


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--apis", required=True, help="逗号分隔，或 all")
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default="2026-10-09")
    ap.add_argument("--compact", action="store_true", help="只合并已下载的分块，不调用接口")
    ap.add_argument("--plan", action="store_true", help="只打印分块数")
    ap.add_argument("--audit-page-cap", action="store_true", help="只列出页数到上限却记为 done 的分块，不调用接口")
    ap.add_argument("--mark", action="store_true", help="与 --audit-page-cap 同用：把这些分块改记为 truncated")
    ap.add_argument("--max-calls", type=int, default=0,
                    help="本次运行最多发出的接口调用数（含分页与拆分），到数即停、退出码 3；0 = 不限")
    ap.add_argument("--rolling-limit", type=int, default=0,
                    help="滚动窗口限额：任意 --rolling-hours 小时内最多这么多次调用（被拒的也算）；0 = 不启用")
    ap.add_argument("--rolling-hours", type=float, default=24.0)
    ap.add_argument("--rolling-max-wait-min", type=float, default=40.0,
                    help="窗口已满时最多等多少分钟让最早的调用过期，超过就停（退出码 3）")
    ap.add_argument("--refresh-recent", type=int, default=0,
                    help="迟到与修订队列：最近 N 个分块不论状态都重取（例如 disclosure_date 的最近两个报告期）；0 = 不启用")
    a = ap.parse_args(argv)
    root = Path(a.data_root)
    events = root / "data/lake/events"
    events.mkdir(parents=True, exist_ok=True)
    apis = list(SPECS) if a.apis == "all" else a.apis.split(",")
    if a.audit_page_cap:
        res = audit_page_cap(events, apis, mark=a.mark)
        print(json.dumps({"max_pages": MAX_PAGES, "marked": a.mark, "n_hit": {k: len(v) for k, v in res.items()},
                          "hits": res}, ensure_ascii=False))
        return 0
    cal = pd.read_parquet(root / "data/lake/trade_cal_sse.parquet")["cal_date"]
    trade_days = pd.DatetimeIndex(pd.to_datetime(cal.astype(str))).sort_values()
    trade_set = {ymd(d) for d in trade_days}
    call = None
    if not (a.compact or a.plan):
        call = _tushare_call()
        if a.rolling_limit:
            call = rolling_quota(call, events / "_checkpoints" / f"calls_{a.apis.replace(',', '_')}.json",
                                 a.rolling_limit, a.rolling_hours * 3600, a.rolling_max_wait_min * 60,
                                 log=lambda m: print(m, flush=True))
        call = budgeted(call, a.max_calls)
    try:
        return _run_all(a, apis, events, trade_days, trade_set, call)
    except QuotaExhaustedError as e:
        print(json.dumps({"stopped_for_today": str(e)}, ensure_ascii=False), flush=True)
        return 3


def _universe_tickers(lake: Path) -> list[str]:
    """按股票查询的接口所用的股票集合：最新的 stock_basic_full_<日期>.parquet（含在市、退市、暂停上市）。"""
    files = sorted(lake.glob("stock_basic_full_2*.parquet"))
    if not files:
        raise FileNotFoundError(f"no stock_basic_full_<date>.parquet in {lake}")
    return sorted(pd.read_parquet(files[-1], columns=["ts_code"])["ts_code"].astype(str).unique())


def _run_all(a, apis, events, trade_days, trade_set, call) -> int:
    sw = None
    tickers = None
    rc = 0
    for api in apis:
        spec = SPECS[api]
        if spec.mode == "ticker" and tickers is None and not a.compact:
            tickers = _universe_tickers(events.parent)
        if a.compact:
            man = compact(api, events, a.end, spec)
            print(json.dumps({k: man[k] for k in ("api", "rows", "date_min", "date_max", "chunk_states")}, ensure_ascii=False))
            continue
        if spec.mode == "sw_l1" and sw is None:
            if call is None:
                sw = ["<L1>"] * 31
            else:
                for attempt in range(5):
                    try:
                        cls = call("index_classify", level="L1", src="SW2021")
                        break
                    except QuotaExhaustedError:
                        raise
                    except Exception as e:  # noqa: BLE001
                        print(f"[index_classify] attempt {attempt + 1} failed: {type(e).__name__}", flush=True)
                        time.sleep(10 * (attempt + 1))
                else:
                    raise RuntimeError("index_classify failed 5 times")
                sw = sorted(cls["index_code"].astype(str))
        start = a.start if spec.mode != "trade_day" else max(a.start, "2019-01-01")
        chunks = chunks_for(api, spec, start, a.end, trade_days, sw, tickers)
        if a.plan:
            print(f"{api:18s} mode={spec.mode:11s} chunks={len(chunks)}")
            continue
        recent = getattr(a, "refresh_recent", 0) or 0
        force = frozenset(k for k, _ in chunks[-recent:]) if recent and spec.mode != "ticker" else frozenset()
        states = run_api(api, spec, chunks, events, call, is_trade_day=lambda k: k[:8] in trade_set, force_keys=force)
        if states.get(FAILED) or states.get(TRUNCATED):
            rc = 2
        sys.stdout.flush()
    return rc


if __name__ == "__main__":
    sys.exit(main())
