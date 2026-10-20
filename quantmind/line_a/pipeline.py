"""quantmind.line_a.pipeline — 主线 A v2 只读影子日任务的编排（03_final_plan_v2 第 5、7.2、13 节；T-036）.

每个阶段都经 JobStore 执行。任务键 = 阶段 + as_of + F0 快照内容哈希 + 代码版本 + 模型产物哈希：
同一键已成功就复用已有产物；输入、代码或模型变了就是新任务，旧产物保留、不覆盖（产物文件名带任务键前缀）。
阶段：F0 快照 → F1 证券状态 → F2 支持域 → F3 参考臂打分（Ridge 20/63、ENS 20；ENS 63 没有冻结的前瞻模型包，
输出"不可用"）→ F4 研究候选 → F9 模板报告。F8 账本在前瞻网格日上汇总运行（books 子命令）。
只读影子：不写模型注册表、不发正式日报、不形成任何交易指令；资金批准状态 NOT_APPROVED。
"""
from __future__ import annotations

import argparse
import io
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from quantmind.line_a import f8_books
from quantmind.line_a.atomic import atomic_write_json, sha256_file
from quantmind.line_a.f1_instruments import instrument_state, load_profiles
from quantmind.line_a.f2f4 import research_candidates, support_domains
from quantmind.line_a.jobs import SUCCEEDED, JobStore, task_key
from quantmind.line_a.snapshot import build as build_snapshot

REPO = Path(__file__).resolve().parents[2]
CFG = REPO / "configs/line_a_v2"
ARMS = {"ridge_native_20d": ("ridge", "20d"), "ens_native_20d": ("ens", "20d"),
        "ridge_native_63d": ("ridge", "63d"), "ens_native_63d": ("ens", "63d")}


@dataclass
class Ctx:
    as_of: str
    data_root: Path                    # 真实数据根（湖、冻结预测、refs）
    model_root: Path                   # 影子数据根 main（日频面板与 model/v1 的前瞻月份）
    out_root: Path                     # data/line_a_v2
    bundle20: Path | None = None       # ENS 20d 冻结模型包；None → 不跑 ENS
    code_rev: str = ""
    snapshot_hash: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def run_dir(self) -> Path:
        return self.out_root / "runs" / self.as_of


def code_rev() -> str:
    try:
        rev = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True,
                             check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no", "quantmind",
                                "configs"], capture_output=True, text=True, check=True).stdout.strip()
        return rev + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _pq(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf)
    return buf.getvalue()


def _stage(js: JobStore, ctx: Ctx, stage: str, ext: str, produce, **extra) -> Path:
    fields = {"as_of": ctx.as_of, "snapshot": ctx.snapshot_hash, "code": ctx.code_rev, **extra}
    key = task_key(stage=stage, **fields)
    out = ctx.run_dir / f"{stage}_{key[:12]}.{ext}"
    rec = js.run(stage, fields, out, produce)
    if rec is None:
        raise RuntimeError(f"{stage} is being run by another process")
    return Path(rec["output_path"])


# ------------------------------------------------------------------ 各阶段
def f0(ctx: Ctx) -> dict:
    manifest, report = build_snapshot(ctx.data_root, ctx.as_of)
    ch = manifest.content_hash()
    out = ctx.out_root / "snapshots" / ctx.as_of / f"snapshot_{ch[:16]}.json"
    if not out.exists():
        atomic_write_json(out, json.loads(manifest.model_dump_json()))
    ctx.snapshot_hash = ch
    return {"path": str(out), "content_hash": ch, **report}


def score_ridge(ctx: Ctx, horizon: str) -> pd.DataFrame:
    from quantmind.line_a.ridge_scoring import features_from_daily, load_ref, score
    ref = load_ref(ctx.data_root / f"data/line_a_v2/refs/ridge_{horizon}_lastfold.npz")
    X = features_from_daily(ctx.model_root, [ctx.as_of], ref.feats)
    return score(ref, X).reset_index()


def score_ens(ctx: Ctx) -> pd.DataFrame:
    from quantmind.line_a.scoring import score_ens20
    out, _ = score_ens20(ctx.model_root, [ctx.as_of], ctx.bundle20)
    return out.reset_index()


def latest_holdings(out_root: Path, as_of: str) -> set[str]:
    books = sorted(p for p in (out_root / "books").glob("*/holdings_latest.parquet") if p.parent.name <= as_of)
    return set(pd.read_parquet(books[-1])["ticker"]) if books else set()


def run_day(ctx: Ctx, js: JobStore) -> dict:
    ctx.code_rev = ctx.code_rev or code_rev()
    snap = f0(ctx)
    res: dict = {"as_of": ctx.as_of, "code": ctx.code_rev, "snapshot": snap, "outputs": {}, "unavailable": {}}
    if not snap["data_ready"]:
        js.skip(task_key(stage="day", as_of=ctx.as_of, snapshot=ctx.snapshot_hash, code=ctx.code_rev), "day",
                {"as_of": ctx.as_of, "snapshot": ctx.snapshot_hash}, "F0 data_ready = false: "
                + "; ".join(snap["critical_missing"]))
        res["skipped"] = snap["critical_missing"]
        return res
    o = res["outputs"]
    profiles = load_profiles(CFG / "account_profiles.yaml")
    holdings = latest_holdings(ctx.out_root, ctx.as_of)
    o["f1"] = _stage(js, ctx, "f1_instruments", "parquet",
                     lambda: _pq(instrument_state(ctx.data_root, ctx.as_of, profiles, holdings)))
    o["f2"] = _stage(js, ctx, "f2_domains", "parquet", lambda: _pq(support_domains(ctx.model_root, ctx.as_of)),
                     model_root=str(ctx.model_root))
    for h in ("20d", "63d"):
        ref = ctx.data_root / f"data/line_a_v2/refs/ridge_{h}_lastfold.npz"
        if ref.exists():
            o[f"ridge_{h}"] = _stage(js, ctx, f"f3_ridge_{h}", "parquet", lambda h=h: _pq(score_ridge(ctx, h)),
                                     model=sha256_file(ref), model_root=str(ctx.model_root))
        else:
            res["unavailable"][f"ridge_native_{h}"] = "参考臂存档不存在（X0 未通过或未运行）"
    if ctx.bundle20 is not None:
        o["ens_20d"] = _stage(js, ctx, "f3_ens_20d", "parquet", lambda: _pq(score_ens(ctx)),
                              model=sha256_file(ctx.bundle20 / "manifest.json"), model_root=str(ctx.model_root))
    else:
        res["unavailable"]["ens_native_20d"] = "本次未提供模型包"
    res["unavailable"]["ens_native_63d"] = "没有冻结的 63 日前瞻模型包（需另行登记）"
    o["f4"] = _stage(js, ctx, "f4_candidates", "parquet", lambda: _pq(candidates(ctx, o, holdings)),
                     inputs=sorted(str(v) for v in o.values()))
    o["f9"] = _stage(js, ctx, "f9_report", "md", lambda: report(ctx, res).encode("utf-8"),
                     inputs=sorted(str(v) for v in o.values()))
    res["outputs"] = {k: str(v) for k, v in o.items()}
    atomic_write_json(ctx.run_dir / f"day_{ctx.snapshot_hash[:12]}.json", res)
    return res


def _scores(o: dict, name: str, col: str = "score") -> pd.Series | None:
    if name not in o:
        return None
    df = pd.read_parquet(o[name])
    return df.set_index("ticker")[col]


def candidates(ctx: Ctx, o: dict, holdings: set[str]) -> pd.DataFrame:
    dom = pd.read_parquet(o["f2"])
    parts = []
    for h, cols in (("20d", {"ens20": ("ens_20d", "ENS"), "ridge20": ("ridge_20d", "score")}),
                    ("63d", {"ridge63": ("ridge_63d", "score")})):
        sc = {k: s for k, (n, c) in cols.items() if (s := _scores(o, n, c)) is not None}
        if not sc:
            continue
        c = research_candidates(pd.DataFrame(sc), dom, ctx.as_of, holdings=holdings)
        c.insert(0, "horizon", h)
        c["reasons"] = c["reasons"].map(lambda r: ",".join(r))
        parts.append(c)
    out = pd.concat(parts)
    out.index.name = "ticker"
    return out.reset_index()


# ------------------------------------------------------------------ F9 模板报告
def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else (f"{v:.4g}" if isinstance(v, float) else str(v))
                                       for v in r) + " |")
    return "\n".join(lines)


def report(ctx: Ctx, res: dict) -> str:
    o = res["outputs"]
    snap = res["snapshot"]
    f1 = pd.read_parquet(o["f1"])
    dom = pd.read_parquet(o["f2"])
    cand = pd.read_parquet(o["f4"])
    L = [f"# 主线 A v2 影子日报 · {ctx.as_of}", "",
         "> 研究用只读影子运行，不是投资建议，不构成交易指令。资金批准状态：NOT_APPROVED。", "",
         f"- 代码版本：`{ctx.code_rev}`", f"- F0 快照：`{snap['content_hash'][:16]}`（data_ready = {snap['data_ready']}）"]
    if snap["stale"]:
        L.append("- 滞后输入：" + "，".join(f"{k}（到 {v}）" for k, v in sorted(snap["stale"].items())))
    if snap["absent"]:
        L.append("- 缺失输入：" + "，".join(snap["absent"]))
    L += ["", "## 1 市场与证券状态（F1）", ""]
    st = {"在市股票": len(f1), "主板": int((f1["board"] == "MAIN").sum()), "创业板": int((f1["board"] == "CHINEXT").sum()),
          "科创板": int((f1["board"] == "STAR").sum()), "北交所": int((f1["board"] == "BSE").sum()),
          "ST": int(f1["is_st"].sum()), "停牌": int(f1["suspended"].sum()), "收盘涨停": int(f1["limit_up_close"].sum()),
          "一字涨停": int(f1["one_line_up"].sum()), "收盘跌停": int(f1["limit_down_close"].sum()),
          "上市不满 120 日": int(f1["c_listed_lt_min"].sum())}
    L.append(_md_table(pd.DataFrame([st])))
    L += ["", "## 2 支持域（F2）", "", _md_table(dom["domain"].value_counts().rename_axis("域").reset_index(name="股票数")),
          "", "## 3 参考臂打分（F3）", ""]
    rows = []
    for arm, key, col in (("ridge_native_20d", "ridge_20d", "score"), ("ens_native_20d", "ens_20d", "ENS"),
                          ("ridge_native_63d", "ridge_63d", "score")):
        s = _scores(o, key, col)
        if s is None:
            continue
        pool = s.reindex(dom.index[dom["domain"] == "research_pool"]).dropna()
        rows.append({"参考臂": arm, "打分行数": len(s), "原研究池内": len(pool)})
    for arm, why in res["unavailable"].items():
        rows.append({"参考臂": arm, "打分行数": "不可用", "原研究池内": why})
    L.append(_md_table(pd.DataFrame(rows)))
    e, r = _scores(o, "ens_20d", "ENS"), _scores(o, "ridge_20d")
    if e is not None and r is not None:
        pool = dom.index[dom["domain"] == "research_pool"]
        j = pd.DataFrame({"e": e, "r": r}).reindex(pool).dropna()
        L += ["", f"20 日 ENS 与 Ridge 在原研究池内的截面秩相关：{j['e'].rank().corr(j['r'].rank()):.3f}（{len(j)} 只）"]
    L += ["", "## 4 研究候选（F4）", "",
          "候选只用于安排研究和证据收集；事件队列与抽查样本不进入任何组合。", ""]
    for h, g in cand.groupby("horizon"):
        reasons = g["reasons"].str.split(",").explode().value_counts().rename_axis("进入原因").reset_index(name="数量")
        L += [f"**{h}**：{len(g)} 只", "", _md_table(reasons), ""]
        top = g.head(20).merge(f1[["name", "board"]], left_on="ticker", right_index=True, how="left")
        show = [c for c in ["ticker", "name", "board", "domain", "reasons", "ens20", "ridge20", "ridge63"] if c in top]
        L += [_md_table(top[show]), ""]
    L += ["## 5 已知限制", "",
          "- 北向资金字段 2024-08-19 起口径改变（T-035）：ENS 的树模型使用该市场级特征，Ridge 截面排序不受影响。",
          "- 北交所不在股票基础表与财报里，模型没有北交所输入：全开通线目前与除北交所线相同。",
          "- 300 只等权的参考组合在 10 万元规模不可执行（每只约 333 元），见 F8 容量情景。",
          "- 研究期评估只保留标签非缺失的行，前瞻无法这样过滤（标签尚未成熟）。", ""]
    return "\n".join(L)


# ------------------------------------------------------------------ F8 前瞻账本
def run_books(data_root: Path, model_root: Path, out_root: Path, as_of: str, js: JobStore,
              prices_path: Path) -> dict:
    from quantmind.research.gpu.folds import calendar_and_grid
    _, frozen = calendar_and_grid(data_root)
    full = pd.DatetimeIndex(pd.to_datetime(pd.read_parquet(data_root / "data/lake/trade_cal_sse.parquet")["cal_date"]
                                           .astype(str))).sort_values()
    grid = [d for d in f8_books.forward_grid(full, frozen) if d <= pd.Timestamp(as_of)]
    contracts = json.loads((CFG / "reference_strategies.json").read_text(encoding="utf-8"))["strategies"]
    stage_of = {"ridge_native_20d": ("f3_ridge_20d", "score"), "ens_native_20d": ("f3_ens_20d", "ENS"),
                "ridge_native_63d": ("f3_ridge_63d", "score")}
    recs = [dict(r) for r in js.db.execute("SELECT stage, key_fields, output_path FROM jobs WHERE status = ? "
                                           "ORDER BY finished_at", (SUCCEEDED,)).fetchall()]
    scores, used, missing = {}, {}, {}
    for arm, (stage, col) in stage_of.items():
        got = {}
        for r in recs:
            kf = json.loads(r["key_fields"])
            if r["stage"] == stage and pd.Timestamp(kf["as_of"]) in grid:
                got[kf["as_of"]] = r["output_path"]           # 同一天多次运行：取最后完成的一条
        if grid and all(str(d.date()) in got for d in grid):
            df = pd.concat([pd.read_parquet(got[str(d.date())]) for d in grid], ignore_index=True)
            scores[arm] = df.rename(columns={col: "score"})[["as_of", "ticker", "score"]]
            used[arm] = [got[str(d.date())] for d in grid]
        else:
            missing[arm] = [str(d.date()) for d in grid if str(d.date()) not in got]
    cols = ["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "adj_factor", "amount"]
    prices = pd.read_parquet(prices_path, columns=cols)
    months = sorted({d.strftime("%Y-%m") for d in grid})
    ami = pd.concat([pd.read_parquet(model_root / f"data/gpu_v1/panel/base/{m}.parquet", columns=["amihud_illiquidity"])
                     for m in months]) if months else None
    profiles = load_profiles(CFG / "account_profiles.yaml")
    res = f8_books.run_forward_books(scores, prices, contracts, profiles, ami)
    od = out_root / "books" / as_of
    od.mkdir(parents=True, exist_ok=True)
    summary = {"as_of": as_of, "grid": [str(d.date()) for d in grid], "inputs": used, "prices": str(prices_path),
               "prices_sha256": sha256_file(prices_path), "arms_missing_scores": missing, "books": {}}
    hold = set()
    for arm, lines in res.items():
        for prof, b in lines.items():
            tag = f"{arm}__{prof}"
            for k in ("nav", "targets", "capacity", "rejected"):
                if isinstance(b[k], pd.DataFrame) and len(b[k]):
                    df = b[k].copy()
                    if k == "targets":
                        df["targets"] = df["targets"].map(list)
                    df.to_parquet(od / f"{tag}__{k}.parquet")
            h = b["holdings"]
            if len(h):
                last = h[h["date"] == h["date"].max()]
                hold |= set(last["ticker"])
            summary["books"][tag] = b["summary"]
    pd.DataFrame({"ticker": sorted(hold)}).to_parquet(od / "holdings_latest.parquet")
    atomic_write_json(od / "books_summary.json", summary)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("day", "books"):
        p = sub.add_parser(name)
        p.add_argument("--as-of", required=True)
        p.add_argument("--data-root", type=Path, required=True)
        p.add_argument("--model-root", type=Path, required=True)
        p.add_argument("--out-root", type=Path)
        if name == "day":
            p.add_argument("--bundle20", type=Path)
        else:
            p.add_argument("--prices", type=Path, required=True)
    a = ap.parse_args()
    out_root = a.out_root or a.data_root / "data/line_a_v2"
    js = JobStore(out_root / "jobs.sqlite")
    if a.cmd == "day":
        ctx = Ctx(a.as_of, a.data_root, a.model_root, out_root, a.bundle20)
        res = run_day(ctx, js)
    else:
        res = run_books(a.data_root, a.model_root, out_root, a.as_of, js, a.prices)
        res = {k: v for k, v in res.items() if k != "books"} | {"books": sorted(res["books"])}
    print(json.dumps(res, ensure_ascii=False, default=str, indent=1))


if __name__ == "__main__":
    main()
