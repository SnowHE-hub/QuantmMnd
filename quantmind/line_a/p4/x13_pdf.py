"""quantmind.line_a.p4.x13_pdf — X13 PDF 解析：年报开发集、文本层基线解析器与打分（03_final_plan_v2 第 9.1、10.4 节 X13；P4）.

开发集 v1（`build_devset`）：
- 年度报告正文：不含摘要、英文版、审计报告、督导报告、更正或补充；报告年度 2019–2023
  （这几年的巨潮元数据已补完）；沪主板、深主板、创业板、科创板各 10 份，每个板块每年 2 份，每只股票最多一份；种子 20261021；
- 北交所的财报不在 lake 的参考臂表里，第一版不抽；
- 标准值来自 lake 的合并报表（元）。lake 每期只留一个版本：ann_date 与年报披露日相差超过 7 天的，
  视为事后重述的版本，这张表的字段不进开发集（年报里印的是原始数）；
- 字段（11 个）见 FIELDS。金融企业没有的字段（标准值为空）跳过。

基线解析器 pymupdf-rules-v1（`extract`）：
- 只读文本层，用单词坐标拼出视觉上的行；
- 报表定位：独占一行的标题"合并资产负债表 / 合并利润表 / 合并现金流量表"开始一段，到下一个报表标题为止；
  同名的段有多个时（目录、正文引用），取找到目标行最多的那段；母公司报表不用；
- 行匹配：去掉"一、""其中：""加：""1."之类的前缀和括号里的说明后，行标签以目标标签开头；
  标签后的第一个金额是本期，第二个是上期；附注编号（不带千分位、不带小数点的短整数）跳过；
- 单位按段内的"单位：元 / 千元 / 万元 / 百万元 / 亿元"换算；括号数和带负号的数都是负数；
- 每个值记录证据坐标（页码、bbox）和读出的小数位数。

打分（`score`），每个字段一类结果：
- correct：|x − g| 不超过读出精度的半个末位单位（至少 0.011 元），再加 1e-9·|g| 的浮点余量；
- unit_error：差 10³、10⁴、10⁶ 或 10⁸ 倍；sign_error：只差符号；
- period_error：等于上期标准值（列取错）；wrong：找到了数但都不是；not_found：没找到。
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

PARSER = "pymupdf-rules-v3"   # v1 → v2：在开发集上改了 4 轮（编号标题、折行标签、横版、连续同类标题）；
#                               v2 → v3：认不带"合并"的单体报表标题（看过 v2 测试集的失败之后改，所以 v3 用新抽的测试集 v2 评估），见 T-038
SEED = 20261021
YEARS = range(2019, 2024)
PER_BOARD_YEAR = 2
RESTATE_DAYS = 7

# 字段：(表, 报表段, 标签候选)。标签按"去掉前缀和括号说明后"的形式写
FIELDS = {
    "total_revenue": ("income", "IS", ("营业总收入",)),
    "revenue": ("income", "IS", ("营业收入",)),
    "operate_profit": ("income", "IS", ("营业利润",)),
    "n_income": ("income", "IS", ("净利润",)),
    "n_income_attr_p": ("income", "IS", ("归属于母公司股东的净利润", "归属于母公司所有者的净利润",
                                         "归属于母公司普通股股东的净利润", "归属于母公司的净利润")),
    "total_assets": ("balancesheet", "BS", ("资产总计",)),
    "total_liab": ("balancesheet", "BS", ("负债合计",)),
    "total_hldr_eqy_exc_min_int": ("balancesheet", "BS", ("归属于母公司所有者权益合计", "归属于母公司股东权益合计",
                                                          "归属于母公司所有者权益", "归属于母公司股东的权益合计")),
    "total_cur_assets": ("balancesheet", "BS", ("流动资产合计",)),
    "total_cur_liab": ("balancesheet", "BS", ("流动负债合计",)),
    "n_cashflow_act": ("cashflow", "CF", ("经营活动产生的现金流量净额",)),
}
HEADINGS = {"合并资产负债表": "BS", "合并利润表": "IS", "合并现金流量表": "CF",
            "合并及母公司资产负债表": "BS", "合并及母公司利润表": "IS", "合并及母公司现金流量表": "CF",
            "合并及公司资产负债表": "BS", "合并及公司利润表": "IS", "合并及公司现金流量表": "CF",
            "母公司资产负债表": "P_BS", "母公司利润表": "P_IS", "母公司现金流量表": "P_CF",
            "合并所有者权益变动表": "EQ", "合并股东权益变动表": "EQ", "母公司所有者权益变动表": "P_EQ",
            "母公司股东权益变动表": "P_EQ",
            # v3：没有子公司的公司只有单体报表，标题不带"合并"（例如 301040.SZ 2022、688538.SH 2023）
            "资产负债表": "BS", "利润表": "IS", "现金流量表": "CF", "所有者权益变动表": "EQ", "股东权益变动表": "EQ"}
STANDALONE = {"资产负债表", "利润表", "现金流量表", "所有者权益变动表", "股东权益变动表"}
# 单体报表没有"归属于母公司"那一行；数据源对这类公司的归母口径就是全体口径
STANDALONE_FALLBACK = {"n_income_attr_p": ("净利润",),
                       "total_hldr_eqy_exc_min_int": ("所有者权益合计", "股东权益合计")}
UNITS = {"元": 1.0, "千元": 1e3, "万元": 1e4, "百万元": 1e6, "亿元": 1e8}
_TITLE = re.compile(r"(\d{4})\s*年?\s*年度报告(?:全文)?(?:（修订版）|\(修订版\)|（更新后）)?$")
_EXCLUDE = re.compile(r"摘要|英文|审计|督导|更正|补充|取消|说明|问询|半年|季度|核查|鉴证|专项|提示")
_NUM = re.compile(r"^[（(]?-?\d[\d,]*(?:\.\d+)?[)）]?$")
_AMOUNT = re.compile(r"[（(]?-?\d{1,3}(?:,\d{3})*\.\d{2}[)）]?")
_AMOUNTS = re.compile(r"(?:[（(]?-?\d{1,3}(?:,\d{3})*\.\d{2}[)）]?){2,}")
_PREFIX = re.compile(r"^(?:[一二三四五六七八九十]+、|[（(][一二三四五六七八九十\d]+[)）]|\d+[.．、]|其中[:：]|加[:：]|减[:：])+")
_PAREN = re.compile(r"[（(][^）)]*[)）]")


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _board(ts_code: str) -> str:
    c, ex = ts_code[:3], ts_code[-2:]
    if c in ("688", "689"):
        return "STAR"
    if c in ("300", "301"):
        return "CHINEXT"
    if ex == "BJ":
        return "BSE"
    return "MAIN_SH" if ex == "SH" else "MAIN_SZ"


# ---------------------------------------------------------------- 开发集
def annual_reports(cninfo_parts: Path, years=YEARS) -> pd.DataFrame:
    """元数据里的年报正文（报告年度 years，次年 1–6 月披露）。"""
    rows = []
    for y in years:
        for p in sorted(Path(cninfo_parts).glob(f"{y + 1}0[1-6]*.parquet")):
            df = pd.read_parquet(p, columns=["announcementId", "secCode", "announcementTitle", "adjunctUrl",
                                             "adjunctSize", "publish_ts_bj"])
            t = df["announcementTitle"].astype(str).str.strip()
            m = t.str.extract(_TITLE)[0]
            keep = (m == str(y)) & ~t.str.contains(_EXCLUDE)
            rows.append(df[keep].assign(year=y, title=t[keep]))
    out = pd.concat(rows, ignore_index=True).drop_duplicates("announcementId")
    code = out["secCode"].astype(str).str.zfill(6)
    suf = code.str[0].map({"6": ".SH", "0": ".SZ", "3": ".SZ"})
    out = out[suf.notna()].assign(ts_code=code + suf,
                                  date=out["publish_ts_bj"].astype(str).str.slice(0, 10).str.replace("-", ""))
    out = out.sort_values(["ts_code", "year", "date"]).drop_duplicates(["ts_code", "year"], keep="last")  # 修订版在后
    return out.assign(board=out["ts_code"].map(_board)).reset_index(drop=True)


def _ymd(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s.astype(str).str.replace("-", "").str.slice(0, 8), format="%Y%m%d",
                          errors="coerce").dt.strftime("%Y%m%d")


def gold_for(lake: Path, ts_code: str, year: int, pub_date: str, tables: dict | None = None) -> dict:
    """{字段: {value, prior, excluded}}。prior 是上年同期的 lake 值（只用来判断列是否取错）。"""
    out = {}
    for tbl in ("income", "balancesheet", "cashflow"):
        df = tables[tbl] if tables else pd.read_parquet(Path(lake) / f"{tbl}.parquet",
                                                        filters=[("ts_code", "==", ts_code)])
        df = df[df["ts_code"] == ts_code]
        end = _ymd(df["end_date"])
        cur = df[end == f"{year}1231"]
        pri = df[end == f"{year - 1}1231"]
        restated = True
        if len(cur):
            ann = pd.Timestamp(_ymd(cur["ann_date"]).iloc[-1])
            restated = abs((ann - pd.Timestamp(pub_date)).days) > RESTATE_DAYS
        for f, (t, _, _) in FIELDS.items():
            if t != tbl:
                continue
            v = cur[f].iloc[-1] if len(cur) and f in cur else None
            pv = pri[f].iloc[-1] if len(pri) and f in pri else None
            out[f] = {"value": None if v is None or pd.isna(v) else float(v),
                      "prior": None if pv is None or pd.isna(pv) else float(pv),
                      "excluded": "restated_or_missing" if restated else None}
    return out


def build_devset(lake: Path, out_dir: Path, seed: int = SEED, *, version: str = "x13-dev-v1",
                 exclude: set | None = None) -> pd.DataFrame:
    """开发集和测试集用同一套规则抽样；测试集换种子，并排除开发集里出现过的股票（exclude）。"""
    lake, out_dir = Path(lake), Path(out_dir)
    ar = annual_reports(lake / "text/cninfo/_parts")
    ar = ar[ar["board"] != "BSE"]
    picks, used = [], set(exclude or ())
    for (_b, y), g in ar.groupby(["board", "year"]):
        g = g[~g["ts_code"].isin(used)].sample(frac=1.0, random_state=seed + y)
        sel = g.head(PER_BOARD_YEAR)
        used |= set(sel["ts_code"])
        picks.append(sel)
    cases = pd.concat(picks, ignore_index=True)
    tick = sorted(cases["ts_code"])
    tables = {t: pd.read_parquet(lake / f"{t}.parquet", filters=[("ts_code", "in", tick)])
              for t in ("income", "balancesheet", "cashflow")}
    gold = []
    for r in cases.itertuples(index=False):
        g = gold_for(lake, r.ts_code, r.year, r.date, tables)
        for f, v in g.items():
            gold.append({"ann_id": str(r.announcementId), "ts_code": r.ts_code, "year": r.year, "field": f, **v})
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = cases[["announcementId", "ts_code", "board", "year", "date", "title", "adjunctUrl", "adjunctSize"]]
    cases.to_csv(out_dir / "cases.csv", index=False, encoding="utf-8-sig")
    gd = pd.DataFrame(gold)
    gd.to_csv(out_dir / "gold.csv", index=False, encoding="utf-8-sig")
    man = {"version": version, "seed": seed, "n_excluded_tickers": len(exclude or ()), "years": list(YEARS), "per_board_year": PER_BOARD_YEAR,
           "n_cases": len(cases), "est_download_mb": round(float(cases["adjunctSize"].astype(float).sum()) / 1024, 1),
           "n_gold": int(gd["value"].notna().sum()), "n_gold_usable": int((gd["value"].notna() & gd["excluded"].isna()).sum()),
           "cases_sha256": _sha((out_dir / "cases.csv").read_text(encoding="utf-8-sig")),
           "gold_sha256": _sha((out_dir / "gold.csv").read_text(encoding="utf-8-sig"))}
    (out_dir / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    return cases


# ---------------------------------------------------------------- 基线解析器
def _rows_of(words) -> list[list[tuple]]:
    words = sorted(words, key=lambda w: ((w[1] + w[3]) / 2, w[0]))
    rows, cur, yc = [], [], None
    for w in words:
        c = (w[1] + w[3]) / 2
        if yc is not None and abs(c - yc) > 3:
            rows.append(sorted(cur, key=lambda x: x[0]))
            cur = []
        cur.append(w[:5])
        yc = c if len(cur) == 1 else (yc * (len(cur) - 1) + c) / len(cur)
    if cur:
        rows.append(sorted(cur, key=lambda x: x[0]))
    return rows


def _rows(page) -> list[tuple[float, list[tuple]]]:
    """单词坐标 → 视觉行（按 y 中心聚类，容差 3 pt），行内按 x 排序。每个词 (x0, y0, x1, y1, text)。
    返回 (阅读顺序坐标, 行)。横版"一张纸印两页"的版式（宽 > 1.2 × 高）先切成左右两半，右半的阅读顺序排在左半之后。"""
    words = page.get_text("words")
    w, h = page.rect.width, page.rect.height
    if w > 1.2 * h:
        mid = w / 2
        left = [x for x in words if (x[0] + x[2]) / 2 < mid]
        right = [x for x in words if (x[0] + x[2]) / 2 >= mid]
        return [(r[0][1], r) for r in _rows_of(left)] + [(h + r[0][1], r) for r in _rows_of(right)]
    return [(r[0][1], r) for r in _rows_of(words)]


def _num(tok: str) -> tuple[float, int] | None:
    """金额 token → (值, 小数位)。附注编号这类不带千分位和小数点的短整数返回 None。"""
    t = tok.strip()
    if t in ("-", "—", "–", "－"):
        return 0.0, 2
    if not _NUM.match(t):
        return None
    neg = t.startswith(("(", "（")) or t.lstrip("(（").startswith("-")
    body = t.strip("()（）").lstrip("-").replace(",", "")
    if "," not in t and "." not in t and len(body) < 5:
        return None
    dec = len(body.split(".")[1]) if "." in body else 0
    v = float(body)
    return (-v if neg else v), dec


def _norm_label(s: str) -> str:
    s = re.sub(r"\s+", "", s)
    s = _PAREN.sub("", s)
    return _PREFIX.sub("", s)


def _segments(doc) -> list[dict]:
    """报表段：{kind, pages, start_y, end_y, unit}。从标题行下方开始，到下一个报表标题的上方为止
    （下一个标题所在页里、标题上方的行仍属于本段，因为报表常常跨页接着印）。"""
    marks = []
    for pno in range(doc.page_count):
        for vy, row in _rows(doc[pno]):
            line = re.sub(r"\s+", "", "".join(w[4] for w in row))
            line = re.sub(r"[（(]续[)）]$", "", line)
            line = _PREFIX.sub("", line)               # 深交所格式的标题带编号："1、合并资产负债表"
            if line in HEADINGS:
                if marks and marks[-1][2] == HEADINGS[line]:
                    continue    # 连续的同类标题（"（续）"、横版每半页重复的表头）是同一张表接着印，不另起一段
                marks.append((pno, vy, HEADINGS[line], line in STANDALONE))
    segs = []
    for i, (pno, y, kind, standalone) in enumerate(marks):
        if i + 1 < len(marks):
            end, end_y = marks[i + 1][0], marks[i + 1][1]
        else:
            end, end_y = min(doc.page_count - 1, pno + 3), float("inf")
        pages = list(range(pno, max(pno, end) + 1))
        text = "".join(doc[p].get_text("text") for p in pages[:2])
        m = re.search(r"单位[:：]\s*(?:人民币)?\s*(百万元|千元|万元|亿元|元)", text)
        segs.append({"kind": kind, "pages": pages, "start_y": y, "end_y": end_y,
                     "unit": UNITS[m.group(1)] if m else 1.0, "standalone": standalone})
    return segs


def _tokens(row) -> list[tuple]:
    """把粘在一起的几个金额（列间距太小时文本层会连成一个词，例如 5,000.004,000.00）拆开。"""
    out = []
    for w in row:
        t = w[4]
        if not _NUM.match(t) and _AMOUNTS.fullmatch(t):
            out.extend((*w[:4], a) for a in _AMOUNT.findall(t))
        else:
            out.append(w)
    return out


def _split(row) -> tuple[str, list]:
    lab, nums = [], []
    for w in _tokens(row):
        n = _num(w[4])
        if n is None and not nums:
            lab.append(w[4])
        elif n is not None:
            nums.append((n, w))
    return _norm_label("".join(lab)), nums


def _find(doc, seg: dict, labels: tuple) -> dict | None:
    """折行标签有三种：前半截单独一行、数字在下一行（pending + 本行）；数字跟着前半截、后半截在下一行（本行 + 下一行）；
    两截各占一行、数字在中间（pending + 本行 + 下一行）。"""
    rows = []
    for pno in seg["pages"]:
        for vy, row in _rows(doc[pno]):
            if (pno == seg["pages"][0] and vy <= seg["start_y"]) or (pno == seg["pages"][-1] and vy >= seg["end_y"]):
                continue
            rows.append((pno, *_split(row)))
    pending = ""
    for i, (pno, text, nums) in enumerate(rows):
        if not nums:                           # 只有文字的行：可能是折行标签的前半截，留给下一行拼接
            pending = text
            continue
        nxt = rows[i + 1][1] if i + 1 < len(rows) and not rows[i + 1][2] else ""
        # 第三种：标签上下两截、数字单独在中间一行（按行中心排序后，数字行夹在两截之间）
        for cand in (text, pending + text, pending, text + nxt if nxt else "", pending + text + nxt if nxt else ""):
            if cand and any(cand.startswith(x) for x in labels):
                (v, dec), w = nums[0]
                prior = nums[1][0][0] * seg["unit"] if len(nums) > 1 else None
                return {"value": v * seg["unit"], "decimals": dec, "unit": seg["unit"], "page": pno + 1,
                        "bbox": [round(c, 1) for c in w[:4]], "label": cand, "prior": prior}
        pending = ""
    return None


def extract(path: Path) -> dict:
    """{字段: 结果或 None}，外加 _meta（页数、找到的报表段）。"""
    import fitz
    doc = fitz.open(str(path))
    segs = _segments(doc)
    out: dict = {"_meta": {"parser": f"{PARSER}/fitz-{fitz.VersionBind}", "pages": doc.page_count,
                           "segments": [{"kind": s["kind"], "first_page": s["pages"][0] + 1, "unit": s["unit"]}
                                        for s in segs]}}
    for kind in ("BS", "IS", "CF"):
        fields = [f for f, (_, k, _) in FIELDS.items() if k == kind]
        best: dict = {}
        for s in (s for s in segs if s["kind"] == kind):           # 同名的段取找到目标行最多的那段
            found = {f: _find(doc, s, FIELDS[f][2]) for f in fields}
            if kind == "IS" and found["total_revenue"] is None:   # 没有"营业总收入"行的公司，营业总收入就是营业收入
                found["total_revenue"] = found["revenue"]
            if s.get("standalone"):                               # 单体报表：归母口径就是全体口径
                for f, labels in STANDALONE_FALLBACK.items():
                    if f in found and found[f] is None:
                        found[f] = _find(doc, s, labels)
            if sum(v is not None for v in found.values()) > sum(v is not None for v in best.values()):
                best = found
        for f in fields:
            out[f] = best.get(f)
    return out


def run_baseline(dev_dir: Path, cache_dir: Path, out_dir: Path) -> dict:
    """下载（已缓存的跳过）开发集里的年报，跑基线解析器并打分。"""
    import time

    from quantmind.line_a.evidence.fetch import fetch_pdf
    dev_dir, out_dir = Path(dev_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = pd.read_csv(dev_dir / "cases.csv", dtype={"announcementId": str})
    gold = pd.read_csv(dev_dir / "gold.csv", dtype={"ann_id": str})
    results, fetch_log = {}, []
    for r in cases.itertuples(index=False):
        t0 = time.time()
        try:
            f = fetch_pdf(r.announcementId, r.adjunctUrl, cache_dir)
        except Exception as e:  # noqa: BLE001  下载失败记下来，不算解析器的错
            fetch_log.append({"ann_id": r.announcementId, "error": str(e)[:160]})
            continue
        fetch_log.append({"ann_id": r.announcementId, "bytes": f["bytes"], "sha256": f["sha256"], "cached": f["cached"]})
        t1 = time.time()
        res = extract(Path(f["path"]))
        res["_meta"].update({"fetch_s": round(t1 - t0, 1), "parse_s": round(time.time() - t1, 1)})
        results[r.announcementId] = res
        print(f"  {r.ts_code} {r.year} pages={res['_meta']['pages']} parse {res['_meta']['parse_s']}s", flush=True)
    (out_dir / "extract.json").write_text(json.dumps(results, ensure_ascii=False, indent=0), encoding="utf-8")
    (out_dir / "fetch_log.json").write_text(json.dumps(fetch_log, ensure_ascii=False, indent=0), encoding="utf-8")
    s = score(results, gold)
    s["parser"] = PARSER
    s["n_fetch_failed"] = sum("error" in x for x in fetch_log)
    s["parse_seconds_mean"] = float(np.mean([v["_meta"]["parse_s"] for v in results.values()])) if results else None
    (out_dir / "score.json").write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")
    return s


# ---------------------------------------------------------------- 打分
def _close(x: float, g: float, dec: int, unit: float) -> bool:
    return abs(x - g) <= max(0.011, 0.5 * 10 ** (-dec)) * unit + 1e-9 * abs(g)


def score_field(r: dict | None, g: dict) -> str:
    if g.get("excluded") or g.get("value") is None:
        return "skipped"
    if r is None:
        return "not_found"
    x, v, dec, unit = r["value"], g["value"], r.get("decimals", 2), r.get("unit", 1.0)
    if _close(x, v, dec, unit):
        return "correct"
    if v and any(abs(x * k - v) <= 1e-4 * abs(v) or abs(x / k - v) <= 1e-4 * abs(v) for k in (1e3, 1e4, 1e6, 1e8)):
        return "unit_error"
    if _close(-x, v, dec, unit):
        return "sign_error"
    if g.get("prior") is not None and _close(x, g["prior"], dec, unit):
        return "period_error"
    return "wrong"


def score(results: dict[str, dict], gold: pd.DataFrame) -> dict:
    """results：{ann_id: extract() 的输出}。返回按字段和总体的结果分布与准确率。"""
    recs = []
    for g in gold.to_dict("records"):
        r = results.get(str(g["ann_id"]))
        if r is None:
            continue
        g = {k: (None if isinstance(v, float) and pd.isna(v) else v) for k, v in g.items()}
        recs.append({"ann_id": str(g["ann_id"]), "field": g["field"], "outcome": score_field(r.get(g["field"]), g)})
    df = pd.DataFrame(recs)
    if df.empty:
        return {"n_docs": 0}
    use = df[df["outcome"] != "skipped"]
    by = use.groupby("field")["outcome"].value_counts().unstack(fill_value=0)
    return {"n_docs": int(df["ann_id"].nunique()), "n_scored": int(len(use)),
            "accuracy": float((use["outcome"] == "correct").mean()) if len(use) else None,
            "outcomes": use["outcome"].value_counts().to_dict(),
            "by_field": {f: {k: int(v) for k, v in row.items()} for f, row in by.iterrows()},
            "records": recs}
