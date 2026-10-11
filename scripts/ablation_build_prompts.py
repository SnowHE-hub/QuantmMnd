"""scripts/ablation_build_prompts.py — T-007 匿名化消融：生成请求并冻结（CPU，不调用模型）.

设计 v3（回应 G1 评审，见 docs/work/tasks/T-007.md "设计 v3"）。本实验是 **19 期探索性估计**，研究问题限定为：
在固定候选集合、数值信息、日期和推理配置下，向模型提供真实股票标识（相对随机匿名标识）是否改变其重排表现、改变多少。
**不**把任何结果解释为"因子推理"或"预训练记忆占比"。

四组（同一批 Top-30 候选、同一 system prompt、同一行次序、同一输出 schema）：

  组      真实代码   七因子值   LGBM 排名 / 候选顺序
  A       提供       提供       与生产一致
  B       未提供     提供       同 A
  Cstar   提供       未提供     同 A
  Dstar   未提供     未提供     同 A

  所有组都用随机编号 S01–S30 作行 ID 与输出 ID（输出 schema 完全相同，enum 按编号排序、与排名无关）；
  真实代码只作为 A / Cstar 表格里的一列出现，B / Dstar 该列写"未提供"。
  每个截面用 N_REPEATS 个预先冻结的随机编号映射（同一重复内四组共用一个映射），分析时先在截面内对重复求平均。

探索性附加请求（不进主判读）：
  A_nodate / B_nodate  重复 0 的 A / B，日期改为"未提供"（标识效应对日期可见性的敏感性）
  PROBE                重复 0 的 B 输入，要求模型猜每个编号对应的 6 位代码（身份探测审计）

冻结输出（目录已存在则拒绝运行）：
  requests.jsonl          每行 {request_id, as_of, group, repeat, payload}；payload 就是执行端发送的 JSON（不含 model 名）
  private_mapping.jsonl   每个 (as_of, repeat) 的 {id, ticker, lgbm_rank, prompt_pos}——**不进 payload**
  truth.csv               (as_of, ticker, forward_return_63d, 七因子)，只用于分析（七因子用于事先固定的简单规则对照）
  selection.csv           每个截面纳入 / 剔除及原因
  time_audit.csv          每期决策日、收益结束日、按知识截止区间的暴露分类
  manifest.json           设计版本、输入 sha256、模板 sha256、解码配置、映射种子、模型锁定值、知识截止依据
匿名组（B、Dstar、B_nodate、PROBE）的完整 payload 在冻结前通过 ``anonymization_violations``。
它证明 payload 里**没有字面标识**，不证明没有身份信息（因子向量、日期、LGBM 排名仍可能被用来识别个股）。

用法（仓库根目录）：
  python scripts/ablation_build_prompts.py --scores <T-010 oos_scores.csv> --panel data/panel/alpha_panel_v4_2.parquet \
      --stock-basic data/lake/stock_basic_full.parquet data/lake/stock_basic_full_20261009.parquet \
      --trade-calendar data/lake/trade_cal_sse.parquet --lock <run_*.json of the engineering pilot> --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DESIGN_VERSION = "t007-design-v3"
FACTORS = ["pe_ttm", "pb", "roe_ttm", "accruals", "distance_to_52w_high", "momentum_6m", "volatility_3m"]
LABEL = "forward_return_63d"
HORIZON_TD = 63
MAIN_GROUPS = ("A", "B", "Cstar", "Dstar")
ID_VISIBLE = {"A": True, "B": False, "Cstar": True, "Dstar": False, "A_nodate": True, "B_nodate": False}
FACTORS_VISIBLE = {"A": True, "B": True, "Cstar": False, "Dstar": False, "A_nodate": True, "B_nodate": True}
DATE_VISIBLE = {"A": True, "B": True, "Cstar": True, "Dstar": True, "A_nodate": False, "B_nodate": False}
EXPLORATORY_GROUPS = ("A_nodate", "B_nodate")
PROBE_GROUP = "PROBE"
ANON_GROUPS = ("B", "Dstar", "B_nodate", PROBE_GROUP)
N_REPEATS = 5
DECODING = {"temperature": 0, "seed": 0, "max_tokens": 1500, "chat_template_kwargs": {"enable_thinking": False}}
NA = "未提供"

# 知识截止：没有官方声明。下界来自 8 条探针（2026-09-24，弱证据）；上界取上游模型公开发布日（训练数据不可能晚于发布）
KNOWLEDGE_CUTOFF = {
    "conservative_bounds": ["2025-01-01", "2026-08-14"],
    "probe_estimate_assumption": ["2025-01-01", "2025-03-31"],
    "evidence": [
        "GGUF: Ternary-Bonsai-2-27B-PTQ1_0（PrismML，2026-09-17 发布）；模型卡列出的上游为 Qwen/Qwen3.8-27B",
        "上游 Qwen3.8-27B 公开发布 2026-08-14（NVIDIA NIM 目录、DataLearner）；所查模型卡与资料均未声明训练数据截止日",
        "下界 2025-01-01 来自本机 8 条知识探针（升级方案 §20），不是证明",
    ],
}

SYSTEM_PROMPT_V3 = """\
你是一名专注于 A 股 Alpha 1374 票池的量化研究员。
任务：对候选股票列表进行精细排名，并提供组合级别的投资分析。

信息说明：
- 表格每行有一个编号（S01、S02……）。编号随机分配，与排名、代码都无关；输出时只用编号指代股票。
- 本组可能不提供某些信息（表中标为"未提供"）。请只根据本组实际提供的信息排序；某列未提供时不要拒绝作答，也不要据此惩罚。
- 若提供因子值，它们是截面标准分，不是原始财务比率：0 为截面均值，正值高于平均，负值低于平均。

评估维度（仅当对应信息已提供；重要性依次递减）：
1. 盈利质量：ROE_TTM 高、应计利润率（accruals）低
2. 估值：PE_TTM 和 PB 综合判断（避免极高估值）
3. 动量与技术：距 52 周高点近（强势）、6 个月动量为正
4. 风险控制：3 个月波动率低优先，但强动量股可适当放宽
5. LGBM 排名：模型粗排结果，可作参考

安全规则：
- 所有数据已截止于 as_of 日期（PIT 原则），不包含任何未来信息
- 严禁基于训练数据中的股票历史知识做主观判断，只能依据本组表格中实际提供的信息
- reason 中引用的具体数字必须来自输入数据，不得捏造
- NaN 表示该项数据缺失，缺失维度不作惩罚

输出格式（严格按此 JSON，不加任何说明文字）：
{
  "rankings": [{"id":"编号","reason":"个股理由（30字内）"}, ...],
  "portfolio_thesis": "整体组合投资逻辑（100字内）",
  "risk_warnings": ["风险1（具体）", "风险2（具体）", "风险3（具体）"]
}
"""

USER_TEMPLATE_V3 = """\
截止日期（as_of）：{as_of_date}
候选股票数量：{n_candidates}
请从以下 {n_candidates} 只股票中选出并排序前 {top_n} 名（第1名=最优）。

本组提供的信息：真实代码：{id_flag}；七个因子值：{factor_flag}；LGBM 排名：提供。
因子值（若提供）为截面标准分：0 为截面均值，正值高于平均。

| 编号 | 真实代码 | LGBM排名 | PE_TTM | PB | ROE_TTM标准分 | Accruals | 距52W高标准分 | 动量6M标准分 | 波动3M标准分 |
|-----|---------|---------|--------|----|--------------|----------|--------------|-------------|-------------|
{table_rows}

请严格按如下格式输出 JSON（只输出 JSON，不要有任何其他文字）：
{{
  "rankings": [{{"id":"编号","reason":"理由（30字内）"}}, ...],
  "portfolio_thesis": "组合投资逻辑（100字内）",
  "risk_warnings": ["风险1", "风险2", "风险3"]
}}
"""

PROBE_SYSTEM = """\
你是一名熟悉 A 股上市公司的研究员。下面的表格来自一次匿名化实验：每行用随机编号（S01、S02……）代替真实股票。
请判断每个编号最可能对应哪家 A 股公司，给出 6 位数字代码；无法判断时填 unknown。不要输出任何解释。
"""

PROBE_USER = """\
截止日期（as_of）：{as_of_date}
下表因子值为截面标准分：0 为截面均值，正值高于平均。

| 编号 | LGBM排名 | PE_TTM | PB | ROE_TTM标准分 | Accruals | 距52W高标准分 | 动量6M标准分 | 波动3M标准分 |
|-----|---------|--------|----|--------------|----------|--------------|-------------|-------------|
{table_rows}

输出 JSON：{{"guesses": [{{"id":"编号","code":"6位代码或unknown"}}, ...]}}，每个编号恰好一次。
"""

# 6 位证券代码：前后都不是数字或小数点（避免把因子值 12.345678 里的数字串当成代码）
CODE_RE = re.compile(r"(?<![\d.])(\d{6})(?![\d.])")


def sha(text: str | bytes) -> str:
    return hashlib.sha256(text.encode("utf-8") if isinstance(text, str) else text).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def anon_ids(n: int) -> list[str]:
    return [f"S{k + 1:02d}" for k in range(n)]


def rankings_schema(ids: list[str], top_n: int) -> dict:
    """llama-server response_format：id 用 enum 约束在编号内（enum 按编号排序，不携带排名信息），恰好 top_n 个."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "rerank",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "rankings": {
                        "type": "array", "minItems": top_n, "maxItems": top_n,
                        "items": {
                            "type": "object",
                            "properties": {"id": {"type": "string", "enum": sorted(ids)},
                                           "reason": {"type": "string", "maxLength": 80}},
                            "required": ["id", "reason"],
                        },
                    },
                    "portfolio_thesis": {"type": "string", "maxLength": 200},
                    "risk_warnings": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
                },
                "required": ["rankings", "portfolio_thesis", "risk_warnings"],
            },
        },
    }


def probe_schema(ids: list[str]) -> dict:
    n = len(ids)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "identity_probe",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "guesses": {
                        "type": "array", "minItems": n, "maxItems": n,
                        "items": {
                            "type": "object",
                            "properties": {"id": {"type": "string", "enum": sorted(ids)},
                                           "code": {"type": "string", "pattern": "^([0-9]{6}|unknown)$"}},
                            "required": ["id", "code"],
                        },
                    },
                },
                "required": ["guesses"],
            },
        },
    }


# ── 匿名化检查 ──────────────────────────────────────────────────────────────
class IdentityDictionary:
    """全部上市 / 退市公司的代码与名称（简称、去掉 ST 前缀的简称、全称、英文名）.

    ``template_text``：各组共用的固定模板文字。与模板文字重合的名称（例如某简称去掉 ST 前缀后是"标准"，
    而模板里有"标准分"）从词典移到 ``template_collisions``，写进 manifest 供审计。
    """

    def __init__(self, stock_basic: pd.DataFrame, template_text: str = ""):
        self.codes = set(stock_basic["ts_code"].astype(str).str[:6])
        names: set[str] = set()
        for col in ("name", "fullname", "enname"):
            if col in stock_basic.columns:
                for v in stock_basic[col].dropna().astype(str):
                    v = v.strip()
                    names.add(v)
                    names.add(re.sub(r"^(\*?ST|S\*?ST|N|C|XD|XR|DR)\s*", "", v))
        names = {n for n in names if len(n) >= 2}
        low_tpl = template_text.lower()
        self.template_collisions = sorted(n for n in names if (n in template_text if not n.isascii() else
                                                               (len(n) >= 6 and n.lower() in low_tpl)))
        self.names = names - set(self.template_collisions)
        self.names_lower = {n.lower() for n in self.names if n.isascii()}

    def violations(self, text: str) -> list[str]:
        hits = {f"code:{c}" for c in CODE_RE.findall(text) if c in self.codes}
        low = text.lower()
        hits |= {f"name:{n}" for n in self.names if not n.isascii() and n in text}
        hits |= {f"name:{n}" for n in self.names_lower if len(n) >= 6 and n in low}
        return sorted(hits)


def schema_enum(payload: dict) -> list[str]:
    props = payload["response_format"]["json_schema"]["schema"]["properties"]
    key = "rankings" if "rankings" in props else "guesses"
    return props[key]["items"]["properties"]["id"]["enum"]


def anonymization_violations(payload: dict, ident: IdentityDictionary, allowed_ids: list[str]) -> list[str]:
    """匿名组整个 payload（所有文本字段 + 输出 schema）里的字面身份；另检查 schema 的 enum 只含匿名编号."""
    bad = ident.violations(json.dumps(payload, ensure_ascii=False))
    enum = schema_enum(payload)
    if sorted(enum) != sorted(allowed_ids) or any(not re.fullmatch(r"S\d{2}", x) for x in enum):
        bad.append("schema_enum_not_anonymous")
    return bad


# ── 构造 ──────────────────────────────────────────────────────────────────
def _fmt(v: float, d: int = 2) -> str:
    return "NaN" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"


def _row(sid: str, ticker: str, rank: int, fac: pd.Series, show_id: bool, show_fac: bool) -> str:
    code = ticker if show_id else NA
    if show_fac:
        cells = [_fmt(fac["pe_ttm"]), _fmt(fac["pb"]), _fmt(fac["roe_ttm"]), _fmt(fac["accruals"], 3),
                 _fmt(fac["distance_to_52w_high"]), _fmt(fac["momentum_6m"]), _fmt(fac["volatility_3m"])]
    else:
        cells = [NA] * len(FACTORS)
    return f"| {sid} | {code} | {rank} | " + " | ".join(cells) + " |"


def main_user_prompt(group: str, order: list[str], sid: dict[str, str], top: pd.DataFrame, top_n: int,
                     as_of: str) -> str:
    rows = [_row(sid[t], t, i + 1, top.loc[t], ID_VISIBLE[group], FACTORS_VISIBLE[group]) for i, t in enumerate(order)]
    return USER_TEMPLATE_V3.format(as_of_date=as_of if DATE_VISIBLE[group] else NA, n_candidates=len(order),
                                   top_n=top_n, id_flag="提供" if ID_VISIBLE[group] else NA,
                                   factor_flag="提供" if FACTORS_VISIBLE[group] else NA,
                                   table_rows="\n".join(rows))


def probe_user_prompt(order: list[str], sid: dict[str, str], top: pd.DataFrame, as_of: str) -> str:
    rows = []
    for i, t in enumerate(order):
        f = top.loc[t]
        rows.append(f"| {sid[t]} | {i + 1} | {_fmt(f['pe_ttm'])} | {_fmt(f['pb'])} | {_fmt(f['roe_ttm'])} | "
                    f"{_fmt(f['accruals'], 3)} | {_fmt(f['distance_to_52w_high'])} | {_fmt(f['momentum_6m'])} | "
                    f"{_fmt(f['volatility_3m'])} |")
    return PROBE_USER.format(as_of_date=as_of, table_rows="\n".join(rows))


def _request(day: str, group: str, repeat: int, payload: dict) -> dict:
    return {"request_id": sha(json.dumps(payload, ensure_ascii=False, sort_keys=True)), "as_of": day,
            "group": group, "repeat": repeat, "payload": payload}


def build_cross_section(as_of: pd.Timestamp, top: pd.DataFrame, rng: np.random.Generator, top_n: int,
                        n_repeats: int = N_REPEATS):
    """top：按样本外分数降序的 Top-K 行（index=ticker，含 score、FACTORS）。返回 (请求, 私有映射)."""
    order = list(top.index)                 # 行次序 = LGBM 排名（四组相同）
    n = len(order)
    day = str(as_of.date())
    ids = anon_ids(n)
    requests, mapping = [], []
    for r in range(n_repeats):
        sid = {t: ids[k] for t, k in zip(order, rng.permutation(n), strict=True)}
        mapping.extend({"as_of": day, "repeat": r, "id": sid[t], "ticker": t, "lgbm_rank": i + 1, "prompt_pos": i + 1}
                       for i, t in enumerate(order))
        groups = MAIN_GROUPS + (EXPLORATORY_GROUPS if r == 0 else ())
        for g in groups:
            payload = {"messages": [{"role": "system", "content": SYSTEM_PROMPT_V3},
                                    {"role": "user", "content": main_user_prompt(g, order, sid, top, top_n, day)}],
                       "response_format": rankings_schema(ids, top_n), **DECODING}
            requests.append(_request(day, g, r, payload))
        if r == 0:
            payload = {"messages": [{"role": "system", "content": PROBE_SYSTEM},
                                    {"role": "user", "content": probe_user_prompt(order, sid, top, day)}],
                       "response_format": probe_schema(ids), **(DECODING | {"max_tokens": 1200})}
            requests.append(_request(day, PROBE_GROUP, r, payload))
    return requests, mapping


def time_audit(days: list[str], cal: pd.DatetimeIndex) -> pd.DataFrame:
    """每期：决策日、收益基准日（≤ 决策日的最后一个交易日）、收益结束日（其后第 63 个交易日）、暴露分类."""
    lo, hi = (pd.Timestamp(x) for x in KNOWLEDGE_CUTOFF["conservative_bounds"])
    plo, phi = (pd.Timestamp(x) for x in KNOWLEDGE_CUTOFF["probe_estimate_assumption"])
    rows = []
    for d in days:
        a = pd.Timestamp(d)
        i = cal.searchsorted(a, side="right") - 1
        base = cal[i]
        end = cal[i + HORIZON_TD] if i + HORIZON_TD < len(cal) else pd.NaT

        def cls(c_min, c_max, a=a, end=end):
            if pd.notna(end) and end < c_min:
                return "潜在完整暴露"
            if a > c_max:
                return "截止后"
            return "边界不确定"
        rows.append({"as_of": d, "decision": "as_of 收盘后", "return_base_day": str(base.date()),
                     "return_end_day": str(end.date()) if pd.notna(end) else None,
                     "exposure_conservative": cls(lo, hi), "exposure_if_probe_estimate": cls(plo, phi)})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--panel", type=Path, required=True)
    ap.add_argument("--stock-basic", type=Path, nargs="+", required=True)
    ap.add_argument("--trade-calendar", type=Path, required=True)
    ap.add_argument("--lock", type=Path, required=True, help="工程试跑的 run_*.json：锁定 GGUF 哈希、llama.cpp 版本、聊天模板")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--min-labels", type=int, default=20, help="Top-K 中至少多少只有 63d 标签才纳入该截面")
    ap.add_argument("--seed", type=int, default=20261012)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.out.exists():
        print(f"refusing to overwrite existing output dir: {a.out}")
        return 1

    fixed_text = (SYSTEM_PROMPT_V3 + USER_TEMPLATE_V3 + PROBE_SYSTEM + PROBE_USER
                  + json.dumps(rankings_schema([], a.top_n), ensure_ascii=False)
                  + json.dumps(probe_schema([]), ensure_ascii=False))
    ident = IdentityDictionary(pd.concat([pd.read_parquet(p) for p in a.stock_basic], ignore_index=True),
                               template_text=fixed_text)
    scores = pd.read_csv(a.scores, parse_dates=["as_of"])
    panel = pd.read_parquet(a.panel, columns=FACTORS + [LABEL])
    cal = pd.DatetimeIndex(pd.to_datetime(pd.read_parquet(a.trade_calendar)["cal_date"].astype(str))).sort_values()
    lock_src = json.loads(a.lock.read_text(encoding="utf-8"))
    props = lock_src.get("server_props", {})
    lock = {"gguf_sha256": lock_src["model_sha256"], "gguf_file": lock_src.get("model_file"),
            "llama_cpp_build_info": props.get("build_info"),
            "chat_template_sha256": sha(props.get("chat_template", "")), "source": a.lock.name}
    rng = np.random.default_rng(a.seed)
    requests, mapping, truth, selection = [], [], [], []
    for as_of, g in scores.groupby("as_of"):
        xs = panel.xs(as_of, level="as_of")
        top = (g.set_index("ticker")[["score"]].join(xs, how="inner")
               .sort_values("score", ascending=False, kind="mergesort").head(a.top))
        n_lab = int(top[LABEL].notna().sum())
        if n_lab < a.min_labels:
            selection.append({"as_of": str(as_of.date()), "included": False, "n_candidates": len(top),
                              "n_labeled": n_lab, "reason": f"Top-{a.top} 中有标签的不足 {a.min_labels}"})
            continue
        reqs, mp = build_cross_section(as_of, top, rng, a.top_n)
        for r in reqs:
            if r["group"] in ANON_GROUPS:
                ids = sorted({m["id"] for m in mp if m["repeat"] == r["repeat"]})
                bad = anonymization_violations(r["payload"], ident, ids)
                if bad:
                    raise AssertionError(f"{as_of.date()} group {r['group']} repeat {r['repeat']} leaks: {bad[:5]}")
        requests.extend(reqs)
        mapping.extend(mp)
        truth.extend({"as_of": str(as_of.date()), "ticker": t, LABEL: float(row[LABEL]),
                      **{f: float(row[f]) for f in FACTORS}} for t, row in top.iterrows())
        selection.append({"as_of": str(as_of.date()), "included": True, "n_candidates": len(top),
                          "n_labeled": n_lab, "reason": ""})

    a.out.mkdir(parents=True)

    def jsonl(name: str, rows: list[dict]) -> Path:
        p = a.out / name
        p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        return p

    days = [s["as_of"] for s in selection if s["included"]]
    files = {"requests": jsonl("requests.jsonl", requests), "private_mapping": jsonl("private_mapping.jsonl", mapping)}
    pd.DataFrame(truth).to_csv(a.out / "truth.csv", index=False)
    pd.DataFrame(selection).to_csv(a.out / "selection.csv", index=False)
    audit = time_audit(days, cal)
    audit.to_csv(a.out / "time_audit.csv", index=False)
    files |= {"truth": a.out / "truth.csv", "selection": a.out / "selection.csv", "time_audit": a.out / "time_audit.csv"}
    by_group = pd.Series([r["group"] for r in requests]).value_counts().to_dict()
    manifest = {
        "design_version": DESIGN_VERSION, "positioning": "19 期探索性估计（非确认性实验）",
        "main_groups": list(MAIN_GROUPS), "exploratory_groups": list(EXPLORATORY_GROUPS), "probe_group": PROBE_GROUP,
        "information": {g: {"id": ID_VISIBLE[g], "factors": FACTORS_VISIBLE[g], "date": DATE_VISIBLE[g]}
                        for g in MAIN_GROUPS + EXPLORATORY_GROUPS},
        "anonymous_output_ids_all_groups": True, "row_order": "LGBM 排名（各组相同）", "schema_enum_order": "按编号排序",
        "n_repeats": N_REPEATS, "industry_in_prompt": False,
        "n_cross_sections": len(days), "n_requests": len(requests), "requests_by_group": by_group,
        "included": days, "excluded": [s for s in selection if not s["included"]],
        "top": a.top, "top_n": a.top_n, "seed": a.seed, "factors": FACTORS, "label": LABEL, "horizon_td": HORIZON_TD,
        "decoding": DECODING, "knowledge_cutoff": KNOWLEDGE_CUTOFF,
        "time_audit_counts": {"conservative": audit["exposure_conservative"].value_counts().to_dict(),
                              "if_probe_estimate": audit["exposure_if_probe_estimate"].value_counts().to_dict()},
        "lock": lock,
        "prompt_template_sha256": {"system": sha(SYSTEM_PROMPT_V3), "user": sha(USER_TEMPLATE_V3),
                                   "probe_system": sha(PROBE_SYSTEM), "probe_user": sha(PROBE_USER)},
        "inputs_sha256": {"scores": file_sha(a.scores), "panel": file_sha(a.panel),
                          "trade_calendar": file_sha(a.trade_calendar),
                          **{f"stock_basic:{p.name}": file_sha(p) for p in a.stock_basic}},
        "inputs_path": {"scores": str(a.scores), "panel": str(a.panel)},
        "frozen_sha256": {k: file_sha(p) for k, p in files.items()},
        "anonymization_check": {"groups": list(ANON_GROUPS), "dictionary_codes": len(ident.codes),
                                "template_collisions_excluded": ident.template_collisions,
                                "dictionary_names": len(ident.names), "result": "passed",
                                "scope": "只证明 payload 中没有字面标识，不证明没有身份信息"},
    }
    (a.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("design_version", "n_cross_sections", "n_requests",
                                               "requests_by_group", "time_audit_counts")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
