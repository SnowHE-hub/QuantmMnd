"""quantmind.line_a.harness.rewrite — F9 转写层（03_final_plan_v2 第 5 节 F9）.

三步：
1. 骨架（代码）：把一个案例的声明按状态分栏：
   - 事实：已核实的 reported_fact 与 computed；
   - 推断：已核实或依据可靠的 inference；
   - 反证与限制：counter_evidence，以及 A6 判为 refuted / insufficient 的声明；
   - 待核实：open_question、未通过校验的声明、A6 的 needs_followup；
   - 设计假设：固定文字（参考策略规则、口径说明）。
2. 转写（模型）：只把骨架改写成通顺的中文段落。所有数字必须写成 {{c:<角色>:<local_id>}} 占位符，不能出现字面数字（日期除外），
   不能引用骨架里没有的声明。
3. 核对（代码）：
   - 占位符必须都在骨架里；
   - 除日期外不允许出现裸数字；
   - 每条事实至少被引用一次。
   通过后用声明的数值和单位替换占位符；不通过就退回模板文本，并记录原因。
"""
from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field, field_validator

from quantmind.line_a.harness.llm import LLM

PH = re.compile(r"\{\{c:([A-Z]\d):([A-Za-z0-9_]+)\}\}")
_DATE = re.compile(r"(?:19|20)\d{2}\s*年(?:\s*\d{1,2}\s*月(?:\s*\d{1,2}\s*日)?)?|\d{1,2}\s*月\s*\d{1,2}\s*日|"
                   r"(?:19|20)\d{2}-\d{1,2}-\d{1,2}|(?:19|20)\d{6}|(?:19|20)\d{2}\s*(?:年度|半年度|Q[1-4])|[A-Z]\d")
ASSUMPTIONS = ["参考臂按冻结的原策略配置运行（持有期、调仓日、300 只等权、成本与成交规则），研究声明不改变排名和权重。",
               "所有数据以 as_of 为上限；财报按公告日、公告按发布时间判断是否可用。",
               "风险事实只是事实或待确认项，不默认扣分；本报告不是投资建议。"]


def skeleton(case_outputs: dict[str, dict], review: dict | None = None) -> dict:
    """case_outputs：{角色: run_slice 的结果}；review：A6 的输出（可选）。返回分栏骨架与可引用声明表。"""
    verdict = {j["claim_ref"]: j["verdict"] for j in (review or {}).get("judgments", [])}
    table, cols = {}, {"facts": [], "inferences": [], "counter_and_limits": [], "to_verify": []}
    for role, res in case_outputs.items():
        st = {c["local_id"]: c["status"] for c in res.get("checks", [])}
        for c in (res.get("output") or {}).get("claims", []):
            ref = f"{role}:{c['local_id']}"
            item = {"ref": ref, "type": c["claim_type"], "text": c["text"], "value": c.get("value"), "unit": c.get("unit"),
                    "period": c.get("period"), "check": st.get(c["local_id"]), "a6": verdict.get(ref)}
            table[ref] = item
            ok = item["check"] in ("verified", "minor_mismatch") and item["a6"] not in ("refuted",)
            if item["a6"] in ("refuted", "insufficient") or c["claim_type"] == "counter_evidence":
                cols["counter_and_limits"].append(ref)
            elif c["claim_type"] == "open_question" or not ok or item["a6"] == "needs_followup":
                cols["to_verify"].append(ref)
            elif c["claim_type"] in ("reported_fact", "computed"):
                cols["facts"].append(ref)
            else:
                cols["inferences"].append(ref)
    return {"columns": cols, "claims": table, "assumptions": ASSUMPTIONS}


def _fmt(item: dict) -> str:
    v, u = item.get("value"), item.get("unit") or ""
    if v is None:
        return "（无数值）"
    if u == "元" and abs(v) >= 1e8:
        return f"{v / 1e8:,.2f} 亿元"
    if u == "元" and abs(v) >= 1e4:
        return f"{v / 1e4:,.2f} 万元"
    return f"{v:,.4g}{(' ' + u) if u and u != '%' else u}"


def template_text(sk: dict) -> dict:
    t = sk["claims"]
    names = {"facts": "事实", "inferences": "推断", "counter_and_limits": "反证与限制", "to_verify": "待核实"}
    return {k: "\n".join(f"- {t[r]['text']}（{r}）" for r in refs) or "（无）" for k, refs in sk["columns"].items()} | \
        {"_titles": names}


def check_rewrite(sk: dict, text: dict) -> list[str]:
    errs = []
    used = set()
    for sec, s in text.items():
        if sec.startswith("_"):
            continue
        for m in PH.finditer(s):
            ref = f"{m.group(1)}:{m.group(2)}"
            if ref not in sk["claims"]:
                errs.append(f"{sec}: unknown claim {ref}")
            used.add(ref)
        bare = PH.sub(" ", s)
        bare = _DATE.sub(" ", bare)
        for n in re.findall(r"\d+(?:\.\d+)?", bare):
            errs.append(f"{sec}: bare number {n}")
    for r in sk["columns"]["facts"]:
        if sk["claims"][r]["value"] is not None and r not in used:
            errs.append(f"fact {r} not cited")
    return errs


# ------------------------------------------------------------------ X11 v2：数字以外的绑定（T-044；评测通过前不接入 rewrite）
DIR_UP = re.compile(r"增长|增加|上升|提高|扩大|盈利|扭亏|回升|上涨|增至|增持")
DIR_DN = re.compile(r"下降|减少|下滑|降低|收窄|亏损|首亏|回落|下跌|降至|减至|减持")
SUBJECTS = ("子公司", "控股股东", "实际控制人", "实控人", "参股公司", "母公司")
PLAN = re.compile(r"拟|计划|预案|草案|预计")
DONE = re.compile(r"已(?:经)?(?:完成|实施|回购|减持|增持|披露|发生|支付|到账|转让)|完成|实施完毕")
# v2.1：任何"已……"都表示已发生（"已披露 / 已公告 / 已发布 / 已核实"是报道用语，不算阶段）
DONE_ANY = re.compile(r"已(?!披露|公告|发布|核实|知)经?.|完成|实施完毕")
YEAR = re.compile(r"(?:19|20)\d{2}")
_CLAUSE = re.compile(r"[，,；;。\n]")


def _clause_of(text: str, start: int, end: int) -> str:
    lo = max([m.end() for m in _CLAUSE.finditer(text, 0, start)], default=0)
    m = _CLAUSE.search(text, end)
    return text[lo: m.start() if m else len(text)]


def check_bindings(sk: dict, text: dict) -> list[str]:
    """对每个占位符所在的分句，与声明原文比较方向、主体、阶段、期间；返回错误列表（空 = 通过）。"""
    errs = []
    for sec, s in text.items():
        if sec.startswith("_"):
            continue
        for m in PH.finditer(s):
            ref = f"{m.group(1)}:{m.group(2)}"
            c = sk["claims"].get(ref)
            if c is None:
                continue
            seg, orig = PH.sub(" ", _clause_of(s, m.start(), m.end())), c.get("text") or ""
            up_o, dn_o = bool(DIR_UP.search(orig)), bool(DIR_DN.search(orig))
            for w in SUBJECTS:
                if (w in seg) != (w in orig) and not (w == "实控人" and "实际控制人" in orig):
                    errs.append(f"{sec}: {ref} subject '{w}' {'added' if w in seg else 'dropped'}")
            # v2.1（T-044 登记的补救：只改方向、阶段两类）：分句里出现了原文没有的反向词 / 另一阶段的词就报错，
            # 不再要求分句里只有一种方向（"减持后持股……增持"这类混写在 v2 里漏掉了）
            dirs_o = set(DIR_UP.findall(orig)) | set(DIR_DN.findall(orig))
            if up_o != dn_o:
                opp = DIR_DN if up_o else DIR_UP
                new = [w for w in opp.findall(seg) if w not in dirs_o]
                if new:
                    errs.append(f"{sec}: {ref} direction changed ({','.join(new)})")
            p_o, d_o = bool(PLAN.search(orig)), bool(DONE.search(orig) or DONE_ANY.search(orig))
            plan_new = [w for w in PLAN.findall(seg) if w not in orig]
            done_new = [m.group(0) for m in DONE_ANY.finditer(seg) if m.group(0) not in orig]
            if (done_new and p_o and not d_o) or (plan_new and d_o and not p_o):
                errs.append(f"{sec}: {ref} stage changed ({','.join(done_new + plan_new)})")
            ok_years = set(YEAR.findall(orig)) | ({str(c["period"])[:4]} if c.get("period") else set())
            bad = sorted(set(YEAR.findall(seg)) - ok_years)
            if bad and ok_years:
                errs.append(f"{sec}: {ref} period {','.join(bad)} not in claim")
    return errs


class RewriteOut(BaseModel):
    """转写输出的结构：四栏都是中文段落。模型偶尔把一栏写成段落列表，按顺序换行拼接（内容不变，只统一格式）。"""
    facts: str = ""
    inferences: str = ""
    counter_and_limits: str = ""
    to_verify: str = ""

    @field_validator("facts", "inferences", "counter_and_limits", "to_verify", mode="before")
    @classmethod
    def _join(cls, v):
        if isinstance(v, list):
            return "\n".join(str(x) for x in v)
        return "" if v is None else v


def rewrite_messages(sk: dict) -> list[dict]:
    brief = {k: [{"ref": r, "text": sk["claims"][r]["text"], "has_value": sk["claims"][r]["value"] is not None}
                 for r in refs] for k, refs in sk["columns"].items()}
    return [{"role": "system", "content": "你是研究报告的转写编辑。只改写给定的条目，不添加任何新事实、新判断或买卖建议。"
             "输出 JSON 对象，键为 facts、inferences、counter_and_limits、to_verify，值为中文段落。"
             "凡是数字一律写成占位符 {{c:<角色>:<编号>}}（例如 {{c:A1:c3}}），日期可以照写；不要写任何其他数字。"
             "has_value 为 true 的事实条目必须各引用一次自己的占位符。"},
            {"role": "user", "content": json.dumps(brief, ensure_ascii=False)}]


def fill(sk: dict, text: dict) -> dict:
    return {k: PH.sub(lambda m: _fmt(sk["claims"][f"{m.group(1)}:{m.group(2)}"]), v) for k, v in text.items()
            if not k.startswith("_")}


def rewrite(llm: LLM | None, sk: dict) -> dict:
    """返回 {"mode": "model" | "template", "sections": {...}, "errors": [...]}。"""
    if llm is None:
        return {"mode": "template", "sections": template_text(sk), "errors": ["no model"]}
    from quantmind.line_a.harness.structured import chat_structured
    try:
        text, rec = chat_structured(llm, rewrite_messages(sk), RewriteOut, thinking=None, max_tokens=4000,
                                    meta={"stage": "f9_rewrite"})
    except Exception as e:  # noqa: BLE001  网络或预算问题：退回模板（第 7.5 节）
        return {"mode": "template", "sections": template_text(sk), "errors": [f"model error: {type(e).__name__}"]}
    if text is None:
        return {"mode": "template", "sections": template_text(sk),
                "errors": ["output not valid after repair: " + str(rec["attempts"][-1]["schema_error"])[:300]]}
    errs = check_rewrite(sk, text)
    if errs:
        return {"mode": "template", "sections": template_text(sk), "errors": errs[:20]}
    return {"mode": "model", "sections": fill(sk, text), "errors": []}


# ------------------------------------------------------------------ F9 v2：研究要点（2026-10-23，T-044 / T-040）
# 试跑里整段转写 40 个案例只有 1 个通过核对（裸数字 144 次、引用不存在的声明 25 次）。改成：
# - 四栏事实由代码按模板生成（数值原样，不经过模型）；
# - 模型只写不超过 8 条要点，每条结构化输出，自带引用的声明编号；逐条核对（编号、裸数字、绑定 v2.1）；
# - 不合格的把错误发回修一次，仍不合格的那条单独丢掉并记录，不再整份退回模板。
class Point(BaseModel):
    text: str = Field(description="一句中文要点；数字一律写成占位符 {{c:<角色>:<编号>}}，年份写成“2025年”的形式")
    refs: list[str] = Field(default_factory=list, description="这条要点引用的声明编号，例如 A1:c3")


class Summary(BaseModel):
    points: list[Point] = Field(default_factory=list)


SUMMARY_SYSTEM = ("你是研究报告的编辑。根据给定的声明写不超过 8 条研究要点：先写最重要的风险与反证，再写关键财务和事件事实，"
                  "最后写待核实的问题。只能使用给定的声明，不添加任何新事实、新判断或买卖建议；"
                  "不要改变声明的方向（增 / 减）、主体（公司 / 子公司 / 股东）、阶段（计划 / 已实施）和期间。"
                  "凡是数字一律写成占位符 {{c:<角色>:<编号>}}，年份写成“2025年”这种形式。输出 JSON："
                  '{"points": [{"text": "...", "refs": ["A1:c3", ...]}]}')


def _brief_refs(sk: dict, max_claims: int) -> list[str]:
    cols = sk["columns"]
    order = cols["counter_and_limits"] + [r for r in cols["to_verify"] if sk["claims"][r].get("a6")] + \
        cols["facts"] + cols["inferences"] + cols["to_verify"]
    return list(dict.fromkeys(order))[:max_claims]


def check_point(sk: dict, p: dict) -> list[str]:
    errs = []
    text = p.get("text") or ""
    for m in PH.finditer(text):
        ref = f"{m.group(1)}:{m.group(2)}"
        if ref not in sk["claims"]:
            errs.append(f"unknown claim {ref}")
    for r in p.get("refs") or []:
        if r not in sk["claims"]:
            errs.append(f"unknown ref {r}")
    bare = _DATE.sub(" ", PH.sub(" ", text))
    errs += [f"bare number {n}" for n in re.findall(r"\d+(?:\.\d+)?", bare)]
    errs += check_bindings(sk, {"point": text})
    return errs


def summarize(llm: LLM | None, sk: dict, *, max_points: int = 8, max_claims: int = 40) -> dict:
    """返回 {"points": [填好数值的要点], "dropped": [...], "errors": [...], "mode": "model" | "none"}。"""
    from quantmind.line_a.harness.structured import chat_structured
    if llm is None or not sk["claims"]:
        return {"mode": "none", "points": [], "dropped": [], "errors": ["no model or no claims"]}
    refs = _brief_refs(sk, max_claims)
    brief = [{"ref": r, "type": sk["claims"][r]["type"], "text": sk["claims"][r]["text"],
              "check": sk["claims"][r].get("check"), "a6": sk["claims"][r].get("a6")} for r in refs]
    msgs = [{"role": "system", "content": SUMMARY_SYSTEM}, {"role": "user", "content": json.dumps(brief, ensure_ascii=False)}]
    try:
        out, _ = chat_structured(llm, msgs, Summary, thinking=None, max_tokens=3000, meta={"stage": "f9_summary"})
    except Exception as e:  # noqa: BLE001  网络或预算问题：只给模板
        return {"mode": "none", "points": [], "dropped": [], "errors": [f"model error: {type(e).__name__}"]}
    pts = (out or {}).get("points", [])[:max_points]
    bad = {i: check_point(sk, p) for i, p in enumerate(pts)}
    bad = {i: e for i, e in bad.items() if e}
    if bad:                                              # 把具体错误发回，修一次
        fix = msgs + [{"role": "assistant", "content": json.dumps({"points": pts}, ensure_ascii=False)},
                      {"role": "user", "content": "下面这些要点没有通过核对，请只修改这些条目，其余照抄，输出完整的 JSON：\n"
                       + json.dumps({str(i): e[:6] for i, e in bad.items()}, ensure_ascii=False)}]
        try:
            out2, _ = chat_structured(llm, fix, Summary, thinking=None, max_tokens=3000, meta={"stage": "f9_summary_fix"})
            pts = (out2 or {}).get("points", pts)[:max_points]
        except Exception:  # noqa: BLE001
            pass
    kept, dropped = [], []
    for p in pts:
        e = check_point(sk, p)
        (dropped if e else kept).append(p | ({"errors": e[:6]} if e else {}))
    filled = [PH.sub(lambda m: _fmt(sk["claims"][f"{m.group(1)}:{m.group(2)}"]), p["text"]) +
              "（" + "，".join(p.get("refs") or []) + "）" for p in kept]
    return {"mode": "model", "points": filled, "dropped": dropped, "errors": []}
