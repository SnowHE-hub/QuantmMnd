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


def fill(sk: dict, text: dict) -> dict:
    return {k: PH.sub(lambda m: _fmt(sk["claims"][f"{m.group(1)}:{m.group(2)}"]), v) for k, v in text.items()
            if not k.startswith("_")}


def rewrite(llm: LLM | None, sk: dict) -> dict:
    """返回 {"mode": "model" | "template", "sections": {...}, "errors": [...]}。"""
    if llm is None:
        return {"mode": "template", "sections": template_text(sk), "errors": ["no model"]}
    brief = {k: [{"ref": r, "text": sk["claims"][r]["text"], "has_value": sk["claims"][r]["value"] is not None}
                 for r in refs] for k, refs in sk["columns"].items()}
    msgs = [{"role": "system", "content": "你是研究报告的转写编辑。只改写给定的条目，不添加任何新事实、新判断或买卖建议。"
             "输出 JSON 对象，键为 facts、inferences、counter_and_limits、to_verify，值为中文段落。"
             "凡是数字一律写成占位符 {{c:<角色>:<编号>}}（例如 {{c:A1:c3}}），日期可以照写；不要写任何其他数字。"
             "has_value 为 true 的事实条目必须各引用一次自己的占位符。"},
            {"role": "user", "content": json.dumps(brief, ensure_ascii=False)}]
    try:
        r = llm.chat(msgs, json_mode=True, thinking=None, max_tokens=4000, meta={"stage": "f9_rewrite"})
        text = json.loads(r.content)
    except Exception as e:  # noqa: BLE001  转写失败退回模板（第 7.5 节）
        return {"mode": "template", "sections": template_text(sk), "errors": [f"model error: {type(e).__name__}"]}
    errs = check_rewrite(sk, text)
    if errs:
        return {"mode": "template", "sections": template_text(sk), "errors": errs[:20]}
    return {"mode": "model", "sections": fill(sk, text), "errors": []}
