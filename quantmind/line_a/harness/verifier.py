"""quantmind.line_a.harness.verifier — 独立校验器 F7（03_final_plan_v2 第 5 节 F7；代码实现，不由任何 Agent 兼任）.

逐条核对角色输出的声明，只认本次运行里工具**实际返回过**的证据记录（RunRecord.evidence）：
- reported_fact：
  - 引用的 evidence_id 必须存在；
  - 给了字段和数值时，数值要与记录一致（允许元 / 万元 / 亿元换算，相对误差 ≤ 1e-6）；
  - 给了引文时，引文要逐字出现在记录的文本字段里；
  - 引用的字段是短文字（审计意见、预告类型）时，这段文字要出现在声明正文或引文里（v3，T-051）。
- computed：公式只允许四则运算、abs 和输入声明的编号；用输入声明的数值重算，相对误差 ≤ 1e-4。
- inference / counter_evidence：依据的声明必须存在；依据里有未通过的声明时，标为 weak_basis。
- 正文数字：text 里出现、但在本条 value、引用记录和输入声明里都找不到的数字，记为 unsourced_number。
状态：verified / minor_mismatch（计算值相对误差 ≤ 0.2%）/ mismatch / missing_evidence / unverifiable / weak_basis；
任何声明都不会因为校验通过而自动进入组合。
"""
from __future__ import annotations

import ast
import operator
import re
from dataclasses import dataclass, field

UNIT = {"元": 1.0, "万元": 1e4, "亿元": 1e8, "%": 1.0, "股": 1.0, "万股": 1e4, "亿股": 1e8}
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.USub: operator.neg, ast.UAdd: operator.pos}
_NUM = re.compile(r"-?\d+(?:\.\d+)?")
# 正文数字溯源前先去掉日期、规则编号与证券代码这类"标识"，它们不是需要出处的数值（第一次真实切片里的误报）
_DATE = re.compile(r"(?:19|20)\d{2}\s*年(?:\s*\d{1,2}\s*月(?:\s*\d{1,2}\s*日)?)?|\d{1,2}\s*月\s*\d{1,2}\s*日"
                   r"|(?:19|20)\d{2}-\d{1,2}-\d{1,2}|(?:19|20)\d{6}")
_IDTOK = re.compile(r"[A-Za-z_]+[:\-]?\d+|\d{6}\.(?:SH|SZ|BJ)")


@dataclass
class ClaimCheck:
    local_id: str
    status: str
    notes: list[str] = field(default_factory=list)


def _close(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def _eval(expr: str, env: dict[str, float]) -> float:
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.Name):
            if n.id not in env:
                raise ValueError(f"unknown input {n.id}")
            return env[n.id]
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "abs" and len(n.args) == 1:
            return abs(ev(n.args[0]))
        raise ValueError(f"disallowed expression element {type(n).__name__}")
    return ev(ast.parse(expr, mode="eval"))


MONEY = {"元", "万元", "亿元"}
MINOR_REL = 2e-3          # 计算值相对误差 ≤ 0.2%（截断、多一位舍入）记为 minor_mismatch，与实质性错误分开
_FORMULA_END = re.compile(r"[，,；;。]|即|其中|=")


def _decimals(v: float) -> int:
    s = repr(float(v))
    return 0 if "e" in s or "." not in s else len(s.split(".")[1].rstrip("0"))


def _matches(stated: float, stated_scale: float, truth: float) -> bool:
    """stated × 单位 是否等于 truth，允许按 stated 的小数位四舍五入（"19.01%" 对 19.0119）和 1e-6 的相对误差。"""
    half = 0.5 * 10 ** (-_decimals(stated)) * stated_scale
    a = stated * stated_scale
    return abs(a - truth) <= max(half, 1e-6 * max(1.0, abs(a), abs(truth)))


def _clean_formula(f: str) -> str:
    """只保留运算部分：去掉"，即 …"、"其中 …"、"= …"这类解释（第二次真实切片里常见）。"""
    m = _FORMULA_END.search(f)
    return (f[:m.start()] if m else f).strip()


def _value_in_quote(v: float, unit: str | None, quote: str) -> bool:
    """引文里的数字（去掉千分位和数字间的空格）是否有一个等于声明值；允许引文与声明用不同的万 / 亿单位。"""
    q = re.sub(r"(?<=\d)[,，\s](?=\d)", "", quote)
    nums = [float(x) for x in _NUM.findall(q)]
    for n in nums:
        for s in (1.0, 1e4, 1e8, 1e-4, 1e-8):
            if _matches(v, s, n):
                return True
    # PDF 表格的单元格常被拼在一起（"370,0000.95%"），拆不出数来（v3.1）：数值的常见写法作为子串出现、且前面不是数字，也算找到
    raw = _norm(quote)
    for s in (1.0, 1e4, 1e8):
        x = v / s
        forms = {f"{x:,.0f}", f"{x:.0f}"} if abs(x - round(x)) < 1e-9 else {f"{x:,.2f}", f"{x:.2f}", f"{x:g}"}
        for fm in forms:
            if re.search(rf"(?<![\d.]){re.escape(fm)}", raw):
                return True
    return False


def _record_numbers(rec: dict) -> list[float]:
    out = []
    for v in rec.values():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out.append(float(v))
        elif isinstance(v, str):
            out += [float(x) for x in _NUM.findall(v)]
    return out


def _norm(t: str) -> str:
    """引文比对的规范化：全角转半角（NFKC）、去掉空白，统一引号。"""
    import unicodedata
    t = unicodedata.normalize("NFKC", t)
    return re.sub(r"\s+", "", t).replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")


def quote_in(q: str, text: str) -> bool:
    """引文要逐字出现在记录文本里；引文里有省略号时，拆成几段，要求各段按顺序出现（v2，T-050）。"""
    nt = _norm(text)
    parts = [x for x in re.split(r"……|…|\.{3,}", _norm(q)) if x]
    pos = 0
    for x in parts:
        i = nt.find(x, pos)
        if i < 0:
            return False
        pos = i + len(x)
    return bool(parts)


# 事件链 stage / state 的英文代码与中文写法（v3.1，A2 跟进）：声明里写"预案""实施"就算写出了这个状态
ENUM_ALIASES = {
    "plan": ("预案", "计划", "拟", "草案", "方案", "预披露"), "progress": ("进展", "进度", "延期", "调整"),
    "implemented": ("实施", "开始", "首次"), "completed": ("完成", "结束", "届满", "完毕"),
    "terminated": ("终止", "取消", "放弃"), "withdrawn": ("撤回", "撤销"), "lifted": ("解除", "撤销", "恢复", "复牌"),
    "correction": ("更正", "补充", "修订"),
    "in": ("增持", "增加", "买入"), "de": ("减持", "减少", "卖出"),
}
ENUM_SKIP = {"single", "open", "closed", "unknown"}       # 没有阶段含义的代码：不要求正文写出


def _enum_stated(raw: str, hay: str) -> bool | None:
    """raw 是事件链的英文代码时，返回正文里是否写出了它（None 表示不是这类代码）。"""
    k = raw.strip().lower()
    if k in ENUM_SKIP or (k in ENUM_ALIASES and k in hay.lower()):     # 正文直接写了代码本身也算
        return True
    if k in ENUM_ALIASES:
        return any(a in hay for a in ENUM_ALIASES[k])
    parts = [p for p in re.split(r"[\\/、,，;；|]+", raw) if p.strip()]
    if len(parts) > 1:                                                  # 多值字段（"总经理\总裁"）：写出其中之一即可
        return any(_norm(p) in hay for p in parts)
    return None


def _doc_records(evidence: dict[str, dict], eid: str) -> list[tuple[str, dict]]:
    """同一份公告的正文段落记录（cninfo:ann:<编号> 或 doc:<编号>:… → doc:<编号>:…）。"""
    m = re.match(r"(?:cninfo:ann:|doc:)(\d+)", eid)
    if not m:
        return []
    pre = f"doc:{m.group(1)}:"
    return [(k, v) for k, v in evidence.items() if k.startswith(pre) and k != eid]


# 分类型字段：取值必须原样写出（"保留意见"和"无保留意见"只差两个字，不能放宽）；其他文字字段允许简写（v3.1）
CATEGORICAL_FIELDS = {"audit_result", "type", "stage", "state", "in_de", "div_proc", "change_type"}


def _common_run(a: str, b: str) -> int:
    """a 在 b 里能找到的最长连续片段长度。"""
    best = 0
    for i in range(len(a)):
        for j in range(i + best + 1, len(a) + 1):
            if a[i:j] in b:
                best = j - i
            else:
                break
    return best


def _is_text_value(raw: str) -> bool:
    """短的文字取值（"标准无保留意见"、"预增"）；纯数字串、日期时间（含时区）和长段落不算。"""
    t = raw.strip()
    return 0 < len(t) <= 30 and not re.fullmatch(r"[\d.\-:T +]+|\d{6}\.(?:SH|SZ|BJ)", t)


def cu_of(r: dict, c: dict, own) -> str | None:
    return r.get("unit") if own is not None else c.get("unit")


def verify(output: dict, evidence: dict[str, dict]) -> list[ClaimCheck]:
    claims = {c["local_id"]: c for c in output.get("claims", [])}
    checks: dict[str, ClaimCheck] = {}
    for cid, c in claims.items():
        ck = ClaimCheck(cid, "verified")
        t = c["claim_type"]
        refs = c.get("evidence") or []
        if t == "reported_fact":
            if not refs:
                ck.status = "missing_evidence"
                ck.notes.append("reported_fact without evidence")
            claim_v, claim_hits, claim_tried = c.get("value"), 0, 0
            for r in refs:
                rec = evidence.get(r["evidence_id"])
                if rec is None:
                    ck.status = "missing_evidence"
                    ck.notes.append(f"evidence {r['evidence_id']} was not returned by any tool in this run")
                    continue
                q = r.get("quote")
                quote_ok = None
                if q:
                    texts = "".join(str(x) for x in rec.values() if isinstance(x, str))
                    quote_ok = quote_in(q, texts)
                    if not quote_ok:
                        # 引用编号错位（v3.1，A2 跟进）：引文不在所引记录里，但逐字出现在同一份公告的正文段落里 →
                        # 由代码改正引用，记备注；引文本身不对（例如跳着摘表格）仍判不一致
                        alt = next((k for k, v in _doc_records(evidence, r["evidence_id"])
                                    if quote_in(q, "".join(str(x) for x in v.values() if isinstance(x, str)))), None)
                        if alt:
                            quote_ok = True
                            ck.notes.append(f"citation_corrected {r['evidence_id']} -> {alt}")
                        else:
                            ck.status = "mismatch"
                            ck.notes.append("quote not found in the cited record")
                f = r.get("field")
                own = r.get("value")                     # 多值声明：每条证据自己的数值
                v = own if own is not None else claim_v
                raw = rec.get(f) if f else None
                numeric = isinstance(raw, (int, float)) and not isinstance(raw, bool)
                if f and isinstance(raw, str) and _is_text_value(raw):
                    # 文字字段（v3，T-051）：审计意见、预告类型这类字段，记录里的文字要出现在声明正文或引文里；
                    # 不看 value（模型常写 1.0 占位）
                    hay = _norm(c.get("text", "") + (q or ""))
                    stated = _enum_stated(raw, hay)
                    if stated is None and f not in CATEGORICAL_FIELDS:
                        stated = _common_run(_norm(raw), hay) >= min(len(_norm(raw)), 4)   # 描述性文字：允许简写
                    if stated is False or (stated is None and _norm(raw) not in hay):
                        ck.status = "mismatch"
                        ck.notes.append(f"text field {f}={raw!r} not stated in claim text or quote")
                    elif own is None:
                        claim_tried += 1
                        claim_hits += 1
                    continue
                if v is not None and not numeric and q and quote_ok:
                    # 文档证据：数值从已核对的引文里找（千分位、空格、单位写法不同都允许）
                    ok = _value_in_quote(float(v), r.get("unit") if own is not None else c.get("unit"), q)
                    if own is not None and not ok:
                        ck.status = "mismatch"
                        ck.notes.append(f"value {v} not found in the quote")
                    if own is None:
                        claim_tried += 1
                        claim_hits += ok
                    continue
                if not (f and v is not None):
                    continue
                if not numeric:
                    # 文档段落没有这个字段（字段名是模型自己起的）：到记录的原文里找这个数（v2，T-050）
                    body = "".join(str(x) for x in rec.values() if isinstance(x, str))
                    if body and f not in rec:
                        ok = _value_in_quote(float(v), cu_of(r, c, own), body)
                        if own is not None and not ok:
                            ck.status = "mismatch"
                            ck.notes.append(f"value {v} not found in the cited record text")
                        if own is None:
                            claim_tried += 1
                            claim_hits += ok
                        continue
                    ck.status = "unverifiable"
                    ck.notes.append(f"field {f} not numeric in record")
                    continue
                cu = r.get("unit") if own is not None and r.get("unit") else (c.get("unit") or "")
                ru = (rec.get("_units") or {}).get(f, "元" if cu in MONEY else cu)
                if own is not None and not r.get("unit"):
                    # 多值声明里逐条给的数值没写单位：按该字段自己的单位理解，不套用声明的单位（v3.1；例如 12 家机构不是 12 万元）
                    cu = ru
                ok = _matches(float(v), UNIT.get(cu, 1.0) / UNIT.get(ru, 1.0), float(raw))
                if own is not None and not ok:
                    ck.status = "mismatch"
                    ck.notes.append(f"value {v} {cu} != record {f}={raw} {ru}")
                if own is None:
                    claim_tried += 1
                    claim_hits += ok
            if claim_tried and not claim_hits and ck.status == "verified":
                ck.status = "mismatch"
                ck.notes.append(f"value {claim_v} {c.get('unit')} matches none of the cited fields")
        elif t == "computed":
            env = {i: claims[i].get("value") for i in c.get("inputs", []) if i in claims}
            if not c.get("formula") or any(v is None for v in env.values()) or len(env) != len(c.get("inputs", [])):
                ck.status = "unverifiable"
                ck.notes.append("formula or numeric inputs missing")
            else:
                try:
                    val = _eval(_clean_formula(c["formula"]), {k: float(v) for k, v in env.items()})
                    if c.get("value") is None or not _matches(float(c["value"]), 1.0, val):
                        rel = abs(float(c["value"]) - val) / max(abs(val), 1e-12) if c.get("value") is not None else 1
                        ck.status = "minor_mismatch" if rel <= MINOR_REL else "mismatch"
                        ck.notes.append(f"recomputed {val:.6g} != stated {c.get('value')} (rel {rel:.2g})")
                except (ValueError, ZeroDivisionError, SyntaxError) as e:
                    ck.status = "unverifiable"
                    ck.notes.append(str(e))
        elif t in ("inference", "counter_evidence"):
            missing = [i for i in c.get("inputs", []) if i not in claims]
            if missing or not (c.get("inputs") or refs):
                ck.status = "unverifiable"
                ck.notes.append(f"basis missing: {missing or 'no inputs or evidence'}")
        checks[cid] = ck
    # 依据链：推断依据的声明未通过 → weak_basis；正文数字溯源
    for cid, c in claims.items():
        ck = checks[cid]
        if c["claim_type"] in ("inference", "counter_evidence") and ck.status == "verified" and \
                any(checks[i].status not in ("verified", "minor_mismatch") for i in c.get("inputs", []) if i in checks):
            ck.status = "weak_basis"
        allowed = set()
        if c.get("value") is not None:
            allowed.add(float(c["value"]))
        for r in c.get("evidence") or []:
            if r["evidence_id"] in evidence:
                allowed.update(_record_numbers(evidence[r["evidence_id"]]))
        for i in c.get("inputs", []):
            if i in claims and claims[i].get("value") is not None:
                allowed.add(float(claims[i]["value"]))
        ids = " ".join(r["evidence_id"] for r in c.get("evidence") or [])
        body = _IDTOK.sub(" ", _DATE.sub(" ", c.get("text", "")))
        body = re.sub(r"(?<=\d)[,，](?=\d{3})", "", body)          # 千分位：先合并成一个数（v2，T-050；之前被拆成几段，几乎全是误报）
        # 数量级错误（v2，T-050）：正文里没有任何地方正确写出本条的数值，却有一个数与它只差 10 的整数次方
        # （例如 24.69 亿写成 246.86 亿）。正确的数值也在正文里时不判，避免同一句里碰巧相差 10 倍的两个数误报
        cv = c.get("value")
        nums = [float(x) for x in _NUM.findall(body)]
        if cv is not None and float(cv) != 0 and c["claim_type"] in ("reported_fact", "computed") and nums:
            ok0 = any(_close(xv * s, float(cv), 2e-3) for xv in nums for s in (1, 1e4, 1e8, 0.01))
            bad = [xv for xv in nums if abs(xv) >= 1 and any(_close(xv * s * 10 ** k, float(cv), 2e-3)
                                                              for s in (1, 1e4, 1e8) for k in (-3, -2, -1, 1, 2, 3))]
            if not ok0 and bad:
                ck.status = "mismatch"
                ck.notes.append(f"magnitude error: text {bad[0]:g} vs value {cv}")
        for x in _NUM.findall(body):
            if x in ids:
                continue
            xv = float(x)
            is_date = re.fullmatch(r"(19|20)\d{2}(\d{4})?", x)          # 年份、报告期这类日期数字不算
            if abs(xv) >= 10 and not is_date and xv not in allowed and \
                    not any(_close(xv, a / s, 1e-3) for a in allowed for s in (1, 1e4, 1e8, 0.01)):
                ck.notes.append(f"unsourced_number {x}")
    return list(checks.values())
