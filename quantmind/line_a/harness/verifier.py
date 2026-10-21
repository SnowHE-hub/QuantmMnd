"""quantmind.line_a.harness.verifier — 独立校验器 F7（03_final_plan_v2 第 5 节 F7；代码实现，不由任何 Agent 兼任）.

逐条核对角色输出的声明，只认本次运行里工具**实际返回过**的证据记录（RunRecord.evidence）：
- reported_fact：
  - 引用的 evidence_id 必须存在；
  - 给了字段和数值时，数值要与记录一致（允许元 / 万元 / 亿元换算，相对误差 ≤ 1e-6）；
  - 给了引文时，引文要逐字出现在记录的文本字段里。
- computed：公式只允许四则运算、abs 和输入声明的编号；用输入声明的数值重算，相对误差 ≤ 1e-4。
- inference / counter_evidence：依据的声明必须存在；依据里有未通过的声明时，标为 weak_basis。
- 正文数字：text 里出现、但在本条 value、引用记录和输入声明里都找不到的数字，记为 unsourced_number。
状态：verified / mismatch / missing_evidence / unverifiable / weak_basis；任何声明都不会因为校验通过而自动进入组合。
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


def _record_numbers(rec: dict) -> list[float]:
    out = []
    for v in rec.values():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out.append(float(v))
        elif isinstance(v, str):
            out += [float(x) for x in _NUM.findall(v)]
    return out


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
            for r in refs:
                rec = evidence.get(r["evidence_id"])
                if rec is None:
                    ck.status = "missing_evidence"
                    ck.notes.append(f"evidence {r['evidence_id']} was not returned by any tool in this run")
                    continue
                f, v = r.get("field"), c.get("value")
                if f and v is not None:
                    raw = rec.get(f)
                    scale = UNIT.get(c.get("unit") or "元", 1.0)
                    if not isinstance(raw, (int, float)):
                        ck.status = "unverifiable"
                        ck.notes.append(f"field {f} not numeric in record")
                    elif not _close(float(raw), float(v) * scale, 1e-6):
                        ck.status = "mismatch"
                        ck.notes.append(f"value {v} {c.get('unit')} != record {f}={raw}")
                q = r.get("quote")
                if q:
                    texts = "".join(str(x) for x in rec.values() if isinstance(x, str))
                    if re.sub(r"\s+", "", q) not in re.sub(r"\s+", "", texts):
                        ck.status = "mismatch"
                        ck.notes.append("quote not found in the cited record")
        elif t == "computed":
            env = {i: claims[i].get("value") for i in c.get("inputs", []) if i in claims}
            if not c.get("formula") or any(v is None for v in env.values()) or len(env) != len(c.get("inputs", [])):
                ck.status = "unverifiable"
                ck.notes.append("formula or numeric inputs missing")
            else:
                try:
                    val = _eval(c["formula"], {k: float(v) for k, v in env.items()})
                    if c.get("value") is None or not _close(val, float(c["value"]), 1e-4):
                        ck.status = "mismatch"
                        ck.notes.append(f"recomputed {val:.6g} != stated {c.get('value')}")
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
                any(checks[i].status != "verified" for i in c.get("inputs", []) if i in checks):
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
        for x in _NUM.findall(body):
            if x in ids:
                continue
            xv = float(x)
            is_date = re.fullmatch(r"(19|20)\d{2}(\d{4})?", x)          # 年份、报告期这类日期数字不算
            if abs(xv) >= 10 and not is_date and xv not in allowed and \
                    not any(_close(xv, a / s, 1e-3) for a in allowed for s in (1, 1e4, 1e8, 0.01)):
                ck.notes.append(f"unsourced_number {x}")
    return list(checks.values())
