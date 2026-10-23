"""quantmind.line_a.harness.structured — 让模型的结构化输出一定合法（T-048）.

三层，按顺序：
1. 约束解码：本机后端（Ollama）按 JSON Schema 约束生成，语法和结构在生成时就保证（`LLM.chat(schema=...)`）；
2. 解析与修复：去掉代码块和思考段，取出最外层 JSON；标准解析失败时用 json-repair 修复常见的语法错误
   （缺逗号、多余逗号、未闭合的括号和引号、单引号、注释），并记录"修复过"；
3. 校验与重问：按 pydantic 模型校验。缺的只是编排器已知的字段（例如 ticker、as_of）就补上；
   其他不符合 Schema 的，把具体错误发回给模型重写一次（本机后端这次走约束解码）。
修复只改语法，不改内容：修复后的对象仍要通过 Schema 校验，数值仍要通过代码校验器。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


@dataclass
class ParseInfo:
    repaired: bool = False
    errors: list[str] = field(default_factory=list)


def _outer(text: str) -> str:
    s, e = text.find("{"), text.rfind("}")
    return text[s:e + 1] if s >= 0 and e > s else text


def parse_json(text: str) -> tuple[Any, ParseInfo]:
    """返回 (对象或 None, 信息)。先严格解析，失败再修复；修复不出对象就返回 None。"""
    info = ParseInfo()
    t = _FENCE.sub("", _THINK.sub("", text or "")).strip()
    cand = _outer(t)
    try:
        return json.loads(cand), info
    except json.JSONDecodeError as e:
        info.errors.append(f"json: {e}")
    import json_repair
    obj = json_repair.repair_json(cand, return_objects=True)
    if isinstance(obj, dict | list) and obj != "":
        info.repaired = True
        return obj, info
    return None, info


def validate(obj: Any, model: type[BaseModel], defaults: dict | None = None) -> tuple[dict | None, str | None]:
    if isinstance(obj, dict) and defaults:
        for k, v in defaults.items():
            obj.setdefault(k, v)
    try:
        return model.model_validate(obj).model_dump(mode="json"), None
    except ValidationError as e:
        return None, str(e)[:1500]


def chat_structured(llm, messages: list[dict], model: type[BaseModel], *, thinking: str | None = None,
                    max_tokens: int = 4000, meta: dict | None = None, defaults: dict | None = None,
                    retries: int = 1) -> tuple[dict | None, dict]:
    """不带工具的结构化调用：约束解码（本机）或 JSON 模式（DeepSeek）→ 解析修复 → 校验 → 必要时重问。
    返回 (校验后的对象或 None, 记录：每次的错误、是否修复、费用)。"""
    schema = model.model_json_schema()
    msgs = list(messages)
    rec: dict = {"attempts": [], "cost_rmb": 0.0}
    for attempt in range(retries + 1):
        r = llm.chat(msgs, schema=schema, thinking=thinking, max_tokens=max_tokens,
                     meta={**(meta or {}), "structured_attempt": attempt})
        rec["cost_rmb"] += 0.0 if r.cached else (r.cost_rmb or 0.0)
        obj, info = parse_json(r.content)
        out, err = validate(obj, model, defaults) if obj is not None else (None, "no JSON object in output")
        rec["attempts"].append({"repaired": info.repaired, "parse_errors": info.errors, "schema_error": err,
                                "finish": r.finish_reason})
        if out is not None:
            rec["repaired"] = info.repaired
            return out, rec
        msgs = msgs + [{"role": "assistant", "content": r.content},
                       {"role": "user", "content": "上一条回答不是符合要求的 JSON，错误如下。请只输出修正后的 JSON 对象，"
                                                   "内容不要改，只改格式：\n" + (err or "")}]
    return None, rec
