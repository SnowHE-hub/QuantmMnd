"""T-048：结构化输出层——约束解码请求体、解析修复、Schema 校验与重问、编排器字段补全。不联网。"""
from __future__ import annotations

import json

from pydantic import BaseModel

from quantmind.line_a.harness.llm import LLM
from quantmind.line_a.harness.structured import chat_structured, parse_json, validate


class Out(BaseModel):
    ticker: str
    value: float
    notes: list[str] = []


def test_parse_json_strict_then_repair():
    obj, info = parse_json('思考……```json\n{"ticker": "600001.SH", "value": 1.5}\n```')
    assert obj == {"ticker": "600001.SH", "value": 1.5} and not info.repaired
    obj, info = parse_json('{"ticker": "600001.SH", "value": 2, "notes": ["a", "b",]')     # 多余逗号、缺右括号
    assert obj["notes"] == ["a", "b"] and info.repaired
    obj, info = parse_json("")
    assert obj is None


def test_validate_fills_orchestrator_defaults_only():
    out, err = validate({"value": 3}, Out, {"ticker": "600001.SH"})
    assert out["ticker"] == "600001.SH" and err is None
    out, err = validate({"ticker": "600001.SH", "value": "abc"}, Out)
    assert out is None and "value" in err


def test_chat_structured_retries_with_error_and_local_uses_schema(tmp_path):
    bodies = []
    replies = iter(['{"ticker": "600001.SH", "value": "不是数字"}', '{"ticker": "600001.SH", "value": 4}'])

    def transport(url, headers, body):
        bodies.append(body)
        return 200, {"choices": [{"message": {"content": next(replies)}, "finish_reason": "stop"}],
                     "usage": {"prompt_tokens": 5, "completion_tokens": 5}, "model": "m"}
    llm = LLM(model="qwen3.5:9b-16k", local=True, transport=transport)
    out, rec = chat_structured(llm, [{"role": "user", "content": "输出 JSON"}], Out)
    assert out == {"ticker": "600001.SH", "value": 4.0, "notes": []} and len(rec["attempts"]) == 2
    assert bodies[0]["response_format"]["type"] == "json_schema"                        # 本机：约束解码
    assert "value" in bodies[1]["messages"][-1]["content"]                              # 重问时带上具体错误
    ds = LLM(transport=lambda u, h, b: (bodies.append(b), (200, {"choices": [{"message": {"content": json.dumps(
        {"ticker": "x", "value": 1})}, "finish_reason": "stop"}], "usage": {}, "model": "d"}))[1])
    chat_structured(ds, [{"role": "user", "content": "输出 JSON"}], Out)
    assert bodies[-1]["response_format"] == {"type": "json_object"}                     # DeepSeek：JSON 对象模式


def test_rewrite_accepts_list_sections():
    from quantmind.line_a.harness.rewrite import RewriteOut
    r = RewriteOut.model_validate({"facts": ["第一段", "第二段"], "inferences": None})
    assert r.facts == "第一段\n第二段" and r.inferences == ""
