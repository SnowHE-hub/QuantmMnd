"""T-007 执行端（设计 v3）：只发冻结的 payload、run_id 绑定、断点续跑不混入其他运行、传输重试上限、锁定值检查（假 HTTP 层）."""
from __future__ import annotations

import json

import pytest

import scripts.ablation_run as ar


class FakeResp:
    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self.body


def _write_requests(d, n=3):
    rows = []
    for i in range(n):
        payload = {"messages": [{"role": "user", "content": f"x{i}"}], "response_format": {"type": "json_schema"},
                   "temperature": 0, "seed": 0, "max_tokens": 10, "chat_template_kwargs": {"enable_thinking": False}}
        rows.append({"request_id": ar.payload_sha(payload), "as_of": f"2023-0{i + 1}-31", "group": "A",
                     "payload": payload})
    (d / "requests.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return rows


@pytest.fixture()
def http(monkeypatch):
    sent = []

    def post(url, json, timeout):
        sent.append((url, json))
        return FakeResp({"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]})

    monkeypatch.setattr(ar.requests, "post", post)
    monkeypatch.setattr(ar.requests, "get", lambda url, timeout: FakeResp({"model_path": "x.gguf"}))
    return sent


def test_sends_exactly_the_frozen_payload(tmp_path, http):
    rows = _write_requests(tmp_path, 1)
    assert ar.main(["--dir", str(tmp_path), "--base-url", "http://h:8090/", "--model", "m"]) == 0
    url, body = http[0]
    assert url == "http://h:8090/v1/chat/completions"
    assert body == {"model": "m", **rows[0]["payload"]}
    rec = json.loads((tmp_path / "responses.jsonl").read_text(encoding="utf-8"))
    assert rec["status"] == "ok" and rec["request_id"] == rows[0]["request_id"] and len(rec["run_id"]) == 64
    assert list(tmp_path.glob("run_*.json"))


def test_tampered_request_is_refused(tmp_path, http):
    rows = _write_requests(tmp_path, 1)
    rows[0]["payload"]["temperature"] = 1
    (tmp_path / "requests.jsonl").write_text(json.dumps(rows[0]), encoding="utf-8")
    with pytest.raises(SystemExit, match="request_id"):
        ar.main(["--dir", str(tmp_path), "--base-url", "http://h", "--model", "m"])


def test_resume_is_scoped_to_the_run(tmp_path, http):
    _write_requests(tmp_path, 3)
    ar.main(["--dir", str(tmp_path), "--base-url", "http://h", "--model", "m"])
    ar.main(["--dir", str(tmp_path), "--base-url", "http://h", "--model", "m"])
    assert len(http) == 3                                           # 同一 run：不重发
    ar.main(["--dir", str(tmp_path), "--base-url", "http://h", "--model", "other"])
    assert len(http) == 6                                           # 换模型 = 新 run：全部重发，不混用
    runs = {json.loads(x)["run_id"] for x in (tmp_path / "responses.jsonl").read_text(encoding="utf-8").splitlines()}
    assert len(runs) == 2


def test_transport_errors_are_retried_in_run_and_recorded(tmp_path, monkeypatch):
    _write_requests(tmp_path, 2)
    calls = []

    def flaky(url, json, timeout):
        calls.append(1)
        if len(calls) == 1:
            raise TimeoutError("slow")
        return FakeResp({"choices": [{"message": {"content": "{}"}}]})

    monkeypatch.setattr(ar.requests, "post", flaky)
    monkeypatch.setattr(ar.requests, "get", lambda url, timeout: FakeResp({}))
    monkeypatch.setattr(ar.time, "sleep", lambda s: None)
    assert ar.main(["--dir", str(tmp_path), "--base-url", "http://h", "--model", "m"]) == 0
    recs = [json.loads(x) for x in (tmp_path / "responses.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["status"] for r in recs] == ["error", "ok", "ok"]
    assert [r["attempt"] for r in recs] == [1, 2, 1]


def test_transport_attempts_are_capped_across_invocations(tmp_path, monkeypatch):
    _write_requests(tmp_path, 1)
    calls = []

    def down(url, json, timeout):
        calls.append(1)
        raise ConnectionError("refused")

    monkeypatch.setattr(ar.requests, "post", down)
    monkeypatch.setattr(ar.requests, "get", lambda url, timeout: FakeResp({}))
    monkeypatch.setattr(ar.time, "sleep", lambda s: None)
    assert ar.main(["--dir", str(tmp_path), "--base-url", "http://h", "--model", "m"]) == 2
    assert len(calls) == ar.MAX_TRANSPORT_ATTEMPTS
    assert ar.main(["--dir", str(tmp_path), "--base-url", "http://h", "--model", "m"]) == 2
    assert len(calls) == ar.MAX_TRANSPORT_ATTEMPTS                    # 用尽后不再重试


def test_lock_mismatch_refuses_to_run(tmp_path, http, monkeypatch):
    _write_requests(tmp_path, 1)
    (tmp_path / "manifest.json").write_text(json.dumps({"lock": {"llama_cpp_build_info": "b1-9a9394a"}}),
                                            encoding="utf-8")
    monkeypatch.setattr(ar.requests, "get", lambda url, timeout: FakeResp({"build_info": "b2-other"}))
    with pytest.raises(SystemExit, match="llama_cpp_build_info"):
        ar.main(["--dir", str(tmp_path), "--base-url", "http://h", "--model", "m"])
    assert http == []


def test_lock_match_runs(tmp_path, http, monkeypatch):
    _write_requests(tmp_path, 1)
    tpl = "{{ template }}"
    lock = {"llama_cpp_build_info": "b1-9a9394a",
            "chat_template_sha256": __import__("hashlib").sha256(tpl.encode()).hexdigest()}
    (tmp_path / "manifest.json").write_text(json.dumps({"lock": lock}), encoding="utf-8")
    monkeypatch.setattr(ar.requests, "get",
                        lambda url, timeout: FakeResp({"build_info": "b1-9a9394a", "chat_template": tpl}))
    assert ar.main(["--dir", str(tmp_path), "--base-url", "http://h", "--model", "m"]) == 0
    assert len(http) == 1
