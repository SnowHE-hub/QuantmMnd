"""quantmind.line_a.harness.llm — 大模型调用层（03_final_plan_v2 第 8.1、8.4、9 节）.

先接口、后实现：上层（工具循环、六个角色）只依赖 `LLM.chat()` 和 `LLMResult`，换提供方不改上层。
第一版实现 DeepSeek（OpenAI 兼容接口，`https://api.deepseek.com/chat/completions`）。

安全与可复现：
- **密钥**：运行时从环境变量读取；WSL 里读不到时，通过 Windows 互操作读取 Windows 的系统环境变量。
  只放在内存里，不打印、不写日志、不进任何产物（`LLMResult`、账本、缓存都不含密钥和请求头）。
- **只发公开材料**：发送前检查消息内容，含账户规模、持仓、资金账户等敏感字样就拒绝发送（`PrivateContentError`）。
- **账本**：每次请求记录模型、请求哈希、token 用量、估算费用、耗时、结果哈希（JSONL，只追加）。
- **预算**：累计估算费用超过上限就拒绝新请求（`BudgetExceededError`）。
- **缓存**：同一请求哈希（模型 + 消息 + 参数 + 工具 + Schema）已有成功结果时直接复用，重放和断点恢复不重复付费。
- **网络**：始终校验证书。证书校验失败（例如校园网防火墙解密 HTTPS，2026-10-21 在悉尼大学网络上遇到）时，
  抛出 TLSInterceptionError 并停止。不关闭校验，不安装第三方根证书，否则密钥会暴露给中间设备。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

BJ = timezone(timedelta(hours=8))
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
# 本机后端（X7，T-041 / T-048）：WSL 里的 llama.cpp llama-server（OpenAI 兼容接口），加载 Qwen3.5-9B 的 GGUF；
# 不需要密钥，不出本机，费用记 0。不用 Ollama：Ollama 0.30.6 跑 qwen3.5 时不执行 JSON Schema 约束（2026-10-23 实测）
LOCAL_URL = os.environ.get("QM_LOCAL_LLM_URL", "http://127.0.0.1:8091/v1/chat/completions")
# 元 / 百万 tokens：(缓存命中输入, 未命中输入, 输出)；来源 api-docs.deepseek.com 价格页（2026-10-21 查阅）
PRICES = {"deepseek-flash": {"idle": (0.02, 1.0, 4.0), "peak": (0.04, 2.0, 8.0)},
          "deepseek-v4-pro": {"idle": (0.15, 4.5, 13.5), "peak": (0.30, 9.0, 27.0)}}
# 兜底检查（结构上 Harness 本来就不把账户信息放进提示词）；只列不会出现在公开公告里的说法，避免误拦。
# 2026-10-23（T-048）：去掉"账户资金""资金账户""持仓市值""可用资金"——上市公司公告里常见（募集资金账户、基金持仓市值），
# 试跑中把两份公开公告误拦了；保留第一人称和专用标记
PRIVATE_PATTERNS = (r"账户规模", r"我的持仓", r"真实持仓", r"我的账户", r"本人账户",
                    r"account[_ ]?size", r"my holdings", r"\[PRIVATE\]")


class PrivateContentError(ValueError):
    pass


class BudgetExceededError(RuntimeError):
    pass


class TLSInterceptionError(RuntimeError):
    """证书校验失败（例如校园网防火墙在解密 HTTPS）：不发送请求、不重试，交给上层停止整个案例。"""


@dataclass
class LLMResult:
    content: str
    reasoning: str
    tool_calls: list[dict]
    finish_reason: str
    model: str
    model_version: str          # 服务端返回的模型标识（API 滚动版本要记录，跨月不声称是同一个模型）
    usage: dict
    cost_rmb: float
    latency_s: float
    request_hash: str
    cached: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def _sha(b: bytes | str) -> str:
    return hashlib.sha256(b.encode("utf-8") if isinstance(b, str) else b).hexdigest()


def canonical(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def is_peak(now: datetime | None = None) -> bool:
    """北京时间周一至周五 9:00–12:00、14:00–18:00 为高峰（法定节假日按空闲处理需要日历，第一版不区分，偏保守算高峰）。"""
    t = (now or datetime.now(BJ)).astimezone(BJ)
    if t.weekday() >= 5:
        return False
    m = t.hour * 60 + t.minute
    return 540 <= m < 720 or 840 <= m < 1080


def estimate_cost(model: str, usage: dict, now: datetime | None = None) -> float:
    p = PRICES.get(model)
    if p is None:
        return float("nan")
    hit, miss, out = p["peak" if is_peak(now) else "idle"]
    h = usage.get("prompt_cache_hit_tokens", 0)
    m = usage.get("prompt_cache_miss_tokens", max(0, usage.get("prompt_tokens", 0) - h))
    return (h * hit + m * miss + usage.get("completion_tokens", 0) * out) / 1e6


def assert_public(messages: list[dict]) -> None:
    for msg in messages:
        text = msg.get("content") or ""
        if not isinstance(text, str):
            text = canonical(text)
        for pat in PRIVATE_PATTERNS:
            if re.search(pat, text, flags=re.IGNORECASE):
                raise PrivateContentError(f"message contains a private-content marker ({pat}); not sent")


_KEY_CACHE: dict[str, str] = {}
_KEY_LOCK = threading.Lock()


def read_key(name: str = "DEEPSEEK_API_KEY") -> str:
    """环境变量优先；WSL 里没有时，从 Windows 的系统（Machine）与用户环境变量读取。返回值只留在内存里。
    进程内缓存，多线程共用一次读取；PowerShell 偶尔启动慢或返回空（6 路并发时见过），最多重试 3 次。"""
    v = os.environ.get(name, "").strip()
    if v:
        return v
    with _KEY_LOCK:
        if name in _KEY_CACHE:
            return _KEY_CACHE[name]
        ps = "powershell.exe" if os.name == "nt" else "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
        if os.name == "nt" or Path(ps).exists():
            for _ in range(3):
                for scope in ("Machine", "User"):
                    try:
                        r = subprocess.run([ps, "-NoProfile", "-NonInteractive", "-Command",
                                            f"[Environment]::GetEnvironmentVariable('{name}','{scope}')"],
                                           capture_output=True, text=True, timeout=30)
                    except subprocess.TimeoutExpired:
                        continue
                    v = r.stdout.strip()
                    if v:
                        _KEY_CACHE[name] = v
                        return v
    raise RuntimeError(f"{name} is not set")


@dataclass
class LLM:
    model: str = "deepseek-flash"
    ledger: Path | None = None              # JSONL 账本
    cache_dir: Path | None = None           # 请求哈希 → 结果
    budget_rmb: float = 20.0                # 本对象生命周期内的估算费用上限
    timeout_s: float = 300.0
    max_retries: int = 4
    transport: Any = None                   # 测试注入：callable(url, headers, body) -> (status, json)
    local: bool = False                     # True：本机 llama-server（LOCAL_URL），不读密钥，费用记 0
    spent_rmb: float = 0.0
    _key: str | None = field(default=None, repr=False)

    def _post(self, body: dict) -> tuple[int, dict]:
        if self.transport is not None:
            return self.transport(LOCAL_URL if self.local else DEEPSEEK_URL, {}, body)
        import httpx
        if self.local:
            r = httpx.post(LOCAL_URL, json=body, timeout=self.timeout_s)
            try:
                return r.status_code, r.json()
            except ValueError:
                return r.status_code, {"error": {"message": r.text[:500]}}
        if self._key is None:
            self._key = read_key()
        try:
            r = httpx.post(DEEPSEEK_URL, json=body, timeout=self.timeout_s,
                           headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"})
        except httpx.ConnectError as e:
            # 证书校验失败发生在 TLS 握手阶段，请求头（含密钥）还没有发出。绝不关闭校验、不重试。
            if "CERTIFICATE_VERIFY_FAILED" in str(e):
                raise TLSInterceptionError(
                    "TLS certificate verification failed for api.deepseek.com - the network may be decrypting HTTPS "
                    "(e.g. a campus firewall). The API key was not sent. Run again on a network without interception.") from None
            raise
        try:
            return r.status_code, r.json()
        except ValueError:
            return r.status_code, {"error": {"message": r.text[:500]}}

    def chat(self, messages: list[dict], *, tools: list[dict] | None = None, json_mode: bool = False,
             thinking: str | None = None, max_tokens: int = 4096, meta: dict | None = None,
             schema: dict | None = None) -> LLMResult:
        """thinking：None = 关闭思考；"low" / "high" / "max" = 开启并设强度。
        schema：本机后端按 JSON Schema 约束解码（输出一定是合法 JSON 且符合结构）；DeepSeek 只支持 JSON 对象模式，
        schema 在那里只作为"要求 JSON 输出"，结构由调用方校验（T-048）。"""
        assert_public(messages)
        body: dict = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "stream": False}
        if tools:
            body["tools"] = tools
        if schema is not None and self.local:
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "output", "schema": schema,
                                                                               "strict": True}}
        elif json_mode or schema is not None:
            body["response_format"] = {"type": "json_object"}
        if self.local:                     # llama-server：Qwen3.5 的思考开关走聊天模板参数
            body["chat_template_kwargs"] = {"enable_thinking": bool(thinking)}
        elif thinking:
            body["thinking"] = {"type": "enabled"}
            body["reasoning_effort"] = thinking
        else:
            body["thinking"] = {"type": "disabled"}
        rh = _sha(canonical(body))
        cpath = self.cache_dir / f"{rh}.json" if self.cache_dir else None
        if cpath is not None and cpath.exists():
            res = LLMResult(**json.loads(cpath.read_text(encoding="utf-8")))
            res.cached = True
            self._log(res, meta, 200)
            return res
        if self.spent_rmb >= self.budget_rmb:
            raise BudgetExceededError(f"spent {self.spent_rmb:.4f} >= budget {self.budget_rmb}")
        t0, status, data = time.time(), 0, {}
        net_tries, attempt = 0, 0
        while True:
            try:
                status, data = self._post(body)
            except TLSInterceptionError:
                raise                                   # 证书校验失败：密钥没有发出，不重试
            except Exception as e:  # noqa: BLE001  手机热点偶尔断一下：连接被重置、读超时、对端断开，退避后重试
                import httpx
                if not isinstance(e, httpx.TransportError) or net_tries >= 6:
                    raise
                net_tries += 1
                time.sleep(min(120, 5 * 2 ** (net_tries - 1)))
                continue
            if status == 200:
                break
            attempt += 1
            if status in (429, 500, 502, 503, 504) and attempt <= self.max_retries:
                time.sleep(min(60, 2 ** attempt * 2))
                continue
            msg = (data.get("error") or {}).get("message", "") if isinstance(data, dict) else ""
            raise RuntimeError(f"LLM request failed: HTTP {status} {msg[:200]}")
        ch = data["choices"][0]
        m = ch.get("message", {})
        usage = data.get("usage", {})
        cost = 0.0 if self.local else estimate_cost(self.model, usage)
        self.spent_rmb += cost if cost == cost else 0.0
        res = LLMResult(content=m.get("content") or "",
                        reasoning=m.get("reasoning_content") or m.get("reasoning") or "",
                        tool_calls=m.get("tool_calls") or [], finish_reason=ch.get("finish_reason", ""),
                        model=self.model, model_version=str(data.get("model", "")), usage=usage, cost_rmb=cost,
                        latency_s=round(time.time() - t0, 3), request_hash=rh)
        if cpath is not None:
            cpath.parent.mkdir(parents=True, exist_ok=True)
            tmp = cpath.with_suffix(".tmp")
            tmp.write_text(canonical(res.to_dict()), encoding="utf-8")
            os.replace(tmp, cpath)
        self._log(res, meta, status)
        return res

    def _log(self, res: LLMResult, meta: dict | None, status: int) -> None:
        if self.ledger is None:
            return
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        rec = {"utc": datetime.now(UTC).isoformat(timespec="seconds"), "status": status,
               "model": res.model, "model_version": res.model_version, "request_hash": res.request_hash,
               "cached": res.cached, "usage": res.usage, "cost_rmb": 0.0 if res.cached else res.cost_rmb,
               "latency_s": res.latency_s, "finish_reason": res.finish_reason,
               "content_sha256": _sha(res.content), "meta": meta or {}}
        with self.ledger.open("a", encoding="utf-8") as f:
            f.write(canonical(rec) + "\n")
