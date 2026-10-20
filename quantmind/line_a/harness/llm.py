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
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

BJ = timezone(timedelta(hours=8))
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
# 元 / 百万 tokens：(缓存命中输入, 未命中输入, 输出)；来源 api-docs.deepseek.com 价格页（2026-10-21 查阅）
PRICES = {"deepseek-flash": {"idle": (0.02, 1.0, 4.0), "peak": (0.04, 2.0, 8.0)},
          "deepseek-v4-pro": {"idle": (0.15, 4.5, 13.5), "peak": (0.30, 9.0, 27.0)}}
# 兜底检查（结构上 Harness 本来就不把账户信息放进提示词）；只列不会出现在公开公告里的说法，避免误拦
PRIVATE_PATTERNS = (r"账户规模", r"账户资金", r"资金账户", r"持仓市值", r"我的持仓", r"真实持仓", r"可用资金",
                    r"account[_ ]?size", r"my holdings", r"\[PRIVATE\]")


class PrivateContentError(ValueError):
    pass


class BudgetExceededError(RuntimeError):
    pass


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


def read_key(name: str = "DEEPSEEK_API_KEY") -> str:
    """环境变量优先；WSL 里没有时，从 Windows 的系统（Machine）与用户环境变量读取。返回值只留在内存里。"""
    v = os.environ.get(name, "").strip()
    if v:
        return v
    ps = "powershell.exe" if os.name == "nt" else "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
    if os.name == "nt" or Path(ps).exists():
        for scope in ("Machine", "User"):
            r = subprocess.run([ps, "-NoProfile", "-NonInteractive", "-Command",
                                f"[Environment]::GetEnvironmentVariable('{name}','{scope}')"],
                               capture_output=True, text=True, timeout=30)
            v = r.stdout.strip()
            if v:
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
    spent_rmb: float = 0.0
    _key: str | None = field(default=None, repr=False)

    def _post(self, body: dict) -> tuple[int, dict]:
        if self.transport is not None:
            return self.transport(DEEPSEEK_URL, {}, body)
        import httpx
        if self._key is None:
            self._key = read_key()
        r = httpx.post(DEEPSEEK_URL, json=body, timeout=self.timeout_s,
                       headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"})
        try:
            return r.status_code, r.json()
        except ValueError:
            return r.status_code, {"error": {"message": r.text[:500]}}

    def chat(self, messages: list[dict], *, tools: list[dict] | None = None, json_mode: bool = False,
             thinking: str | None = None, max_tokens: int = 4096, meta: dict | None = None) -> LLMResult:
        """thinking：None = 关闭思考；"low" / "high" / "max" = 开启并设强度。"""
        assert_public(messages)
        body: dict = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "stream": False}
        if tools:
            body["tools"] = tools
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if thinking:
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
        for attempt in range(self.max_retries + 1):
            status, data = self._post(body)
            if status == 200:
                break
            if status in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                time.sleep(min(60, 2 ** attempt * 2))
                continue
            msg = (data.get("error") or {}).get("message", "") if isinstance(data, dict) else ""
            raise RuntimeError(f"LLM request failed: HTTP {status} {msg[:200]}")
        ch = data["choices"][0]
        m = ch.get("message", {})
        usage = data.get("usage", {})
        cost = estimate_cost(self.model, usage)
        self.spent_rmb += cost if cost == cost else 0.0
        res = LLMResult(content=m.get("content") or "", reasoning=m.get("reasoning_content") or "",
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
