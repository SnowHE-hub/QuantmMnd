#!/usr/bin/env python3
"""双 Token 健康探针 — 验证 Token A / B 的连通性、限速和积分状态.

用法
----
    python scripts/verify_tokens.py
    # 或指定只测某个
    python scripts/verify_tokens.py --token-a-only
    python scripts/verify_tokens.py --token-b-only

输出
----
    reports/data/tokens.md  （包含时间戳 + 结果表）
    以及标准输出颜色打印
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ── 复用既有 .env 机制读取凭证 + 屏蔽 provider 日志（防 token 外泄）──────────────
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
except ImportError:
    pass
try:
    from quantmind.utils.silence_provider_logging import silence_provider_logging
    silence_provider_logging()
except Exception:
    pass
REPORT_DIR = ROOT / "reports" / "data"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TOKEN_A = os.environ.get(
    "TUSHARE_TOKEN",
    "",
)
TOKEN_B = os.environ.get(
    "TUSHARE_TOKEN_HI",
    "",
)
URL_B = os.environ.get("TUSHARE_HI_URL", "http://tsy.xiaodefa.cn")


# ────────────────────────────────────────────────────────────────────────────

def _probe(token: str, url: str | None, name: str) -> dict:
    """对单个 Token 做全套探针，返回结果字典。"""
    import tushare as ts  # type: ignore[import-untyped]

    result: dict = {
        "name": name,
        "token_prefix": token[:8] + "...",
        "url": url or "https://api.waditu.com (官方)",
        "ok": False,
        "error": None,
        "latency_ms": None,
        "rows_returned": None,
        "rate_test": None,
    }

    try:
        # 直接实例化 DataApi，绕过 get_token() 优先读 TUSHARE_TOKEN 环境变量的问题
        from tushare.pro.client import DataApi  # type: ignore[import-untyped]
        pro = DataApi(token=token, timeout=30)
        if url:
            pro._DataApi__http_url = url  # type: ignore[attr-defined]

        # ── 连通性（stock_basic 只拉 3 行）────────────────────────────────
        t0 = time.monotonic()
        df = pro.stock_basic(
            list_status="L",
            fields="ts_code,name,list_date",
            limit=3,
        )
        latency = (time.monotonic() - t0) * 1000

        if df is None or len(df) == 0:
            result["error"] = "连通正常但返回空数据（可能权限不足）"
            return result

        result["ok"] = True
        result["latency_ms"] = round(latency, 1)
        result["rows_returned"] = len(df)
        print(f"  [{name}] ✓ 连通 latency={latency:.0f}ms rows={len(df)}", flush=True)

        # ── 限速测试（连续 5 次 trade_cal）────────────────────────────────
        speeds = []
        for _ in range(5):
            t0 = time.monotonic()
            pro.trade_cal(
                exchange="SSE",
                start_date="20240101",
                end_date="20240110",
                is_open="1",
                fields="cal_date",
            )
            speeds.append((time.monotonic() - t0) * 1000)
            time.sleep(0.1)

        avg_speed = sum(speeds) / len(speeds)
        max_rpm = round(60_000 / avg_speed)
        result["rate_test"] = f"avg {avg_speed:.0f}ms/req → 理论上限 ~{max_rpm} req/min"
        print(f"  [{name}] 速率测试 avg={avg_speed:.0f}ms → ~{max_rpm}/min", flush=True)

    except Exception as e:
        result["error"] = str(e)[:200]
        print(f"  [{name}] ✗ 失败: {result['error']}", flush=True)

    return result


def _write_report(results: list[dict]):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# Token 健康报告",
        f"",
        f"生成时间：`{now}`",
        f"",
        f"| 项目 | Token A（自有 2000积分） | Token B（15000积分，到期 2026-05-19） |",
        f"|---|---|---|",
    ]

    def _get(r: dict | None, key: str, default: str = "—") -> str:
        if r is None:
            return default
        v = r.get(key)
        return str(v) if v is not None else default

    ra = next((r for r in results if r["name"] == "Token A"), None)
    rb = next((r for r in results if r["name"] == "Token B"), None)

    rows = [
        ("Token prefix", "token_prefix"),
        ("URL", "url"),
        ("连通", "ok"),
        ("延迟", "latency_ms"),
        ("返回行数", "rows_returned"),
        ("限速估计", "rate_test"),
        ("错误", "error"),
    ]
    for label, key in rows:
        va = _get(ra, key)
        vb = _get(rb, key)
        lines.append(f"| {label} | {va} | {vb} |")

    lines += [
        "",
        "## 结论",
        "",
    ]
    if ra and ra["ok"] and rb and rb["ok"]:
        lines.append("✅ 双 Token 均可用，可按分工方案启动 sidecar。")
    elif ra and ra["ok"]:
        lines.append("⚠️ Token A 正常，Token B 有问题，sidecar 暂停，主进程继续。")
    elif rb and rb["ok"]:
        lines.append("⚠️ Token B 正常，Token A 有问题，请检查主进程环境变量。")
    else:
        lines.append("❌ 双 Token 均失败，请检查网络/到期/环境变量。")

    report = "\n".join(lines) + "\n"
    out = REPORT_DIR / "tokens.md"
    out.write_text(report, encoding="utf-8")
    print(f"\n[report] 已写入 {out}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description="双 Token 健康探针")
    parser.add_argument("--token-a-only", action="store_true")
    parser.add_argument("--token-b-only", action="store_true")
    args = parser.parse_args()

    results = []

    if not args.token_b_only:
        print("\n── Token A (官方 2000积分) ──", flush=True)
        results.append(_probe(TOKEN_A, url=None, name="Token A"))

    if not args.token_a_only:
        print("\n── Token B (高频代理 15000积分) ──", flush=True)
        results.append(_probe(TOKEN_B, url=URL_B, name="Token B"))

    report = _write_report(results)
    print("\n" + "─" * 60)
    print(report)

    # 退出码：全部 ok = 0，部分失败 = 1，全部失败 = 2
    all_ok = all(r.get("ok") for r in results)
    any_ok = any(r.get("ok") for r in results)
    return 0 if all_ok else (1 if any_ok else 2)


if __name__ == "__main__":
    sys.exit(main())
