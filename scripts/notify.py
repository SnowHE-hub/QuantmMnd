"""轻量级通知工具：PushPlus（微信）/ Bark（iOS）/ 企业微信 Bot.

用法：
  from scripts.notify import notify
  notify("QuantMind", "step4 财务数据下载完成，1374只全部就绪")

配置（二选一，写入 api_key.txt 或 export 到环境变量）：
  PUSHPLUS_TOKEN=your_token      # pushplus.plus 注册，微信扫码绑定
  BARK_KEY=your_bark_key         # iOS Bark App → 复制 Key
  WECOM_WEBHOOK=https://...      # 企业微信机器人 Webhook URL
"""
from __future__ import annotations

import os
import json
import urllib.request
import urllib.parse
from pathlib import Path


def _load_api_keys() -> dict[str, str]:
    """从 api_key.txt 加载，格式同现有文件（key：value）。"""
    keys: dict[str, str] = {}
    key_file = Path(__file__).resolve().parent.parent / "api_key.txt"
    if key_file.exists():
        for line in key_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if "：" in line:
                k, v = line.split("：", 1)
                keys[k.strip()] = v.strip()
            elif ":" in line and not line.startswith("#"):
                k, v = line.split(":", 1)
                keys[k.strip()] = v.strip()
    return keys


def _pushplus(token: str, title: str, content: str) -> bool:
    url = "https://www.pushplus.plus/send"
    data = json.dumps({
        "token": token,
        "title": title,
        "content": content,
        "template": "txt",
    }).encode("utf-8")
    try:
        req = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
            return result.get("code") == 200
    except Exception as e:
        print(f"[notify/pushplus] 失败: {e}")
        return False


def _bark(key: str, title: str, content: str) -> bool:
    encoded = urllib.parse.quote(content)
    url = f"https://api.day.app/{key}/{urllib.parse.quote(title)}/{encoded}"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return resp.status == 200
    except Exception as e:
        print(f"[notify/bark] 失败: {e}")
        return False


def _wecom(webhook: str, title: str, content: str) -> bool:
    data = json.dumps({
        "msgtype": "text",
        "text": {"content": f"【{title}】\n{content}"},
    }).encode("utf-8")
    try:
        req = urllib.request.Request(
            webhook, data=data,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read()).get("errcode") == 0
    except Exception as e:
        print(f"[notify/wecom] 失败: {e}")
        return False


def notify(title: str, content: str, silent: bool = False) -> None:
    """发送通知，自动检测可用渠道（优先级：PushPlus > Bark > WeCom）。

    Args:
        title:   通知标题（如 "step4 完成"）
        content: 通知正文
        silent:  True=失败不报错（在脚本末尾调用时设为True）
    """
    keys = _load_api_keys()

    pushplus = os.environ.get("PUSHPLUS_TOKEN") or keys.get("PUSHPLUS_TOKEN")
    bark     = os.environ.get("BARK_KEY")       or keys.get("BARK_KEY")
    wecom    = os.environ.get("WECOM_WEBHOOK")  or keys.get("WECOM_WEBHOOK")

    sent = False
    if pushplus:
        sent = _pushplus(pushplus, title, content)
        if sent:
            print(f"[notify] PushPlus 已发送：{title}")

    if not sent and bark:
        sent = _bark(bark, title, content)
        if sent:
            print(f"[notify] Bark 已发送：{title}")

    if not sent and wecom:
        sent = _wecom(wecom, title, content)
        if sent:
            print(f"[notify] 企业微信 已发送：{title}")

    if not sent:
        if not silent:
            print(f"[notify] 无可用通知渠道（配置 PUSHPLUS_TOKEN / BARK_KEY / WECOM_WEBHOOK）")
        print(f"[notify] 消息内容：{title} — {content}")


if __name__ == "__main__":
    import sys
    title   = sys.argv[1] if len(sys.argv) > 1 else "QuantMind Test"
    content = sys.argv[2] if len(sys.argv) > 2 else "通知功能测试"
    notify(title, content)
