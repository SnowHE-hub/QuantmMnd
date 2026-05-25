"""quantmind/notify/daily_alert.py — 每日选股预警推送（企业微信 Webhook + SMTP 邮件）.

设计决策：
- DailyAlertSender 是纯推送类，不包含任何数据计算逻辑；
  调用方（daily_update.py）负责传入 picks DataFrame 与 regime/market_flow。
- 所有凭证从环境变量读取，不写死在代码中：
    WECOM_WEBHOOK_URL  企业微信机器人 Webhook
    SMTP_HOST          SMTP 服务器主机
    SMTP_PORT          SMTP 端口（默认 465 SSL）
    SMTP_USER          SMTP 用户名（发件人）
    SMTP_PASS          SMTP 密码
    ALERT_EMAIL        收件人邮箱（逗号分隔多个）
- 任一通道失败不影响另一通道；方法返回 bool 表示整体是否至少一路成功。
"""
from __future__ import annotations

import logging
import os
import smtplib
import ssl
import textwrap
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ─── 配置键名 ─────────────────────────────────────────────────────────────────
_ENV_WECOM  = "WECOM_WEBHOOK_URL"
_ENV_HOST   = "SMTP_HOST"
_ENV_PORT   = "SMTP_PORT"
_ENV_USER   = "SMTP_USER"
_ENV_PASS   = "SMTP_PASS"
_ENV_EMAIL  = "ALERT_EMAIL"

_DEFAULT_SMTP_PORT = 465

# Regime 对应 emoji
_REGIME_EMOJI: dict[str, str] = {
    "bull":    "🐂",
    "neutral": "⚖️",
    "bear":    "🐻",
}

# 推荐评级颜色（Markdown 企微不支持颜色，仅用于 HTML 邮件）
_RATING_HEX: dict[str, str] = {
    "强烈买入": "#00B894",
    "买入":     "#55EFC4",
    "持有":     "#FDCB6E",
    "观望":     "#636E72",
}


class DailyAlertSender:
    """每日选股预警推送器.

    Args:
        wecom_url:  企业微信 Webhook URL；``None`` 则从 ``WECOM_WEBHOOK_URL`` 读取。
        smtp_host:  SMTP 服务器；``None`` 则从 ``SMTP_HOST`` 读取。
        smtp_port:  SMTP 端口；``None`` 则从 ``SMTP_PORT`` 读取（默认 465）。
        smtp_user:  SMTP 用户名；``None`` 则从 ``SMTP_USER`` 读取。
        smtp_pass:  SMTP 密码；``None`` 则从 ``SMTP_PASS`` 读取。
        alert_email: 收件人（逗号分隔）；``None`` 则从 ``ALERT_EMAIL`` 读取。
        timeout:    HTTP/SMTP 超时秒数（默认 10）。
    """

    def __init__(
        self,
        *,
        wecom_url:   str | None = None,
        smtp_host:   str | None = None,
        smtp_port:   int | None = None,
        smtp_user:   str | None = None,
        smtp_pass:   str | None = None,
        alert_email: str | None = None,
        timeout:     int = 10,
    ) -> None:
        self._wecom_url   = wecom_url   or os.environ.get(_ENV_WECOM,  "")
        self._smtp_host   = smtp_host   or os.environ.get(_ENV_HOST,   "")
        self._smtp_port   = smtp_port   or int(os.environ.get(_ENV_PORT, _DEFAULT_SMTP_PORT))
        self._smtp_user   = smtp_user   or os.environ.get(_ENV_USER,   "")
        self._smtp_pass   = smtp_pass   or os.environ.get(_ENV_PASS,   "")
        self._alert_email = alert_email or os.environ.get(_ENV_EMAIL,  "")
        self._timeout     = timeout

    # ─────────────────────────────────────────────────────────────────────────
    # 公开 API
    # ─────────────────────────────────────────────────────────────────────────

    def send_daily_picks(
        self,
        picks: pd.DataFrame,
        regime: str = "neutral",
        market_flow: float = 0.0,
    ) -> bool:
        """推送每日选股到企业微信 + 邮件.

        Args:
            picks:       包含 ticker/name/composite_score/rating/industry 列的 DataFrame；
                         行按推荐优先级排序（第1行=首选）。
            regime:      HMM 当前 Regime（'bull'/'neutral'/'bear'）。
            market_flow: 北向资金5日净流入（M元，正=净买入）。

        Returns:
            True 表示至少一个通道发送成功。
        """
        md_text  = self.format_markdown(picks, regime, market_flow)
        html_str = self.format_html(picks, regime, market_flow)
        subject  = self._build_subject(picks, regime)

        wecom_ok = self._send_wecom(md_text)
        email_ok = self._send_email(subject, html_str, md_text)

        if not wecom_ok and not email_ok:
            logger.error("[DailyAlert] 企业微信与邮件均推送失败")
        return wecom_ok or email_ok

    # ─────────────────────────────────────────────────────────────────────────
    # 格式化：Markdown（企微）
    # ─────────────────────────────────────────────────────────────────────────

    def format_markdown(
        self,
        picks: pd.DataFrame,
        regime: str = "neutral",
        market_flow: float = 0.0,
    ) -> str:
        """生成企业微信 Markdown 正文（不超过 4096 字节）."""
        emoji  = _REGIME_EMOJI.get(regime, "❓")
        flow_s = f"+{market_flow:.0f}" if market_flow >= 0 else f"{market_flow:.0f}"
        flow_emoji = "🟢" if market_flow > 0 else ("🔴" if market_flow < 0 else "⚪")

        lines: list[str] = [
            f"# 📈 QuantMind 每日选股推荐",
            f"> Regime: **{regime.upper()}** {emoji}　北向资金5d: **{flow_s} M元** {flow_emoji}",
            "",
        ]

        if picks.empty:
            lines.append("*今日无推荐股票*")
        else:
            for _, row in picks.iterrows():
                ticker   = row.get("ticker", "")
                name     = row.get("name", ticker)
                score    = row.get("composite_score", row.get("predicted_score", float("nan")))
                rating   = row.get("rating", "")
                industry = row.get("industry", "")

                score_s = f"{score:.4f}" if pd.notna(score) else "—"
                tag_s   = f"[{industry}]" if industry else ""
                lines.append(
                    f"- **{name}** `{ticker}` {tag_s}　评级: **{rating}**　分: `{score_s}`"
                )

        lines += [
            "",
            f"*下次审查：查看 Streamlit 持仓跟踪页*",
        ]
        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────────────────────
    # 格式化：HTML（邮件）
    # ─────────────────────────────────────────────────────────────────────────

    def format_html(
        self,
        picks: pd.DataFrame,
        regime: str = "neutral",
        market_flow: float = 0.0,
    ) -> str:
        """生成 HTML 邮件正文."""
        emoji  = _REGIME_EMOJI.get(regime, "❓")
        flow_s = f"+{market_flow:.0f}" if market_flow >= 0 else f"{market_flow:.0f}"
        regime_color = {"bull": "#00B894", "neutral": "#FDCB6E", "bear": "#D63031"}.get(
            regime, "#636E72"
        )

        rows_html = ""
        if picks.empty:
            rows_html = (
                "<tr><td colspan='5' style='text-align:center;color:#999'>今日无推荐</td></tr>"
            )
        else:
            for i, (_, row) in enumerate(picks.iterrows(), 1):
                ticker   = row.get("ticker", "")
                name     = row.get("name", ticker)
                score    = row.get("composite_score", row.get("predicted_score", float("nan")))
                rating   = row.get("rating", "")
                industry = row.get("industry", "")

                score_s      = f"{score:.4f}" if pd.notna(score) else "—"
                rating_color = _RATING_HEX.get(rating, "#636E72")
                bg           = "#f9f9f9" if i % 2 == 0 else "#ffffff"
                rows_html += textwrap.dedent(f"""\
                    <tr style='background:{bg}'>
                      <td style='padding:8px;text-align:center'>{i}</td>
                      <td style='padding:8px;font-weight:600'>{name}</td>
                      <td style='padding:8px;color:#555'>{ticker}</td>
                      <td style='padding:8px;color:#555'>{industry}</td>
                      <td style='padding:8px;text-align:center;color:{rating_color};font-weight:600'>{rating}</td>
                      <td style='padding:8px;text-align:right;font-family:monospace'>{score_s}</td>
                    </tr>
                """)

        html = textwrap.dedent(f"""\
            <!DOCTYPE html><html><body style='font-family:sans-serif;max-width:680px;margin:auto'>
            <div style='background:{regime_color};padding:16px 24px;border-radius:8px;color:white;margin-bottom:20px'>
              <h2 style='margin:0'>📈 QuantMind 每日选股推荐</h2>
              <p style='margin:6px 0 0'>Regime: <b>{regime.upper()}</b> {emoji}
                 &nbsp;|&nbsp; 北向资金5d: <b>{flow_s} M元</b></p>
            </div>
            <table width='100%' border='0' cellspacing='0' cellpadding='0'
                   style='border-collapse:collapse;border:1px solid #ddd;border-radius:6px;overflow:hidden'>
              <thead>
                <tr style='background:#2d3436;color:white'>
                  <th style='padding:10px 8px'>排名</th>
                  <th style='padding:10px 8px;text-align:left'>名称</th>
                  <th style='padding:10px 8px;text-align:left'>代码</th>
                  <th style='padding:10px 8px;text-align:left'>行业</th>
                  <th style='padding:10px 8px'>评级</th>
                  <th style='padding:10px 8px;text-align:right'>综合分</th>
                </tr>
              </thead>
              <tbody>
                {rows_html}
              </tbody>
            </table>
            <p style='color:#999;font-size:12px;margin-top:20px'>
              由 QuantMind 自动生成 &nbsp;|&nbsp; 本邮件仅供参考，不构成投资建议
            </p>
            </body></html>
        """)
        return html

    # ─────────────────────────────────────────────────────────────────────────
    # 内部：企业微信
    # ─────────────────────────────────────────────────────────────────────────

    def _send_wecom(self, markdown_text: str) -> bool:
        if not self._wecom_url:
            logger.debug("[DailyAlert] WECOM_WEBHOOK_URL 未配置，跳过企业微信推送")
            return False
        payload = {
            "msgtype": "markdown",
            "markdown": {"content": markdown_text[:4096]},
        }
        try:
            resp = requests.post(
                self._wecom_url, json=payload, timeout=self._timeout
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("errcode", -1) != 0:
                logger.warning(f"[DailyAlert] 企业微信返回错误: {data}")
                return False
            logger.info("[DailyAlert] ✅ 企业微信推送成功")
            return True
        except Exception as exc:
            logger.warning(f"[DailyAlert] ⚠️ 企业微信推送失败: {exc}")
            return False

    # ─────────────────────────────────────────────────────────────────────────
    # 内部：邮件
    # ─────────────────────────────────────────────────────────────────────────

    def _send_email(self, subject: str, html_body: str, plain_body: str) -> bool:
        if not (self._smtp_host and self._smtp_user and self._alert_email):
            logger.debug("[DailyAlert] SMTP 配置不完整，跳过邮件推送")
            return False
        recipients = [addr.strip() for addr in self._alert_email.split(",") if addr.strip()]
        if not recipients:
            logger.debug("[DailyAlert] ALERT_EMAIL 为空，跳过邮件推送")
            return False

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = self._smtp_user
        msg["To"]      = ", ".join(recipients)
        msg.attach(MIMEText(plain_body, "plain", "utf-8"))
        msg.attach(MIMEText(html_body,  "html",  "utf-8"))

        try:
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(
                self._smtp_host, self._smtp_port, context=context, timeout=self._timeout
            ) as server:
                server.login(self._smtp_user, self._smtp_pass)
                server.sendmail(self._smtp_user, recipients, msg.as_string())
            logger.info(f"[DailyAlert] ✅ 邮件推送成功 → {recipients}")
            return True
        except Exception as exc:
            logger.warning(f"[DailyAlert] ⚠️ 邮件推送失败: {exc}")
            return False

    # ─────────────────────────────────────────────────────────────────────────
    # 辅助
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_subject(picks: pd.DataFrame, regime: str) -> str:
        n = len(picks)
        emoji = _REGIME_EMOJI.get(regime, "❓")
        return f"QuantMind 每日选股 {emoji} {regime.upper()} | 推荐 {n} 只"
