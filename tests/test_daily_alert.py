"""tests/test_daily_alert.py — DailyAlertSender 单元测试（全部 mock，无真实网络）."""
from __future__ import annotations

import smtplib
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import requests

from quantmind.notify.daily_alert import DailyAlertSender


# ─── fixtures ────────────────────────────────────────────────────────────────

_RATINGS   = ["强烈买入", "买入", "持有", "观望"]
_INDUSTRIES = ["科技", "金融", "消费", "工业", "医药", "能源", "建筑", "化工", "农业", "零售"]


def _make_picks(n: int = 5) -> pd.DataFrame:
    return pd.DataFrame({
        "ticker":          [f"00000{i}.SZ"            for i in range(n)],
        "name":            [f"股票{i}"                  for i in range(n)],
        "composite_score": [0.9 - i * 0.05            for i in range(n)],
        "rating":          [_RATINGS[i % len(_RATINGS)]       for i in range(n)],
        "industry":        [_INDUSTRIES[i % len(_INDUSTRIES)] for i in range(n)],
    })


def _sender(**kwargs) -> DailyAlertSender:
    """返回不依赖环境变量的测试用 sender."""
    defaults = dict(
        wecom_url="https://fake.wecom.example/webhook",
        smtp_host="smtp.fake.example",
        smtp_port=465,
        smtp_user="bot@fake.example",
        smtp_pass="password",
        alert_email="user@fake.example",
        timeout=5,
    )
    defaults.update(kwargs)
    return DailyAlertSender(**defaults)


# ─── 1. format_markdown ───────────────────────────────────────────────────────

def test_format_markdown_contains_regime():
    s = _sender()
    md = s.format_markdown(_make_picks(), regime="bull", market_flow=1234.5)
    assert "BULL" in md
    assert "🐂" in md


def test_format_markdown_contains_flow():
    s = _sender()
    md = s.format_markdown(_make_picks(), regime="neutral", market_flow=-500.0)
    assert "-500" in md


def test_format_markdown_contains_tickers():
    picks = _make_picks(3)
    s = _sender()
    md = s.format_markdown(picks)
    for tk in picks["ticker"]:
        assert tk in md


def test_format_markdown_empty_picks():
    s = _sender()
    md = s.format_markdown(pd.DataFrame())
    assert "无推荐" in md


def test_format_markdown_length_within_limit():
    """企业微信 Markdown 限制 4096 bytes."""
    s = _sender()
    picks = _make_picks(5)
    md = s.format_markdown(picks, regime="bull", market_flow=99999.0)
    assert len(md.encode("utf-8")) <= 4096


# ─── 2. format_html ──────────────────────────────────────────────────────────

def test_format_html_is_valid_html():
    s = _sender()
    html = s.format_html(_make_picks())
    assert "<html>" in html or "<!DOCTYPE html>" in html
    assert "</body>" in html


def test_format_html_contains_tickers():
    picks = _make_picks(3)
    s = _sender()
    html = s.format_html(picks)
    for tk in picks["ticker"]:
        assert tk in html


def test_format_html_contains_regime_color():
    s = _sender()
    html = s.format_html(_make_picks(), regime="bear")
    assert "#D63031" in html  # bear 颜色


def test_format_html_empty_picks():
    s = _sender()
    html = s.format_html(pd.DataFrame())
    assert "今日无推荐" in html


# ─── 3. _send_wecom（企业微信）────────────────────────────────────────────────

def test_send_wecom_success():
    s = _sender()
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"errcode": 0, "errmsg": "ok"}
    mock_resp.raise_for_status.return_value = None
    with patch("requests.post", return_value=mock_resp) as mock_post:
        result = s._send_wecom("test markdown")
    assert result is True
    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args
    assert call_kwargs[1]["json"]["msgtype"] == "markdown"


def test_send_wecom_api_error():
    s = _sender()
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"errcode": 40001, "errmsg": "invalid webhook url"}
    mock_resp.raise_for_status.return_value = None
    with patch("requests.post", return_value=mock_resp):
        result = s._send_wecom("test")
    assert result is False


def test_send_wecom_network_error():
    s = _sender()
    with patch("requests.post", side_effect=requests.exceptions.ConnectionError("timeout")):
        result = s._send_wecom("test")
    assert result is False


def test_send_wecom_no_url():
    s = DailyAlertSender(wecom_url="")
    result = s._send_wecom("test")
    assert result is False


# ─── 4. _send_email（SMTP）───────────────────────────────────────────────────

def test_send_email_success():
    s = _sender()
    mock_server = MagicMock()
    mock_server.__enter__ = lambda self: self
    mock_server.__exit__ = MagicMock(return_value=False)
    with patch("smtplib.SMTP_SSL", return_value=mock_server):
        result = s._send_email("Test Subject", "<html></html>", "plain text")
    assert result is True
    mock_server.login.assert_called_once_with("bot@fake.example", "password")
    mock_server.sendmail.assert_called_once()


def test_send_email_auth_failure():
    s = _sender()
    with patch("smtplib.SMTP_SSL", side_effect=smtplib.SMTPAuthenticationError(535, b"fail")):
        result = s._send_email("Subject", "<html></html>", "plain")
    assert result is False


def test_send_email_missing_config():
    s = DailyAlertSender(smtp_host="", smtp_user="", alert_email="")
    result = s._send_email("Subject", "<html></html>", "plain")
    assert result is False


def test_send_email_multiple_recipients():
    s = _sender(alert_email="a@x.com, b@x.com, c@x.com")
    mock_server = MagicMock()
    mock_server.__enter__ = lambda self: self
    mock_server.__exit__ = MagicMock(return_value=False)
    with patch("smtplib.SMTP_SSL", return_value=mock_server):
        result = s._send_email("Subject", "<html></html>", "plain")
    assert result is True
    _, send_args, _ = mock_server.sendmail.mock_calls[0]
    assert len(send_args[1]) == 3  # 3 recipients


# ─── 5. send_daily_picks（集成）────────────────────────────────────────────────

def test_send_daily_picks_both_channels_ok():
    s = _sender()
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"errcode": 0}
    mock_resp.raise_for_status.return_value = None
    mock_server = MagicMock()
    mock_server.__enter__ = lambda self: self
    mock_server.__exit__ = MagicMock(return_value=False)
    with patch("requests.post", return_value=mock_resp), \
         patch("smtplib.SMTP_SSL", return_value=mock_server):
        ok = s.send_daily_picks(_make_picks(), regime="bull", market_flow=1500.0)
    assert ok is True


def test_send_daily_picks_wecom_fail_email_ok():
    """企业微信失败但邮件成功 → 整体返回 True."""
    s = _sender()
    mock_server = MagicMock()
    mock_server.__enter__ = lambda self: self
    mock_server.__exit__ = MagicMock(return_value=False)
    with patch("requests.post", side_effect=Exception("wecom down")), \
         patch("smtplib.SMTP_SSL", return_value=mock_server):
        ok = s.send_daily_picks(_make_picks())
    assert ok is True


def test_send_daily_picks_all_fail():
    """两个通道都失败 → 返回 False."""
    s = _sender()
    with patch("requests.post", side_effect=Exception("down")), \
         patch("smtplib.SMTP_SSL", side_effect=Exception("down")):
        ok = s.send_daily_picks(_make_picks())
    assert ok is False


# ─── 6. _build_subject ───────────────────────────────────────────────────────

def test_build_subject_contains_n_and_regime():
    picks = _make_picks(7)
    subject = DailyAlertSender._build_subject(picks, "bear")
    assert "7" in subject
    assert "BEAR" in subject
    assert "🐻" in subject


def test_build_subject_empty_picks():
    subject = DailyAlertSender._build_subject(pd.DataFrame(), "neutral")
    assert "0" in subject
