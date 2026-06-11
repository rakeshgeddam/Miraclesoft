"""Tests for email_client content generation."""

from __future__ import annotations

from tools.email_client import build_html_body, send_weather_alert


def test_build_html_body():
    html = build_html_body("Novi, MI", 72.0, 80, 0.12, True)
    assert "Novi, MI" in html
    assert "72.0" in html
    assert "80%" in html
    assert "Rain Likely" in html
    assert "Yes" in html


def test_build_html_body_no_rain():
    html = build_html_body("Detroit", 65.0, 10, 0.0, False)
    assert "No" in html or "rain_likely" in html


def test_send_weather_alert_dry_run():
    # Should not raise and should return dry_run status
    result = send_weather_alert(
        to_email="test@example.com",
        city="Novi, MI",
        temperature=72.0,
        rain_prob=80,
        rain_amount=0.12,
        rain_likely=True,
        dry_run=True,
    )
    assert result["status"] == "dry_run"
    assert result["recipient"] == "test@example.com"
