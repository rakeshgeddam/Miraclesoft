"""Tests for weather_api caching, retry, and data extraction."""

from __future__ import annotations

from tools.weather_api import extract_current_hour


def _make_forecast(temp: float, prob: float, precip: float,
                   hour: str = "2026-06-08T14:00") -> dict:
    return {
        "hourly": {
            "time": [hour, "2026-06-08T15:00", "2026-06-08T16:00"],
            "temperature_2m": [temp, temp + 2, temp + 4],
            "precipitation_probability": [prob, 10, 5],
            "precipitation": [precip, 0.0, 0.0],
        }
    }


def test_extract_rain_likely():
    data = _make_forecast(72.0, 80.0, 0.12)
    result = extract_current_hour(data)
    assert result["status"] == "success"
    assert result["temperature_f"] == 72.0
    assert result["precipitation_probability"] == 80.0
    assert result["rain_likely"] is True


def test_extract_no_rain():
    data = _make_forecast(65.0, 10.0, 0.0)
    result = extract_current_hour(data)
    assert result["rain_likely"] is False


def test_extract_empty_hourly():
    result = extract_current_hour({"hourly": {}})
    assert result["status"] == "error"


def test_extract_empty_times():
    result = extract_current_hour({"hourly": {"time": []}})
    assert result["status"] == "error"
