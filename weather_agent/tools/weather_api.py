"""
Async weather data client with multi-layer caching, retry, and fallback.

Provides:
  - resolve_city()        city name → coordinates (cached 1 h)
  - get_forecast()        lat/lon → raw forecast  (cached 5 min)
  - extract_current_hour() raw forecast → current-hour summary
  - check_weather()       high-level: one-shot weather for a city
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any

import httpx

try:
    # When loaded as part of the weather_agent package (ADK from parent dir)
    from ..config import GEO_CACHE_TTL, WEATHER_CACHE_TTL
except ImportError:
    # When run standalone from inside the project directory
    from config import GEO_CACHE_TTL, WEATHER_CACHE_TTL  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

# ── In-memory caches ─────────────────────────────────────────────
_geo_cache: dict[str, dict] = {}
_forecast_cache: dict[tuple[float, float], dict] = {}


def _cache_get(
    cache: dict, key: Any, ttl: int
) -> Any | None:
    entry = cache.get(key)
    if entry and time.time() < entry["expiry"]:
        return entry["data"]
    return None


def _cache_set(cache: dict, key: Any, data: Any, ttl: int) -> None:
    cache[key] = {"data": data, "expiry": time.time() + ttl}


# ── Retry helper ─────────────────────────────────────────────────

async def _retry_http(
    client: httpx.AsyncClient,
    url: str,
    params: dict,
    max_retries: int = 2,
    base_delay: float = 1.0,
) -> dict[str, Any]:
    """GET with exponential-backoff retry."""
    last_exc: Exception | None = None
    for attempt in range(1 + max_retries):
        try:
            r = await client.get(url, params=params, timeout=15)
            r.raise_for_status()
            return r.json()
        except (httpx.HTTPError, httpx.TimeoutException) as exc:
            last_exc = exc
            logger.warning("HTTP error on %s (attempt %d): %s", url, attempt + 1, exc)
            if attempt < max_retries:
                await _asleep(base_delay * (2 ** attempt))
    raise last_exc  # type: ignore[misc]


async def _asleep(secs: float) -> None:
    """Async sleep — works inside ADK's event loop (no asyncio import conflict)."""
    import asyncio
    await asyncio.sleep(secs)


# ── Public API ───────────────────────────────────────────────────

async def resolve_city(city: str) -> dict[str, Any]:
    """Resolve a city name to coordinates using Open-Meteo geocoding.

    Results are cached for *GEO_CACHE_TTL* seconds (default 1 hour).
    Falls back to stripping state suffixes (', MI' → city-only) if the
    full string returns nothing.
    """
    cache_key = city.lower().strip()
    cached = _cache_get(_geo_cache, cache_key, GEO_CACHE_TTL)
    if cached:
        return cached

    candidates: list[str] = [city.strip()]
    if "," in city:
        candidates.append(city.split(",")[0].strip())

    url = "https://geocoding-api.open-meteo.com/v1/search"

    async with httpx.AsyncClient() as client:
        for name in candidates:
            try:
                data = await _retry_http(
                    client, url, {"name": name, "count": 1, "language": "en", "format": "json"}
                )
                results = data.get("results", [])
                if results:
                    loc = results[0]
                    result: dict[str, Any] = {
                        "status": "success",
                        "name": loc.get("name"),
                        "admin1": loc.get("admin1"),
                        "country": loc.get("country"),
                        "latitude": loc.get("latitude"),
                        "longitude": loc.get("longitude"),
                        "timezone": loc.get("timezone", "auto"),
                    }
                    _cache_set(_geo_cache, cache_key, result, GEO_CACHE_TTL)
                    return result
                logger.info("No geocoding results for '%s'", name)
            except Exception as exc:
                logger.warning("Geocoding failed for '%s': %s", name, exc)
                continue

    err: dict[str, Any] = {"status": "error", "message": f"City not found: {city}"}
    _cache_set(_geo_cache, cache_key, err, GEO_CACHE_TTL)
    return err


async def get_forecast(
    latitude: float,
    longitude: float,
    timezone: str = "auto",
) -> dict[str, Any]:
    """Fetch raw hourly forecast from Open-Meteo.

    Cached for *WEATHER_CACHE_TTL* seconds (default 5 min).
    """
    cache_key = (round(latitude, 4), round(longitude, 4))
    cached = _cache_get(_forecast_cache, cache_key, WEATHER_CACHE_TTL)
    if cached:
        return cached

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "temperature_2m,precipitation_probability,precipitation",
        "temperature_unit": "fahrenheit",
        "precipitation_unit": "inch",
        "timezone": timezone or "auto",
        "forecast_days": 1,
    }

    async with httpx.AsyncClient() as client:
        try:
            data = await _retry_http(client, url, params)
            result: dict[str, Any] = {"status": "success", "data": data}
            _cache_set(_forecast_cache, cache_key, result, WEATHER_CACHE_TTL)
            return result
        except Exception as exc:
            logger.error("Forecast API error: %s", exc)
            return {"status": "error", "message": f"Weather API unavailable: {exc}"}


def extract_current_hour(forecast_data: dict[str, Any]) -> dict[str, Any]:
    """Extract the current (or nearest) hour from raw Open-Meteo data."""
    hourly = forecast_data.get("hourly", {})
    times: list[str] = hourly.get("time", [])
    pops: list[float] = hourly.get("precipitation_probability", [])
    precip: list[float] = hourly.get("precipitation", [])
    temps: list[float] = hourly.get("temperature_2m", [])

    if not times:
        return {"status": "error", "message": "No hourly forecast data"}

    now_prefix = datetime.now().strftime("%Y-%m-%dT%H:")
    idx = next((i for i, t in enumerate(times) if t.startswith(now_prefix)), 0)

    prob = pops[idx] if idx < len(pops) else 0
    amt = precip[idx] if idx < len(precip) else 0
    temp = temps[idx] if idx < len(temps) else None

    return {
        "status": "success",
        "time": times[idx],
        "temperature_f": temp,
        "precipitation_probability": prob,
        "precipitation_in": amt,
        "rain_likely": (prob or 0) >= 50 or (amt or 0) > 0,
    }


async def check_weather(city: str) -> dict[str, Any]:
    """High-level convenience: resolve a city, fetch forecast, return current hour.

    This is the primary function exposed as an ADK tool.
    """
    loc = await resolve_city(city)
    if loc.get("status") != "success":
        return loc

    forecast = await get_forecast(loc["latitude"], loc["longitude"], loc.get("timezone", "auto"))
    if forecast.get("status") != "success":
        return forecast

    current = extract_current_hour(forecast["data"])
    if current.get("status") != "success":
        return current

    return {
        "status": "success",
        "location": {
            "name": loc["name"],
            "admin1": loc.get("admin1"),
            "country": loc.get("country"),
        },
        "current_hour": {
            "time": current["time"],
            "temperature_f": current["temperature_f"],
            "precipitation_probability": current["precipitation_probability"],
            "precipitation_in": current["precipitation_in"],
            "rain_likely": current["rain_likely"],
        },
    }
