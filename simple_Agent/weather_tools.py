"""
Weather Tools — pure functions for geocoding and weather forecasts.

PURPOSE:
  Extracted from agent.py so the same functions can be served via:
    - MCP server (weather_mcp_server.py)
    - Direct ADK Agent (agent.py — backward compatible)
    - Standalone scripts (run_hourly.py imports root_agent which uses these)

DEPENDENCIES:
  - requests (HTTP calls to Open-Meteo API)
  - No API keys required — Open-Meteo is free and open

AGENT USAGE:
  These functions can be called directly from Python code:
    >>> from weather_tools import get_weather
    >>> result = get_weather("Miami, Florida")

  Or wrapped as MCP tools (see weather_mcp_server.py).
  Or used as ADK Agent tools (see agent.py).
"""

import requests
from datetime import datetime


def resolve_city(city: str) -> dict:
    """Resolve a city name into geographic coordinates via Open-Meteo Geocoding.

    Args:
        city: City name, optionally with state/country, e.g. 'Novi, MI'
              or 'Miami, Florida' or 'Paris, France'.

    Returns:
        dict with keys:
            status: "success" or "error"
            name: City name (str)
            admin1: State/region (str or None)
            country: Country name (str)
            latitude: float
            longitude: float
            timezone: IANA timezone string (str)
        On error: {"status": "error", "message": "..."}

    HOW IT WORKS:
        1. Calls Open-Meteo Geocoding API (no key needed)
        2. If "City, State" returns nothing, retries with just "City"
        3. Returns structured result or error

    API: https://geocoding-api.open-meteo.com/v1/search
    """
    url = "https://geocoding-api.open-meteo.com/v1/search"
    candidates = [city.strip()]
    if "," in city:
        candidates.append(city.split(",")[0].strip())

    for name in candidates:
        params = {"name": name, "count": 1, "language": "en", "format": "json"}
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        results = r.json().get("results", [])
        if results:
            loc = results[0]
            return {
                "status": "success",
                "name": loc.get("name"),
                "admin1": loc.get("admin1"),
                "country": loc.get("country"),
                "latitude": loc.get("latitude"),
                "longitude": loc.get("longitude"),
                "timezone": loc.get("timezone"),
            }

    return {"status": "error", "message": f"No location found for '{city}'"}


def get_weather(city: str) -> dict:
    """Fetch the current hour's weather forecast for a city.

    Combines geocoding + forecast in one call. Returns temperature,
    precipitation probability, and rain likelihood for the current hour.

    Args:
        city: City name, e.g. 'Novi, MI' or 'Miami, Florida'.

    Returns:
        dict with keys:
            status: "success" or "error"
            location: dict with name, admin1, country
            current_hour: dict with time, temperature_f, precipitation_probability,
                          precipitation_in, rain_likely
        On error: {"status": "error", "message": "..."}

    HOW IT WORKS:
        1. Resolves city to coordinates via resolve_city()
        2. Calls Open-Meteo Forecast API with those coordinates
        3. Finds the current hour's data in the hourly array
        4. Returns structured result

    API: https://api.open-meteo.com/v1/forecast
    """
    loc = resolve_city(city)
    if loc.get("status") != "success":
        return loc

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": loc["latitude"],
        "longitude": loc["longitude"],
        "hourly": "temperature_2m,precipitation_probability,precipitation",
        "temperature_unit": "fahrenheit",
        "precipitation_unit": "inch",
        "timezone": loc["timezone"] or "auto",
        "forecast_days": 1,
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    hourly = r.json().get("hourly", {})

    times = hourly.get("time", [])
    pop = hourly.get("precipitation_probability", [])
    precip = hourly.get("precipitation", [])
    temps = hourly.get("temperature_2m", [])

    if not times:
        return {"status": "error", "message": "No forecast data returned"}

    now_prefix = datetime.now().strftime("%Y-%m-%dT%H:")
    idx = next((i for i, t in enumerate(times) if t.startswith(now_prefix)), 0)

    prob = pop[idx] if idx < len(pop) else 0
    amount = precip[idx] if idx < len(precip) else 0
    temp = temps[idx] if idx < len(temps) else None

    return {
        "status": "success",
        "location": {
            "name": loc["name"],
            "admin1": loc.get("admin1"),
            "country": loc.get("country"),
        },
        "current_hour": {
            "time": times[idx],
            "temperature_f": temp,
            "precipitation_probability": prob,
            "precipitation_in": amount,
            "rain_likely": (prob or 0) >= 50 or (amount or 0) > 0,
        },
    }
