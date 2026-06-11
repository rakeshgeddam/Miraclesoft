"""
Weather Agent — root orchestrator with stage-by-stage pipeline visibility.

Exposes three separate tools that the LLM calls sequentially:
  1. geocode_city   — resolve city name → coordinates
  2. fetch_forecast — lat/lon → raw hourly forecast
  3. extract_hour   — raw forecast → current-hour summary

Each tool appears as a distinct card in the ADK web UI, showing its
arguments and returned state variables.  An after_tool_callback stores
intermediate results in session.state for cross-stage traceability.

Sub-agent: AlertAgent handles email alerts via AgentTool delegation.
"""

from __future__ import annotations

import asyncio
import logging

from google.adk.agents import Agent
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.base_tool import BaseTool

try:
    # When loaded as part of the weather_agent package (ADK from parent dir)
    from .agents.alert_agent import alert_agent
    from .tools.weather_api import (
        resolve_city as _resolve_city_async,
        get_forecast as _get_forecast_async,
        extract_current_hour,
    )
except ImportError:
    # When run standalone from inside the project directory
    from agents.alert_agent import alert_agent  # type: ignore[import-untyped]
    from tools.weather_api import (  # type: ignore[import-untyped]
        resolve_city as _resolve_city_async,
        get_forecast as _get_forecast_async,
        extract_current_hour,
    )

logger = logging.getLogger(__name__)


# ── Async wrappers (ADK runs in an async event loop) ──────────────


async def geocode_city(city: str) -> dict:
    """Resolve a city name to geographic coordinates.

    Args:
        city: City name, e.g. 'Novi, MI' or 'Detroit'.

    Returns:
        dict with ``status``, ``name``, ``admin1``, ``country``,
        ``latitude``, ``longitude``, ``timezone``.
    """
    return await _resolve_city_async(city)


async def fetch_forecast(
    latitude: float, longitude: float, timezone: str = "auto"
) -> dict:
    """Fetch the hourly forecast for a set of coordinates.

    Args:
        latitude:  Latitude in decimal degrees.
        longitude: Longitude in decimal degrees.
        timezone:  IANA timezone string or ``'auto'``.

    Returns:
        dict with ``status`` and ``data`` (raw Open‑Meteo hourly forecast
        containing ``hourly`` → ``time``, ``temperature_2m``, …).
    """
    return await _get_forecast_async(latitude, longitude, timezone)


async def extract_hour(forecast_data: dict) -> dict:
    """Extract the current (nearest) hour from forecast data.

    Args:
        forecast_data: Raw forecast dict, ``{status, data}`` envelope,
            or ``{fetch_forecast_response: {status, data}}`` wrapping.

    Returns:
        dict with ``time``, ``temperature_f``,
        ``precipitation_probability``, ``precipitation_in``,
        ``rain_likely``.
    """
    # Unwrap ADK-style {tool_name_response: ...} wrapping
    if forecast_data and len(forecast_data) == 1:
        key = next(iter(forecast_data))
        if key.endswith("_response"):
            forecast_data = forecast_data[key]
    # Unwrap {status, data} envelope
    if "data" in forecast_data and isinstance(forecast_data["data"], dict):
        forecast_data = forecast_data["data"]
    return extract_current_hour(forecast_data)


# ── After-tool callback (stores stage state for traceability) ─────


async def _after_tool_callback(
    tool: BaseTool,
    args: dict,
    tool_context,
    tool_response: dict,
) -> dict | None:
    """Record stage results in session.state for the ADK web UI.

    For each pipeline tool the result is stored under
    ``tool_context.state['stage_<tool_name>']`` and key fields are promoted to
    top-level state keys for easy cross-stage reference.
    """
    tool_name = tool.name

    # Only annotate our three pipeline tools
    if tool_name not in ("geocode_city", "fetch_forecast", "extract_hour"):
        return tool_response

    tool_context.state[f"stage_{tool_name}"] = {
        "args": args,
        "result": tool_response,
    }

    if tool_name == "geocode_city" and isinstance(tool_response, dict) and tool_response.get("status") == "success":
        tool_context.state["coordinates"] = {
            "latitude": tool_response["latitude"],
            "longitude": tool_response["longitude"],
            "timezone": tool_response.get("timezone", "auto"),
            "name": tool_response.get("name"),
            "admin1": tool_response.get("admin1"),
        }

    elif tool_name == "fetch_forecast" and isinstance(tool_response, dict) and tool_response.get("status") == "success":
        tool_context.state["forecast_raw"] = tool_response.get("data")

    elif tool_name == "extract_hour" and isinstance(tool_response, dict) and tool_response.get("status") == "success":
        tool_context.state["current_hour"] = {
            k: tool_response[k]
            for k in (
                "time",
                "temperature_f",
                "precipitation_probability",
                "precipitation_in",
                "rain_likely",
            )
            if k in tool_response
        }

    return tool_response  # pass through unchanged


# ── Root agent ───────────────────────────────────────────────────

root_agent = Agent(
    model="gemini-2.5-flash",
    name="WeatherAgent",
    description=(
        "Weather assistant that reports current conditions and can "
        "send email alerts when rain is expected."
    ),
    instruction=(
        "You are a weather assistant with a **3-stage pipeline** for checking "
        "weather and a sub-agent for email alerts.  "
        "Always follow the stages **in order**.\n\n"

        "**STAGE 1 — Geocode**\n"
        "Call the ``geocode_city`` tool to resolve the city name to coordinates.\n"
        "  → Report the resolved location (name, state/province, country) to the user.\n\n"

        "**STAGE 2 — Forecast**\n"
        "Call the ``fetch_forecast`` tool with the **latitude**, **longitude**, "
        "and **timezone** from Stage 1 to get the hourly forecast.\n"
        "  → Say 'Forecast fetched.'\n\n"

        "**STAGE 3 — Extract**\n"
        "Call the ``extract_hour`` tool — pass the **entire result dict** from "
        "Stage 2 as the argument — to get the current hour's conditions.\n"
        "  → Report clearly: temperature (°F), rain probability (%), "
        "rain amount (inches), and whether rain is likely.\n\n"

        "**Email alerts**\n"
        "If the user asks you to send a weather alert or notify someone by "
        "email, first complete Stages 1–3, then **delegate to AlertAgent** "
        "(your sub-agent).  Pass it the recipient email, city, temperature, "
        "rain probability, rain amount, and whether rain is likely.\n\n"

        "**Examples**\n"
        "- User: \"What is the weather in Novi?\"\n"
        "  → geocode_city(\"Novi, MI\") → fetch_forecast(lat, lon, tz) → extract_hour(data)\n"
        "- User: \"Email rakeshgeddam2025@gmail.com about rain in Novi\"\n"
        "  → Same 3 stages, then delegate to AlertAgent with the weather data."
    ),
    tools=[
        geocode_city,
        fetch_forecast,
        extract_hour,
        AgentTool(agent=alert_agent),
    ],
    after_tool_callback=_after_tool_callback,
)
