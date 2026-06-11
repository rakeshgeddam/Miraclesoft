"""
Alert Agent (ADK sub-agent).

Specialises in sending weather-alert emails via Gmail SMTP.
Used by the root orchestrator via AgentTool when the user requests
email notifications for weather conditions.
"""

from __future__ import annotations

import logging

from google.adk.agents import Agent

try:
    # When loaded as part of the weather_agent package (ADK from parent dir)
    from ..tools.email_client import send_weather_alert
except ImportError:
    # When run standalone from inside the project directory
    from tools.email_client import send_weather_alert  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

# ── Tool function exposed to the sub-agent ───────────────────────

def _alert_tool(
    to_email: str,
    city: str,
    temperature_f: float | None = None,
    rain_probability: float = 0.0,
    rain_amount_inch: float = 0.0,
    rain_likely: bool = False,
) -> dict:
    """Send a weather-alert email to a recipient.

    Args:
        to_email: Recipient email address (e.g. 'steve@megansoft.com').
        city: Human-readable city name (e.g. 'Novi, MI').
        temperature_f: Current temperature in Fahrenheit.
        rain_probability: Precipitation probability 0-100.
        rain_amount_inch: Forecast precipitation in inches.
        rain_likely: Whether rain is expected this hour.

    Returns:
        dict with keys: status, recipient, subject.
    """
    return send_weather_alert(
        to_email=to_email,
        city=city,
        temperature=temperature_f,
        rain_prob=rain_probability,
        rain_amount=rain_amount_inch,
        rain_likely=rain_likely,
    )


# ── Sub-agent definition ─────────────────────────────────────────

alert_agent = Agent(
    model="gemini-2.5-flash",
    name="AlertAgent",
    description="Sends weather-alert emails to recipients via Gmail SMTP.",
    instruction=(
        "You are an email notification specialist. "
        "When given weather data and a recipient email address, "
        "use the send_weather_alert_email tool to dispatch an alert. "
        "Always confirm the email was sent successfully (or note dry-run mode). "
        "If the recipient email looks invalid, let the user know."
    ),
    tools=[_alert_tool],
)
