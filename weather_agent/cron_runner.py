"""
Cron runner: called every hour by cron.

Always logs the weather check to stdout (captured by cron log).
Only sends an email alert when rain is likely, to avoid noise
on sunny days.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from datetime import datetime

from tools.weather_api import check_weather
from tools.email_client import send_weather_alert

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("cron")


def main() -> None:
    from config import DEFAULT_CITY, DRY_RUN

    city = DEFAULT_CITY
    email = "rakeshgeddam2025@gmail.com"

    result = asyncio.run(check_weather(city))

    if result.get("status") != "success":
        logger.error("Weather check failed: %s", result.get("message"))
        sys.exit(1)

    cur = result["current_hour"]
    loc = result["location"]
    icon = "🌧️" if cur["rain_likely"] else "☀️"

    print(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] "
        f"{icon} {loc['name']}, {loc.get('admin1','')} | "
        f"{cur['temperature_f']}°F | rain {cur['precipitation_probability']}% | "
        f"{'RAIN' if cur['rain_likely'] else 'clear'}"
    )

    # Only send email when rain is actually likely
    if cur["rain_likely"]:
        logger.info("Rain detected — sending alert to %s", email)
        email_result = send_weather_alert(
            to_email=email,
            city=city,
            temperature=cur["temperature_f"],
            rain_prob=cur["precipitation_probability"],
            rain_amount=cur["precipitation_in"],
            rain_likely=cur["rain_likely"],
            dry_run=DRY_RUN,
        )
        logger.info("Email: %s", email_result["status"])
    else:
        logger.info("No rain — no email sent")


if __name__ == "__main__":
    main()
