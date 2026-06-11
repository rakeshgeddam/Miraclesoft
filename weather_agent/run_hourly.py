"""
Hourly weather scheduler using APScheduler.

Runs the weather check at the top of every hour and optionally sends
email alerts when rain is detected.

Usage:
    python run_hourly.py                                          # default city, no email
    python run_hourly.py --city "Detroit" --email steve@megansoft.com
    python run_hourly.py --city "Novi, MI" --dry-run              # see what would be sent
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import datetime

from tools.weather_api import check_weather
from tools.email_client import send_weather_alert

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("scheduler")


# ── The job function ─────────────────────────────────────────────

async def weather_job(city: str, email: str | None, dry_run: bool) -> None:
    """Run one weather check and optionally send an alert."""
    logger.info("Checking weather for %s ...", city)
    result = await check_weather(city)

    if result.get("status") != "success":
        logger.error("Weather check failed: %s", result.get("message"))
        return

    cur = result["current_hour"]
    loc = result["location"]
    icon = "🌧️" if cur["rain_likely"] else "☀️"

    logger.info(
        "%s %s | %s°F | rain %d%% %s",
        icon,
        cur["time"],
        cur["temperature_f"],
        cur["precipitation_probability"],
        "(rain likely)" if cur["rain_likely"] else "",
    )

    if email and cur["rain_likely"]:
        logger.info("Rain detected — sending alert to %s ...", email)
        email_result = send_weather_alert(
            to_email=email,
            city=city,
            temperature=cur["temperature_f"],
            rain_prob=cur["precipitation_probability"],
            rain_amount=cur["precipitation_in"],
            rain_likely=cur["rain_likely"],
            dry_run=dry_run,
        )
        logger.info("Email status: %s", email_result["status"])


# ── Scheduler ────────────────────────────────────────────────────

def run_scheduler(city: str, email: str | None, dry_run: bool) -> None:
    """Run the APScheduler event loop with an hourly cron trigger."""
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
        from apscheduler.triggers.cron import CronTrigger
    except ImportError:
        logger.error(
            "apscheduler not installed. Run: pip install apscheduler"
        )
        return

    def _job_wrapper() -> None:
        """Sync wrapper for the async job function."""
        asyncio.run(weather_job(city, email, dry_run))

    scheduler = BlockingScheduler()
    trigger = CronTrigger(minute=0)  # top of every hour
    scheduler.add_job(_job_wrapper, trigger, id="weather_check")

    print(f"\n  ⏰ Hourly weather monitor started")
    print(f"     City:     {city}")
    print(f"     Email:    {email or '(none — no alerts)'}")
    print(f"     Dry run:  {dry_run}")
    print(f"     Schedule: top of every hour\n")

    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user.")


# ── CLI ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hourly weather monitor with optional email alerts.")
    parser.add_argument("--city", default=None, help="City to monitor (default: from .env)")
    parser.add_argument("--email", default=None, help="Email address for rain alerts")
    parser.add_argument("--dry-run", action="store_true", default=None,
                        help="Override dry-run mode")

    args = parser.parse_args()

    from config import DEFAULT_CITY, DRY_RUN

    city = args.city or DEFAULT_CITY
    dry_run = args.dry_run if args.dry_run is not None else DRY_RUN

    run_scheduler(city, args.email, dry_run)
