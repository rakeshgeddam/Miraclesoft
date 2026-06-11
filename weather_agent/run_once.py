"""
One-shot CLI: check weather for a city, optionally send an email alert.

Usage:
    python run_once.py                                          # default city (Novi, MI)
    python run_once.py --city "Detroit, MI"
    python run_once.py --city "Novi, MI" --email steve@megansoft.com
    python run_once.py --city "Novi, MI" --email steve@megansoft.com --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from tools.weather_api import check_weather
from tools.email_client import send_weather_alert

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Check weather and optionally send an alert email.")
    parser.add_argument("--city", default=None, help="City name (default: from .env or Novi, MI)")
    parser.add_argument("--email", default=None, help="Send weather alert to this email address")
    parser.add_argument("--dry-run", action="store_true", default=None,
                        help="Override dry-run mode (default: from .env)")
    args = parser.parse_args()

    from config import DEFAULT_CITY, DRY_RUN

    city = args.city or DEFAULT_CITY
    dry_run = args.dry_run if args.dry_run is not None else DRY_RUN

    print(f"\n  🌤️  Checking weather for: {city}")
    print(f"  {'🔴 DRY RUN' if dry_run else '📧 LIVE MODE'}\n")

    result = await check_weather(city)

    if result.get("status") != "success":
        print(f"  ❌ Error: {result.get('message', 'Unknown error')}")
        sys.exit(1)

    loc = result["location"]
    cur = result["current_hour"]
    icon = "🌧️" if cur["rain_likely"] else "☀️"

    print(f"  {icon}  {loc['name']}{', ' + loc['admin1'] if loc.get('admin1') else ''}, {loc.get('country', '')}")
    print(f"     Time:       {cur['time']}")
    print(f"     Temp:       {cur['temperature_f']} °F")
    print(f"     Rain prob:  {cur['precipitation_probability']}%")
    print(f"     Rain amt:   {cur['precipitation_in']} in")
    print(f"     Rain?       {'YES' if cur['rain_likely'] else 'No'}")

    if args.email:
        print(f"\n  📧 Sending alert to {args.email} ...")
        email_result = send_weather_alert(
            to_email=args.email,
            city=city,
            temperature=cur["temperature_f"],
            rain_prob=cur["precipitation_probability"],
            rain_amount=cur["precipitation_in"],
            rain_likely=cur["rain_likely"],
            dry_run=dry_run,
        )
        print(f"     Status: {email_result['status']}")

    print()


if __name__ == "__main__":
    asyncio.run(main())
