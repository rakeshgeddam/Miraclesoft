"""
Hourly weather scheduler for simple_Agent.

Runs the weather_agent once at the top of every hour (00:00 of each hour).
Each run checks the current hour's forecast for the configured city.

Usage:
    python simple_Agent/run_hourly.py --city "Novi, MI"

Optional flags:
    --city       City to monitor (default: "Novi, MI")
    --interval   Override interval in seconds for testing (default: align to next hour)

Examples:
    # Production: fires at the top of every real hour
    python simple_Agent/run_hourly.py --city "Detroit, MI"

    # Testing: fires every 30 seconds
    python simple_Agent/run_hourly.py --city "Novi, MI" --interval 30
"""

import argparse
import asyncio
import uuid
from datetime import datetime, timedelta

from dotenv import load_dotenv
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

load_dotenv()

# Import the agent defined in agent.py
from simple_Agent.agent import root_agent  # noqa: E402


APP_NAME = "hourly_weather_monitor"
USER_ID = "scheduler"


async def run_once(runner: Runner, session_service: InMemorySessionService, city: str) -> None:
    """Create a fresh session and run the agent once for the given city."""
    session = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID)
    message = types.Content(
        role="user",
        parts=[types.Part(text=f"What is the weather like right now in {city}?")],
    )
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Checking weather for {city}...")
    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=session.id,
        new_message=message,
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if getattr(part, "text", None):
                    print(f"[weather_agent]: {part.text}")


def seconds_until_next_hour() -> float:
    """Return the number of seconds until the top of the next hour."""
    now = datetime.now()
    next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    return (next_hour - now).total_seconds()


async def main(city: str, interval: int | None) -> None:
    session_service = InMemorySessionService()
    runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=session_service)

    print(f"Hourly weather monitor started for: {city}")
    if interval:
        print(f"Test mode: running every {interval}s")
    else:
        secs = seconds_until_next_hour()
        print(f"Production mode: first run at top of next hour (in {secs:.0f}s)")

    run_count = 0
    while True:
        # Wait until the right moment
        if interval:
            wait_secs = interval
        else:
            wait_secs = seconds_until_next_hour()

        await asyncio.sleep(wait_secs)

        run_count += 1
        await run_once(runner, session_service, city)
        print(f"  [run #{run_count} complete — next check in {'%ds' % interval if interval else '~1 hour'}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the weather agent on a schedule.")
    parser.add_argument("--city", default="Novi, MI", help="City to monitor")
    parser.add_argument(
        "--interval",
        type=int,
        default=None,
        help="Override interval in seconds (for testing). Omit for real hourly schedule.",
    )
    args = parser.parse_args()
    asyncio.run(main(city=args.city, interval=args.interval))
