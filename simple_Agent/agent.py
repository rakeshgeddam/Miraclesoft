"""
ADK Weather Agent — standalone ADK agent wrapping weather_tools functions.

PURPOSE:
  - Provides root_agent for backwards compatibility with run_hourly.py
  - For new projects, prefer weather_mcp_server.py (MCP) over this ADK agent

BACKWARD COMPATIBILITY:
  run_hourly.py imports root_agent from this file:
    from simple_Agent.agent import root_agent
  This still works because agent.py imports from weather_tools.py.

AGENT USAGE:
  # Direct ADK usage:
  from google.adk.runners import Runner
  from google.adk.sessions import InMemorySessionService
  from simple_Agent.agent import root_agent
  runner = Runner(agent=root_agent, ...)

  # Or serve via MCP (weather_mcp_server.py) for multi-agent orchestration.
"""

from simple_Agent.weather_tools import resolve_city, get_weather
from google.adk.agents import Agent

root_agent = Agent(
    name="weather_agent",
    model="gemini-2.5-flash",
    description=(
        "Reports the current hour's weather and rain likelihood "
        "for a given city."
    ),
    instruction=(
        "You are a weather assistant. When given a city, call get_weather "
        "with that city name. Report the forecast time, temperature, "
        "precipitation probability, and clearly state whether rain is "
        "likely this hour."
    ),
    tools=[get_weather],
)
