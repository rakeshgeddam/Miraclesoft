#!/usr/bin/env python3
"""
Weather MCP Server — exposes weather tools via Model Context Protocol.

PURPOSE:
  Takes the pure weather functions from weather_tools.py and wraps them
  as MCP tools so any MCP client can discover and call them.

TRANSPORT:
  stdio (standard input/output). The MCP protocol runs over stdin/stdout.
  Other transports (SSE, StreamableHTTP) are supported by FastMCP but
  stdio is simplest for inter-process communication with ADK agents.

USAGE (standalone):
  python weather_mcp_server.py
  # Listens on stdio — connect via any MCP client

USAGE (by another agent via McpToolset):
  from google.adk.tools.mcp_tool import McpToolset
  from mcp import StdioServerParameters

  toolset = McpToolset(
      connection_params=StdioConnectionParams(
          server_params=StdioServerParameters(
              command="python3",
              args=["/path/to/weather_mcp_server.py"]
          )
      )
  )

TOOLS EXPOSED:
  1. get_weather(city) — Get current hour weather + rain likelihood
  2. geocode(city)     — Resolve city name to coordinates (raw)

DEPENDENCIES:
  - mcp (MCP SDK): pip install mcp
  - weather_tools.py (in same directory)
  - requests (for Open-Meteo API)

AGENT NOTES:
  - Returns JSON strings (MCP standard). Parse with json.loads().
  - All tools are idempotent — no side effects.
  - Open-Meteo API is free, no API key needed.
"""

import json
import os
import sys

# Ensure weather_tools.py is importable regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp.server.fastmcp import FastMCP
from weather_tools import resolve_city, get_weather as _get_weather

# ── MCP Server Setup ────────────────────────────────────────────────
# FastMCP handles all MCP protocol details (discovery, calling, errors).
# The name "Weather Tools" is what MCP clients see in their tool lists.
mcp = FastMCP("Weather Tools")


# ── Tool Definitions ────────────────────────────────────────────────
# Each @mcp.tool() decorator registers the function as an MCP tool.
# The docstring becomes the tool's description — MCP clients use this
# to understand when to call the tool. Be descriptive.

@mcp.tool()
def get_weather(city: str) -> str:
    """Get the current hour's weather, temperature, and rain likelihood for a city.

    Combines geocoding and forecast in one step. Returns the temperature
    in Fahrenheit, precipitation probability, and whether rain is likely
    for the current hour in the city's local timezone.

    Args:
        city: City name, optionally with state/country.
              Examples: 'Novi, MI', 'Miami, Florida', 'Paris, France'

    Returns:
        JSON string. Parse with json.loads().
        On success: {"status":"success", "location":{...}, "current_hour":{...}}
        On error:   {"status":"error", "message":"..."}

    Error handling:
        - If the city is not found, returns error with message.
        - If the forecast API fails, raises HTTPError (caught by MCP framework).
    """
    result = _get_weather(city)
    return json.dumps(result)


@mcp.tool()
def geocode(city: str) -> str:
    """Resolve a city name to geographic coordinates (latitude, longitude, timezone).

    Useful when you need raw coordinates for mapping or other calculations.
    For a combined weather result, use get_weather() instead.

    Args:
        city: City name, e.g. 'Miami, Florida' or 'Tokyo, Japan'

    Returns:
        JSON string. Parse with json.loads().
        On success: {"status":"success", "name":"...", "latitude":..., "longitude":..., "timezone":"..."}
        On error:   {"status":"error", "message":"..."}
    """
    result = resolve_city(city)
    return json.dumps(result)


# ── Entry Point ─────────────────────────────────────────────────────
# transport="stdio" means communicate over stdin/stdout.
# This is the default transport for McpToolset in ADK.
if __name__ == "__main__":
    mcp.run(transport="stdio")
