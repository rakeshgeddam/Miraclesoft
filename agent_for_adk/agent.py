"""
Mom Agent — ADK Web serving layer.

PURPOSE:
  This file exports the Mom Agent for the ADK Web UI. When you run
  `adk web <dir>` it discovers this agent and serves an interactive
  chat UI at http://localhost:8000.

ARCHITECTURE:
  Mom Agent (LlmAgent) with 3 MCP toolsets — all called IN PARALLEL:
    ├── Weather MCP  → get_weather(city), geocode(city)
    ├── News MCP     → search_health_news(location)
    └── Food MCP     → find_food_near(location)

  The LLM calls all three tools in a single response (parallel function
  calls), then synthesizes the results with a warm Mom persona.

RUNNING:
  cd /Users/rakeshgeddam/Documents/Miraclesoft
  GOOGLE_GENAI_USE_VERTEXAI=1 GOOGLE_CLOUD_PROJECT=round-catfish-493721-b4 adk web

DEPENDENCIES:
  - google.adk (Agent Development Kit)
  - mcp (Model Context Protocol SDK)
  - requests, dotenv
"""

import os

from dotenv import load_dotenv
from google.adk.agents import Agent as LlmAgent

from google.adk.tools.mcp_tool import McpToolset, StdioConnectionParams
from mcp import StdioServerParameters


# ── Resolve paths ──────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)  # Miraclesoft/

WEATHER_MCP_PATH = os.path.join(_PARENT_DIR, "simple_Agent", "weather_mcp_server.py")
NEWS_MCP_PATH = os.path.join(_THIS_DIR, "news_mcp_server.py")
FOOD_MCP_PATH = os.path.join(_THIS_DIR, "food_mcp_server.py")

# Load env (for Vertex AI auth) — must happen BEFORE importing ADK
for candidate in [
    os.path.join(_THIS_DIR, ".env"),
    os.path.join(_PARENT_DIR, "simple_Agent", ".env"),
]:
    if os.path.exists(candidate):
        load_dotenv(candidate, override=False)
        break


# ── Create MCP Toolsets (eager) ────────────────────────────────────
# Each spawns its own MCP server subprocess.

weather_tools = [
    McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command="python3",
                args=[WEATHER_MCP_PATH],
            ),
            timeout=20,
        )
    )
]

news_tools = [
    McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command="python3",
                args=[NEWS_MCP_PATH],
            ),
            timeout=20,
        )
    )
]

food_tools = [
    McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command="python3",
                args=[FOOD_MCP_PATH],
            ),
            timeout=30,
        )
    )
]


# ── Mom Agent instruction ──────────────────────────────────────────

MOM_INSTRUCTION = """\
You are Mom Agent — a warm, caring assistant who helps people prepare for events and trips.

PERSONALITY:
  - Warm, attentive, nurturing (like a real mom)
  - Always asks "Have you eaten, dear?" and recommends food by name
  - Gives practical, specific advice ("pack a raincoat!", "bring sunscreen!")
  - Uses warm language: "Make sure to...", "Don't forget to..."
  - Signs off with "Love, Mom Agent 💕"

AVAILABLE TOOLS — you have 3 tools from separate MCP servers.
You MUST call ALL THREE in PARALLEL (same LLM response, never skip any):
  1. get_weather(city)     — real-time temperature, rain, conditions
  2. search_health_news(location) — real health advisories & disease news
  3. find_food_near(location)     — real restaurant recommendations near the destination

CRITICAL RULES:
  - Call ALL THREE tools in the SAME response (parallel) — don't wait for one to finish
  - DO NOT answer from your training knowledge — use the tool results
  - Never say "I can't access real-time data" — you have the tools, use them
  - Ask "Have you eaten, dear?" before giving food recommendations
  - Synthesize all three data sources into one cohesive caring response

YOUR RESPONSE MUST INCLUDE:
  ✅ Weather: exact temperature and conditions from tool, what to pack
  ✅ Health: real advisories from tool, specific precautions (mask? bottled water?)
  ✅ Food: "Have you eaten?" + 2-3 specific restaurant names from the results
  ✅ Closing: "Take care of yourself! Love, Mom Agent 💕"
"""


# ── Build and Export the Agent ─────────────────────────────────────

agent = LlmAgent(
    name="mom_agent",
    model="gemini-2.5-flash",
    description=(
        "A warm, caring travel assistant that checks weather, "
        "health advisories, and food options at any destination, "
        "and always asks if you've eaten."
    ),
    instruction=MOM_INSTRUCTION,
    tools=weather_tools + news_tools + food_tools,
)

# ADK web discovers `root_agent` — this is the required export name.
root_agent = agent

__all__ = ["agent", "root_agent"]
