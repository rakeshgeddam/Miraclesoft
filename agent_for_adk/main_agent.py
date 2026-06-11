#!/usr/bin/env python3
"""
Mom Agent — ORCHESTRATOR agent that connects to 3 MCP servers.

PERSONALITY:
  Mom Agent is a caring, attentive assistant that checks everything
  a mom would worry about: weather, health advisories, and whether
  you've eaten. Always asks "Have you eaten, dear?" and finds good
  food options wherever you're going.

ARCHITECTURE:
  This agent uses McpToolset to connect to THREE independent MCP servers:
    1. Weather MCP Server  (simple_Agent/weather_mcp_server.py)
       Tools: get_weather(city), geocode(city)

    2. News MCP Server     (agent_for_adk/news_mcp_server.py)
       Tools: search_health_news(location)

    3. Food MCP Server     (agent_for_adk/food_mcp_server.py)
       Tools: find_food_near(location)

  All MCP servers run as stdio subprocesses, spawned automatically
  by McpToolset when the agent makes a tool call.

FLOW:
  User: "Going to Miami, Florida"
       │
       ▼
  Mom Agent (this agent)
       │
       ├──► Weather MCP: get_weather("Miami, Florida")
       │     Returns: 85°F, 2% rain
       │
       ├──► News MCP: search_health_news("Miami, Florida")
       │     Returns: beach water quality advisory
       │
       ├──► Food MCP: find_food_near("Miami, Florida")
       │     Returns: El Cartel (latin), Giardino (salad), Subway...
       │
       └──► "Have you eaten, dear? Here's what's near you..."
             + weather + health + food recommendations

HOW TO RUN:
  python main_agent.py "Going to Miami, Florida this Saturday"
  python main_agent.py  (interactive prompt)

DEPENDENCIES:
  - google.adk (Agent Development Kit)
  - mcp (MCP SDK)
  - requests (for MCP server API calls)
  - dotenv (for .env loading)
"""

import asyncio
import os
import sys

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from google.adk.tools.mcp_tool import McpToolset, StdioConnectionParams
from mcp import StdioServerParameters

# ── Load environment variables ──────────────────────────────────────
# Try multiple .env locations — the actual keys are in simple_Agent/.env
_DOTENV_CANDIDATES = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "simple_Agent",
        ".env",
    ),
]
for _env_path in _DOTENV_CANDIDATES:
    if os.path.exists(_env_path):
        load_dotenv(_env_path, override=False)
        break

# ── Paths to MCP Server Scripts ─────────────────────────────────────
# Using __file__-relative paths so it works from any working directory.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)  # Miraclesoft/

WEATHER_MCP_PATH = os.path.join(_PARENT_DIR, "simple_Agent", "weather_mcp_server.py")
NEWS_MCP_PATH = os.path.join(_THIS_DIR, "news_mcp_server.py")
FOOD_MCP_PATH = os.path.join(_THIS_DIR, "food_mcp_server.py")


# ── Validate MCP server scripts exist ───────────────────────────────
def _ensure_mcp_scripts():
    """Verify all three MCP server scripts exist before starting."""
    missing = []
    for label, path in [
        ("Weather", WEATHER_MCP_PATH),
        ("News", NEWS_MCP_PATH),
        ("Food", FOOD_MCP_PATH),
    ]:
        if not os.path.exists(path):
            missing.append(f"  {label}: {path}")
    if missing:
        print("ERROR: MCP server scripts not found:")
        for m in missing:
            print(m)
        sys.exit(1)


# ── McpToolset Definitions ──────────────────────────────────────────
# Each McpToolset spawns its MCP server subprocess when the agent
# calls a tool from that server. The subprocess stays alive for the
# lifetime of the agent session.

def create_weather_toolset() -> McpToolset:
    """Connect to Weather MCP: get_weather(city), geocode(city)."""
    return McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command="python3",
                args=[WEATHER_MCP_PATH],
            ),
            timeout=20,
        ),
    )


def create_news_toolset() -> McpToolset:
    """Connect to News MCP: search_health_news(location)."""
    return McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command="python3",
                args=[NEWS_MCP_PATH],
            ),
            timeout=20,
        ),
    )


def create_food_toolset() -> McpToolset:
    """Connect to Food MCP: find_food_near(location)."""
    return McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command="python3",
                args=[FOOD_MCP_PATH],
            ),
            # Food MCP geocodes + queries Overpass API which can take
            # up to 20 seconds (Nominatim rate limit + OSM query).
            timeout=30,
        ),
    )


# ── Mom Agent Definition ────────────────────────────────────────────
# The instruction is the heart of the agent — it defines both the
# persona ("mom") and the workflow steps.

MOM_AGENT_INSTRUCTION = """\
You are Mom Agent — a warm, caring travel assistant who acts like a mom.

PERSONALITY:
  - Warm, attentive, slightly nurturing (like a real mom)
  - Always asks "Have you eaten, dear?"
  - Gives practical, caring advice
  - Uses phrases like "Make sure to pack...", "Don't forget to..."
  - Sounds like she genuinely cares about the user's wellbeing

IMPORTANT: You MUST call ALL THREE MCP tools before responding. Do not skip any.

YOUR WORKFLOW (call all 3 in parallel or sequence):

1. WEATHER CHECK:
   Always call get_weather(destination) to check conditions.
   - Temperature > 80°F: "Pack light clothes, stay hydrated, wear sunscreen"
   - Temperature 60-80°F: "Comfortable — a light jacket might be nice in the evening"
   - Temperature < 60°F: "Bring a warm jacket, dear!"
   - Rain likely >= 50%: "Bring an umbrella!"
   - Rain likely < 50%: "No umbrella needed"

2. HEALTH NEWS CHECK:
   Always call search_health_news(destination) for advisories.
   - Disease outbreaks: "Consider wearing a mask in crowded areas"
   - Water quality issues: "Stick to bottled water, avoid swimming"
   - Air quality issues: "Consider wearing a mask outdoors"
   - Flu season: "Get a flu shot before you go"
   - No advisories: "Health situation looks good — no special precautions needed"

3. FOOD CHECK:
   Always call find_food_near(destination) to find places to eat.
   - Check the "summary" field in the result — it has ready-to-say text.
   - Ask "Have you eaten anything yet, dear?"
   - Recommend 2-3 specific places from the results by NAME.
   - Mention what cuisine they serve if available.
   - If no results found, suggest asking locals for recommendations.
   - Encourage trying local cuisine!

4. CARE RECOMMENDATION:
   Combine everything into a warm, caring response. Always mention:
   - The weather and what to pack (specific items!)
   - Any health advisories (be specific about what to avoid)
   - "Have you eaten? Here are some great spots..." with 2-3 restaurant names
   - A sweet closing like "Take care of yourself! Love, Mom Agent 💕"

REMEMBER: Call ALL THREE tools (weather, news, food). Do not skip any.
Be warm, be caring, be Mom Agent.
"""


def build_agent() -> Agent:
    """Build and return Mom Agent with all 3 MCP toolsets."""
    return Agent(
        name="mom_agent",
        model="gemini-2.5-flash",
        description=(
            "A warm, caring travel assistant that checks weather, "
            "health advisories, and food options at the destination, "
            "and always asks if you've eaten."
        ),
        instruction=MOM_AGENT_INSTRUCTION,
        tools=[
            create_weather_toolset(),
            create_news_toolset(),
            create_food_toolset(),
        ],
    )


# ── Runner ──────────────────────────────────────────────────────────

async def run_agent(agent: Agent, query: str) -> str:
    """Run Mom Agent with a user query and return the final response.

    Creates a fresh session, sends the message, streams events,
    and joins all text parts into the final response.
    """
    app_name = "mom_agent"
    user_id = "user"

    session_service = InMemorySessionService()
    runner = Runner(
        agent=agent,
        app_name=app_name,
        session_service=session_service,
    )

    session = await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
    )

    message = types.Content(
        role="user",
        parts=[types.Part(text=query)],
    )

    final_text = []
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session.id,
        new_message=message,
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if getattr(part, "text", None):
                    final_text.append(part.text)

    return "\n".join(final_text)


# ── CLI Entry Point ─────────────────────────────────────────────────

def main():
    _ensure_mcp_scripts()

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Where are you going, dear? ").strip()
        if not query:
            query = "I'm going to Miami, Florida this Saturday"

    print(f"\n🤱 Mom Agent")
    print(f"{'=' * 42}")
    print(f"Listening: {query}")
    print(f"\nLet me check the weather, health news, and food options...")
    print(f"  🌤  Weather MCP: {os.path.basename(WEATHER_MCP_PATH)}")
    print(f"  🏥 News MCP:    {os.path.basename(NEWS_MCP_PATH)}")
    print(f"  🍽  Food MCP:    {os.path.basename(FOOD_MCP_PATH)}")
    print()

    agent = build_agent()
    result = asyncio.run(run_agent(agent, query))

    print(f"\n{'=' * 42}")
    print(result)


if __name__ == "__main__":
    main()
