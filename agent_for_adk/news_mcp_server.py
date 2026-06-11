#!/usr/bin/env python3
"""
News MCP Server — exposes health news search as MCP tools.

PURPOSE:
  Searches for recent health advisories, disease outbreaks, and
  concerning news in a given geographic area using Google News RSS.

  Designed to be used alongside weather_mcp_server.py by the main
  event planner agent (agent_for_adk/main_agent.py).

DATA SOURCE:
  Google News RSS (https://news.google.com/rss/search)
  - FREE — no API key required
  - No rate limits for reasonable usage
  - Returns results from major news sources worldwide

TRANSPORT:
  stdio (standard input/output). Runs as an MCP subprocess.

TOOLS EXPOSED:
  1. search_health_news(location) — Search for recent health/disease news
     in a location. Returns articles from the last 30 days.

DEPENDENCIES:
  - mcp (MCP SDK): pip install mcp
  - requests (stdlib: HTTP)
  - xml.etree.ElementTree (stdlib: RSS/XML parsing)
  - email.utils (stdlib: RFC 2822 date parsing)

AGENT NOTES:
  - Returns empty articles array when nothing relevant is found.
  - Google News may return articles in any language (filtered client-side).
  - The tool searches multiple keyword combinations internally for
    better coverage.
"""

import json
import os
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote

import requests

from mcp.server.fastmcp import FastMCP

# ── MCP Server Setup ────────────────────────────────────────────────
mcp = FastMCP("Health News Search")

# ── Constants ───────────────────────────────────────────────────────
# Multiple search queries for comprehensive coverage
SEARCH_QUERIES = [
    "disease outbreak {location}",
    "health advisory {location}",
    "{location} virus flu season",
    "{location} public health alert",
]

# Google News RSS base URL
GNEWS_RSS_URL = "https://news.google.com/rss/search"

# Look back window: articles older than this are excluded
MAX_ARTICLE_AGE_DAYS = 30


def _parse_rss_date(date_str: str) -> datetime | None:
    """Parse RSS date string (RFC 2822) to datetime.

    Google News RSS uses format like:
      'Mon, 01 Jun 2026 14:30:00 GMT'

    Returns None if parsing fails.
    """
    try:
        return parsedate_to_datetime(date_str)
    except (ValueError, TypeError):
        return None


def _search_google_news(query: str) -> list[dict]:
    """Search Google News RSS and return parsed articles.

    Args:
        query: URL-encoded search string

    Returns:
        List of article dicts with keys: title, source, date, snippet, url

    HOW IT WORKS:
        1. Constructs Google News RSS URL with the query
        2. GET request to Google News (no auth needed)
        3. Parses XML RSS feed with ElementTree
        4. Filters to last MAX_ARTICLE_AGE_DAYS
        5. Deduplicates by title (basic)
    """
    params = {
        "q": query,
        "hl": "en-US",
        "gl": "US",
        "ceid": "US:en",
    }
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36"
        )
    }

    try:
        r = requests.get(GNEWS_RSS_URL, params=params, headers=headers, timeout=15)
        r.raise_for_status()
    except requests.RequestException:
        return []

    try:
        root = ET.fromstring(r.content)
    except ET.ParseError:
        return []

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=MAX_ARTICLE_AGE_DAYS)
    articles = []
    seen_titles = set()

    # RSS items are in <channel><item>...</item>
    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        link = item.findtext("link") or ""
        pub_date_str = (item.findtext("pubDate") or "").strip()
        source = (item.findtext("source") or "Google News").strip()
        description = (item.findtext("description") or "").strip()

        # Skip empty titles (duplicate detection)
        if not title or title in seen_titles:
            continue
        seen_titles.add(title)

        # Filter by date — skip articles outside the lookback window
        pub_date = _parse_rss_date(pub_date_str)
        if pub_date and pub_date < cutoff:
            continue

        # Strip HTML tags from description for a clean snippet
        import re as _re
        clean_snippet = _re.sub(r"<[^>]+>", "", description)[:300]

        articles.append({
            "title": title,
            "source": source,
            "date": pub_date_str,
            "snippet": clean_snippet,
            "url": link,
        })

    return articles


# ── Tool Definitions ────────────────────────────────────────────────

@mcp.tool()
def search_health_news(location: str) -> str:
    """Search for recent health advisories, disease outbreaks, and concerning news in a location.

    Use this to check if there are any active disease outbreaks, health
    advisories, flu season warnings, or public health alerts in an area
    within the last 30 days.

    Args:
        location: Geographic area to search, e.g. 'Miami, Florida' or
                  'Wayne County, Michigan'. Be specific — include state
                  or region for best results.

    Returns:
        JSON string. Parse with json.loads().
        On success: {"status":"success", "location":"...", "articles":[...]}
        On error:   {"status":"error", "message":"..."}

        Each article has: title, source, date, snippet, url.
        If no articles found, articles array is empty.

    Error handling:
        - If Google News is unreachable, returns empty articles with a warning.
        - Invalid locations return empty results (not an error — Google News
          does its own geolocation of content).
    """
    try:
        all_articles = []
        seen = set()

        # Search multiple keyword combinations for comprehensive coverage
        for template in SEARCH_QUERIES:
            query = template.format(location=location)
            articles = _search_google_news(query)
            for article in articles:
                # Deduplicate across search queries
                key = (article["title"], article["url"])
                if key not in seen:
                    seen.add(key)
                    all_articles.append(article)

        # Sort by date (newest first), limit to 15 results
        def _sort_key(a):
            d = _parse_rss_date(a["date"])
            return d.timestamp() if d else 0
        all_articles.sort(key=_sort_key, reverse=True)

        result = {
            "status": "success",
            "location": location,
            "articles": all_articles[:15],
        }
        return json.dumps(result)

    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Health news search failed: {e}",
        })


# ── Entry Point ─────────────────────────────────────────────────────
if __name__ == "__main__":
    mcp.run(transport="stdio")
