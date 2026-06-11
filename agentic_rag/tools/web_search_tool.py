"""
Web Search Tool — DuckDuckGo (free, no API key required).

PURPOSE:
  Provides web search capabilities to the agentic RAG pipeline so
  sub-agents can retrieve supplementary information from the web.

AGENT USAGE:
  from tools.web_search_tool import web_search

  results = await web_search("Dintta ERP architecture patterns")
  # Returns: {"status": "ok", "results": [...], "total": 5}

DEPENDENCIES:
  - httpx (installed with google-adk)

AGENT NOTES:
  - Uses DuckDuckGo's HTML search API (no API key needed).
  - Each result contains title, snippet, url.
  - Returns error dict on failure — LLM handles gracefully.
  - Respects a short timeout (10s) to avoid blocking the agent loop.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Optional
from xml.etree import ElementTree

import httpx

logger = logging.getLogger("web_search_tool")

_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)

_GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"


def _search_google_news(query: str, max_results: int = 5) -> list[dict]:
    """Search Google News RSS for current articles on a topic."""
    params = {"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"}
    headers = {"User-Agent": _USER_AGENT}
    try:
        r = httpx.get(_GOOGLE_NEWS_RSS, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        root = ElementTree.fromstring(r.content)
        articles = []
        for item in root.findall(".//item"):
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            snippet = (item.findtext("description") or "").strip()
            source = (item.findtext("source") or "Google News").strip()
            pub_str = (item.findtext("pubDate") or "").strip()
            articles.append({
                "title": title,
                "snippet": snippet,
                "url": link,
                "source": source,
                "date": pub_str,
            })
            if len(articles) >= max_results:
                break
        return articles
    except Exception as e:
        logger.warning("Google News RSS search failed: %s", e)
        return []


async def _search_duckduckgo(query: str, max_results: int = 5) -> list[dict]:
    """Search DuckDuckGo HTML version for web results."""
    url = "https://html.duckduckgo.com/html/"
    headers = {"User-Agent": _USER_AGENT}
    data = {"q": query}
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post(url, data=data, headers=headers)
            r.raise_for_status()
        # Parse the HTML response for result links
        from html.parser import HTMLParser

        class DDGResultParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.results = []
                self._in_result = False
                self._in_link = False
                self._current = {}
                self._tag_stack = []
                self._skip_a = False

            def handle_starttag(self, tag, attrs):
                attrs_dict = dict(attrs)
                if tag == "a" and "result__a" in attrs_dict.get("class", ""):
                    self._in_link = True
                    self._current["url"] = attrs_dict.get("href", "")
                    self._current["title"] = ""
                elif tag == "a" and self._in_result and not self._in_link:
                    if "href" in attrs_dict:
                        self._current.setdefault("url", attrs_dict["href"])
                elif tag == "div" and "result__snippet" in attrs_dict.get("class", ""):
                    self._current["snippet"] = ""

            def handle_data(self, data):
                if self._in_link:
                    self._current["title"] = (self._current.get("title", "") + data).strip()
                elif "snippet" in self._current and isinstance(self._current.get("snippet"), str):
                    self._current["snippet"] = (self._current["snippet"] + data).strip()

            def handle_endtag(self, tag):
                if tag == "a" and self._in_link:
                    self._in_link = False
                    if self._current.get("title") and self._current.get("url"):
                        self.results.append(dict(self._current))
                        self._current = {}
                    if len(self.results) >= max_results:
                        pass

        parser = DDGResultParser()
        parser.feed(r.text)
        return parser.results[:max_results]
    except Exception as e:
        logger.warning("DuckDuckGo search failed: %s", e)
        return []


async def web_search(query: str, max_results: int = 5) -> dict:
    """Search the web for current information on a topic.

    Args:
        query: Search query string.
        max_results: Max results to return (default 5).

    Returns:
        dict with keys: status, query, results (list of {title, snippet, url, source}), total
    """
    # Try DuckDuckGo first, fall back to Google News RSS
    results = await _search_duckduckgo(query, max_results)
    if not results:
        results = _search_google_news(query, max_results)

    return {
        "status": "ok" if results else "empty",
        "query": query,
        "results": [
            {
                "title": r.get("title", ""),
                "snippet": r.get("snippet", ""),
                "url": r.get("url", ""),
                "source": r.get("source", "web"),
            }
            for r in results
        ],
        "total": len(results),
    }
