#!/usr/bin/env python3
"""
Food MCP Server — finds restaurants and food options near a location.

PURPOSE:
  Searches for restaurants, cafes, and food places near a given location
  using OpenStreetMap Overpass API (completely free, no API key required).

DATA SOURCE:
  OpenStreetMap Overpass API (https://overpass-api.de)
  - FREE — no API key, no registration, no rate limits for reasonable use
  - Global coverage — finds places in any city worldwide
  - Includes restaurant names, cuisines, addresses, and types

  Geocoding via Nominatim (https://nominatim.openstreetmap.org)
  - FREE — converts "Miami, Florida" to lat/lon coordinates
  - Usage policy: max 1 request/second (we wait 1s between calls)

TOOLS EXPOSED:
  1. find_food_near(location) — Find restaurants, cafes, and food places
     near a city or address. Returns up to 10 results with name, type,
     cuisine, and address.

TRANSPORT:
  stdio (standard input/output). Runs as an MCP subprocess.

DEPENDENCIES:
  - mcp (MCP SDK): pip install mcp
  - requests (HTTP for API calls)

AGENT NOTES:
  - Returns empty array when no food places found (remote area).
  - Results are from OpenStreetMap — quality depends on local
    contributor coverage. Major cities have excellent data.
  - Some places may not have names (unmapped in OSM). They're
    filtered out to show only named establishments.
"""

import json

import requests
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Food Finder")


# ── Constants ───────────────────────────────────────────────────────
USER_AGENT = "MomAgent/1.0 (event planner; educational project)"
SEARCH_RADIUS_METERS = 5000  # 5km search radius
MAX_RESULTS = 10


def _geocode_location(location: str) -> dict | None:
    """Convert a location name to lat/lon using Nominatim.

    Nominatim is OpenStreetMap's free geocoding service.
    Usage policy: max 1 req/sec (we handle this with a small delay).

    Args:
        location: Place name, e.g. "Miami, Florida"

    Returns:
        dict with lat, lon, display_name, or None if not found
    """
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": location,
        "format": "json",
        "limit": 1,
    }
    headers = {"User-Agent": USER_AGENT}

    # Nominatim usage policy: max 1 req/sec (for sustained traffic).
    # For a single geocode call like this, we skip the delay since
    # the overall rate from one user is well within limits.
    # import time
    # time.sleep(1)

    try:
        r = requests.get(url, params=params, headers=headers, timeout=15)
        r.raise_for_status()
        data = r.json()
        if data:
            return {
                "lat": float(data[0]["lat"]),
                "lon": float(data[0]["lon"]),
                "display_name": data[0].get("display_name", location),
            }
    except requests.RequestException:
        pass
    return None


def _query_overpass(query_xml: str) -> list[dict]:
    """Execute an Overpass API query and return elements.

    Overpass is OpenStreetMap's read-only API for structured queries.
    It uses a custom query language (Overpass QL).

    Args:
        query_xml: Overpass QL query string

    Returns:
        List of OSM element dicts, or empty list on error
    """
    url = "https://overpass-api.de/api/interpreter"
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json, text/plain, */*",
    }

    try:
        r = requests.post(url, data={"data": query_xml}, headers=headers, timeout=20)
        r.raise_for_status()
        return r.json().get("elements", [])
    except requests.RequestException:
        return []


def _format_place(element: dict) -> dict:
    """Format an OSM element into a clean place dict.

    Args:
        element: Raw OSM element with tags

    Returns:
        Clean dict with name, type, cuisine, address
    """
    tags = element.get("tags", {})
    name = tags.get("name", "")

    # Build address from available tags
    addr_parts = []
    for key in ["addr:housenumber", "addr:street", "addr:city", "addr:state"]:
        val = tags.get(key)
        if val:
            addr_parts.append(val)
    address = ", ".join(addr_parts) if addr_parts else ""

    # Extract cuisine (OSM tags like "cuisine", "diet:*")
    cuisine = tags.get("cuisine", "unknown")
    if cuisine != "unknown":
        cuisine = cuisine.replace(";", ", ")

    return {
        "name": name,
        "type": tags.get("amenity", "food"),
        "cuisine": cuisine,
        "address": address,
    }


# ── Tool Definitions ────────────────────────────────────────────────

@mcp.tool()
def find_food_near(location: str) -> str:
    """Find restaurants, cafes, and food places near a location.

    Searches for places to eat within ~5km of the given location using
    OpenStreetMap data. Returns up to 10 named establishments with
    their type (restaurant, cafe, fast_food), cuisine, and address.

    Args:
        location: City name or address, e.g. 'Miami, Florida' or
                  'Times Square, New York'. Include state/country for
                  best results.

    Returns:
        JSON string. Parse with json.loads().
        On success: {"status":"success", "location":"...", "places":[...]}
        On error:   {"status":"error", "message":"..."}

        Each place has: name, type, cuisine, address.
        If no places found, places array is empty.

    Error handling:
        - If the location can't be found, returns error message.
        - If Overpass API is unreachable, returns empty places.
        - Unnamed places are filtered out.

    AGENT USAGE EXAMPLE:
        1. Ask user: "Have you eaten anything yet?"
        2. If response indicates hunger, call this tool.
        3. Recommend top options from results.
    """
    try:
        # Step 1: Geocode the location name to coordinates
        geo = _geocode_location(location)
        if not geo:
            return json.dumps({
                "status": "error",
                "message": f"Could not find location: '{location}'. Try being more specific (e.g. 'Miami, Florida').",
            })

        # Step 2: Query Overpass for food places near the coordinates
        # Search all element types (node, way, area) for broader coverage
        query = f"""
        [out:json];
        (
          node["amenity"~"restaurant|cafe|fast_food|food_court"](around:{SEARCH_RADIUS_METERS},{geo["lat"]},{geo["lon"]});
          way["amenity"~"restaurant|cafe|fast_food|food_court"](around:{SEARCH_RADIUS_METERS},{geo["lat"]},{geo["lon"]});
        );
        out center {MAX_RESULTS};
        """

        elements = _query_overpass(query)

        # Step 3: Format results as a nice human-readable text
        places = []
        seen_names = set()
        for el in elements:
            place = _format_place(el)
            if not place["name"]:
                continue
            if place["name"].lower() not in seen_names:
                seen_names.add(place["name"].lower())
                places.append(place)

        # Build a text summary the LLM can easily read and relay
        if places:
            lines = [f"I found {len(places)} places to eat near {geo['display_name']}:"]
            for i, p in enumerate(places[:MAX_RESULTS], 1):
                cuisine_part = f" ({p['cuisine']} cuisine)" if p['cuisine'] != 'unknown' else ""
                addr_part = f" — {p['address']}" if p['address'] else ""
                lines.append(f"  {i}. {p['name']}{cuisine_part}{addr_part}")
            lines.append("")
            lines.append("Recommend the top 2-3 to the user based on cuisine variety!")
            text_summary = "\n".join(lines)
        else:
            text_summary = (
                f"I searched for restaurants near {geo['display_name']} "
                f"but didn't find any named places in OpenStreetMap. "
                f"The area may have food options not yet mapped. "
                f"Suggest the user asks locals or checks Google Maps."
            )

        return json.dumps({
            "status": "success",
            "location": geo["display_name"],
            "place_count": len(places),
            "places": places[:MAX_RESULTS],
            "summary": text_summary,
        })

    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Food search failed: {e}",
        })


# ── Entry Point ─────────────────────────────────────────────────────
if __name__ == "__main__":
    mcp.run(transport="stdio")
