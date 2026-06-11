"""
Central configuration loader.

Loads from .env via python-dotenv and exports typed constants
used throughout the project.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Google / Vertex AI ──────────────────────────────────────────
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_CLOUD_PROJECT: str = os.getenv("GOOGLE_CLOUD_PROJECT", "")
GOOGLE_CLOUD_LOCATION: str = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

# This is set so ADK knows to route through Vertex AI
_ = os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "1")

# ── LLM ──────────────────────────────────────────────────────────
LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-2.5-flash")

# ── Email (Gmail SMTP) ──────────────────────────────────────────
GMAIL_APP_PASSWORD: str = os.getenv("GMAIL_APP_PASSWORD", "")
GMAIL_SENDER: str = os.getenv("GMAIL_SENDER", "rakeshgeddam2025@gmail.com")
GMAIL_SENDER_NAME: str = os.getenv("GMAIL_SENDER_NAME", "Rakesh Geddam")
SMTP_HOST: str = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT: int = int(os.getenv("SMTP_PORT", "587"))
SEND_DELAY_SECONDS: int = int(os.getenv("SEND_DELAY_SECONDS", "8"))

# ── Safety ───────────────────────────────────────────────────────
DRY_RUN: bool = os.getenv("DRY_RUN", "true").lower() in ("true", "1", "yes")

# ── Defaults ────────────────────────────────────────────────────
DEFAULT_CITY: str = os.getenv("DEFAULT_CITY", "Novi, MI")

# ── Cache TTLs (seconds) ─────────────────────────────────────────
WEATHER_CACHE_TTL: int = int(os.getenv("WEATHER_CACHE_TTL", "300"))   # 5 min
GEO_CACHE_TTL: int = int(os.getenv("GEO_CACHE_TTL", "3600"))          # 1 hour

# ── Validation helper ────────────────────────────────────────────

MISSING: list[str] = []

if not GOOGLE_API_KEY and not GOOGLE_CLOUD_PROJECT:
    MISSING.append("GOOGLE_API_KEY or GOOGLE_CLOUD_PROJECT (Vertex AI)")
if not GMAIL_APP_PASSWORD:
    print("[config] WARNING: GMAIL_APP_PASSWORD not set — email alerts disabled")
