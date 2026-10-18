"""All knobs in one place, overridable by env vars (Cloud Run sets these)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Local convenience: read .env if python-dotenv is installed (dev only). In Cloud Run there is
# no .env; the key comes from Secret Manager. Real environment variables always win.
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
except ImportError:
    pass

# Model IDs change often: https://ai.google.dev/gemini-api/docs/models
EXTRACT_MODEL = os.getenv("EXTRACT_MODEL", "gemini-3.5-flash-lite")
MATCH_MODEL = os.getenv("MATCH_MODEL", "gemini-3.5-flash-lite")
VERIFY_MODEL = os.getenv("VERIFY_MODEL", "gemini-3.8-flash")  # stronger model as the judge
SPLIT_MODEL = os.getenv("SPLIT_MODEL", "gemini-3.5-flash-lite")
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))  # Firestore vector indexes cap at 2048 dims

# Thinking tokens are billed as output and dominate cost. LOW keeps quality for these
# structured tasks. Set to "" if a model rejects the setting.
THINKING_LEVEL = os.getenv("THINKING_LEVEL", "LOW")
VERIFY_THINKING_LEVEL = os.getenv("VERIFY_THINKING_LEVEL", "MEDIUM")

VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "lancedb")  # lancedb | firestore
LANCEDB_PATH = os.getenv("LANCEDB_PATH", str(ROOT / ".lancedb"))
FIRESTORE_COLLECTION = os.getenv("FIRESTORE_COLLECTION", "resume_bullets")
SESSIONS_COLLECTION = os.getenv("SESSIONS_COLLECTION", "sessions")
TOP_K = int(os.getenv("TOP_K", "4"))
MAX_SEARCHES = int(os.getenv("MAX_SEARCHES", "6"))      # ReAct tool calls per check
MAX_LLM_CALLS = int(os.getenv("MAX_LLM_CALLS", "16"))    # hard stop for one check

PROFILE_PATH = os.getenv("PROFILE_PATH", str(ROOT / "data" / "profile.yaml"))
BULLETS_PATH = os.getenv("BULLETS_PATH", str(ROOT / "data" / "bullets.yaml"))

# Visitors' resumes
OWNER_SESSION = "owner"     # your real resume: only reachable with your owner or share link
SAMPLE_SESSION = "sample"   # fictional resume behind the public "Sample resume" button
SAMPLE_BULLETS_PATH = os.getenv("SAMPLE_BULLETS_PATH", str(ROOT / "data" / "sample_bullets.yaml"))
SAMPLE_PROFILE_PATH = os.getenv("SAMPLE_PROFILE_PATH", str(ROOT / "data" / "sample_profile.yaml"))
SESSION_TTL_HOURS = int(os.getenv("SESSION_TTL_HOURS", "24"))
MAX_RESUME_CHARS = int(os.getenv("MAX_RESUME_CHARS", "15000"))
MAX_BULLETS = int(os.getenv("MAX_BULLETS", "60"))

# Spending guard for public traffic, in USD per Pacific-time day, using measured cost.
# Your owner token is never blocked, but its spend still counts toward the total.
DAILY_BUDGET_USD = float(os.getenv("DAILY_BUDGET_USD", "0.50"))

# USD per 1M tokens: (input, output incl. thinking). Paid-tier Standard prices,
# checked Oct 2026 at https://ai.google.dev/gemini-api/docs/pricing .
# The 3.8 Flash promo price ends Dec 31, 2026 (doubles to 1.50 / 7.50).
PRICES: dict[str, tuple[float, float] | None] = {
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.8-flash": (0.75, 3.75),
    "gemini-3.6-flash": (0.75, 3.75),
    "gemini-embedding-001": (0.15, 0.0),  # verify on the pricing page
}
