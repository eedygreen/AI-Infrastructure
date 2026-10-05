"""Policy knobs (business decisions, not code decisions: tune freely).

Read them as `config.NAME` in code. Never `from lib.config import NAME`: that copies the value at
import time, so tuning at runtime (and the tests) would silently stop working."""

import dotenv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_env() -> None:
    """Load .env from the project root (the folder that holds main.py), whatever the current directory is."""
    dotenv.load_dotenv(dotenv_path=PROJECT_ROOT / ".env")


# ---------------------------------------------------------------------------
# Policy knobs (business decisions, not code decisions: tune freely)
# ---------------------------------------------------------------------------
MAX_ITERATIONS = 3             # bound on the customer loop
AGENT_MAX_STEPS = 8            # bound on LLM steps per specialist run
AGENT_TIMEOUT_S = 120          # stop waiting on a specialist after this long
AGENT_RETRIES = 1              # extra specialist attempts (only for missing ledger entries)
DB_RETRIES = 3                 # attempts for a DB / supplier call
BACKOFF_S = 0.3                # base backoff (doubles each attempt)

LIST_MARKUP = 0.30             # list price = unit_price * (1 + LIST_MARKUP)
MIN_MARGIN = 0.10              # never sell below cost * (1 + MIN_MARGIN)
DISCOUNT_TIERS = [(1000, 0.15), (500, 0.10), (100, 0.05), (0, 0.0)]  # (min qty, max discount)
HIGH_DEMAND_CAP_FACTOR = 0.5   # top sellers get half the discount cap
MIN_CASH_RESERVE = 5000.0      # a restock may never push cash below this
REORDER_TARGET_MULT = 5        # replenish up to this multiple of min_stock_level
ALLOW_PARTIAL_FULFILLMENT = True   # False = all-or-nothing per request
SHEETS_PER_REAM = 500
