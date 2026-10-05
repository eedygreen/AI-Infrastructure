"""Catalog, atomic/idempotent writes, and the bounded-retry helper."""

import json
import threading
import time
from datetime import datetime
from sqlalchemy.sql import text
from typing import Optional

from lib import config
from lib.starter_utils import db_engine, paper_supplies


CATALOG = {p["item_name"]: p for p in paper_supplies}
WRITE_LOCK = threading.RLock()   # single-process writer lock (SQLite is single-writer anyway)


# ---------------------------------------------------------------------------
# Database layer: atomic, idempotent writes (the starter's create_transaction is
# not atomic with a stock check and cannot carry an idempotency key)
# ---------------------------------------------------------------------------
def ensure_runtime_tables(reset: bool = False) -> None:
    with db_engine.begin() as conn:
        if reset:
            conn.execute(text("DROP TABLE IF EXISTS idempotency_keys"))
            conn.execute(text("DROP TABLE IF EXISTS run_state"))
        conn.execute(text(
            "CREATE TABLE IF NOT EXISTS idempotency_keys ("
            "key TEXT PRIMARY KEY, kind TEXT, transaction_id INTEGER, "
            "result TEXT, created_at TEXT)"
        ))
        conn.execute(text("CREATE TABLE IF NOT EXISTS run_state (id INTEGER PRIMARY KEY, run_id TEXT)"))


def save_run_id(run_id: str) -> None:
    """Remember which run owns the idempotency keys in this database, so --resume can reuse it."""
    with db_engine.begin() as conn:
        conn.execute(text("INSERT OR REPLACE INTO run_state (id, run_id) VALUES (1, :run_id)"), {"run_id": run_id})


def load_run_id() -> Optional[str]:
    """The run id saved by save_run_id, or None (including for a database from before this existed)."""
    try:
        with db_engine.connect() as conn:
            return conn.execute(text("SELECT run_id FROM run_state WHERE id = 1")).scalar()
    except Exception:  # noqa: BLE001
        return None

def _stock_in_conn(conn, item_name: str, as_of_date: str) -> int:
    value = conn.execute(text("""
        SELECT COALESCE(SUM(CASE
            WHEN transaction_type = 'stock_orders' THEN units
            WHEN transaction_type = 'sales' THEN -units
            ELSE 0 END), 0)
        FROM transactions
        WHERE item_name = :item_name AND transaction_date <= :as_of_date
    """), {"item_name": item_name, "as_of_date": as_of_date}).scalar()
    return int(float(value or 0))

def _cash_in_conn(conn, as_of_date: str) -> float:
    """Cash on hand read THROUGH the caller's connection (sales add, stock orders subtract).

    Mirrors the starter's get_cash_balance, which opens its own connection and so cannot be used
    by a guard that must see the same transaction as its insert. A test pins the two together."""
    value = conn.execute(text("""
        SELECT COALESCE(SUM(CASE
            WHEN transaction_type = 'sales' THEN price
            WHEN transaction_type = 'stock_orders' THEN -price
            ELSE 0 END), 0)
        FROM transactions
        WHERE transaction_date <= :as_of_date
    """), {"as_of_date": as_of_date}).scalar()
    return float(value or 0.0)

def _commit_transaction(*, key: str, kind: str, item_name: str, transaction_type: str,
                        units: int, price: float, date: str, guard, meta: Optional[dict] = None) -> dict:
    """One DB transaction: replay-if-seen -> guard -> insert -> remember key.

    guard(conn) returns None to proceed, or a reason string to deny (nothing is written).
    """
    if transaction_type not in {"stock_orders", "sales"}:
        raise ValueError("transaction_type must be 'stock_orders' or 'sales'")
    with WRITE_LOCK, db_engine.begin() as conn:
        previous = conn.execute(
            text("SELECT result FROM idempotency_keys WHERE key = :k"), {"k": key}
        ).scalar()
        if previous is not None:
            result = json.loads(previous)
            result["replayed"] = True
            return result
        denial = guard(conn)
        if denial:
            return {"status": "denied", "reason": denial}
        inserted = conn.execute(text(
            "INSERT INTO transactions (item_name, transaction_type, units, price, transaction_date) "
            "VALUES (:item_name, :transaction_type, :units, :price, :date)"
        ), {"item_name": item_name, "transaction_type": transaction_type,
            "units": int(units), "price": float(price), "date": date})
        result = {"status": "committed", "transaction_id": int(inserted.lastrowid), **(meta or {})}
        conn.execute(text(
            "INSERT INTO idempotency_keys (key, kind, transaction_id, result, created_at) "
            "VALUES (:k, :kind, :tx, :result, :created)"
        ), {"k": key, "kind": kind, "tx": result["transaction_id"],
            "result": json.dumps(result), "created": datetime.now().isoformat()})
        return result

def _retry(fn, attempts: Optional[int] = None, base: Optional[float] = None):
    """Bounded retry with exponential backoff. Reads only, or idempotent writes."""
    attempts = config.DB_RETRIES if attempts is None else attempts   # read at call time so knobs are tunable
    base = config.BACKOFF_S if base is None else base
    last = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - deliberately broad at an I/O boundary
            last = exc
            if i < attempts - 1:
                time.sleep(base * (2 ** i))
    raise last
