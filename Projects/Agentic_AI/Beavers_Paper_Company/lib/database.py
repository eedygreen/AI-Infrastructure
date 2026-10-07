"""Catalog, once-only (idempotent) writes through the starter's create_transaction, and the retry helper."""

import json
import threading
import time
from datetime import datetime
from sqlalchemy.sql import text
from typing import Optional

from lib import config
from lib.starter_utils import(
    db_engine, paper_supplies,
    create_transaction
)

CATALOG = {product["item_name"]: product for product in paper_supplies}
WRITE_LOCK = threading.RLock()   # single-process writer lock (SQLite is single-writer anyway)


# ---------------------------------------------------------------------------
# Database layer: once-only writes. The row itself is written by the starter's create_transaction;
# an idempotency key, with a "started" marker, makes a repeated write recognisable.
# ---------------------------------------------------------------------------
def ensure_runtime_tables(reset: bool = False) -> None:
    """Create the two tables this project adds to the starter database: idempotency_keys (one row per
    write) and run_state (the run id). `reset=True` empties them first. A table made before the
    before_rowid column existed is upgraded in place."""
    with db_engine.begin() as conn:
        if reset:
            conn.execute(text("DROP TABLE IF EXISTS idempotency_keys"))
            conn.execute(text("DROP TABLE IF EXISTS run_state"))
        conn.execute(text(
            "CREATE TABLE IF NOT EXISTS idempotency_keys ("
            "key TEXT PRIMARY KEY, kind TEXT, transaction_id INTEGER, "
            "result TEXT, created_at TEXT)"
        ))
        existing = {row._mapping["name"] for row in conn.execute(text("PRAGMA table_info(idempotency_keys)"))}
        if "before_rowid" not in existing:          # a database made before this column existed
            conn.execute(text("ALTER TABLE idempotency_keys ADD COLUMN before_rowid INTEGER"))  
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
    """Stock of one item as of a date, read through the caller's connection (stock orders add, sales
    subtract)."""
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
                        units: int, price: float, date: str, guard, meta: Optional[dict] = None,
                        writer=create_transaction) -> dict:
    """Record one purchase or sale exactly once. The row itself is written by the starter's create_transaction.

    The steps run under a process-wide lock, each in its own short database transaction:
      1. Look the key up.
           - finished before: return the stored result again (a replay);
           - started but never finished (a crash, or a failed write): look for the row that attempt wrote and
             adopt it, or write it now;
           - new: run guard(conn) (a reason string means "refuse", and nothing is written), then note the key
             as started, together with the last row id seen.
      2. Write the row with `writer`.
      3. Store the result against the key.
    A crash between steps 2 and 3 therefore cannot make a repeat write the row twice.
    """
    if transaction_type not in {"stock_orders", "sales"}:
        raise ValueError("transaction_type must be 'stock_orders' or 'sales'")
    with WRITE_LOCK:
        adopted_row = None
        with db_engine.begin() as conn:
            stored = conn.execute(
                text("SELECT result FROM idempotency_keys WHERE key = :k"), {"k": key}).scalar()
            if stored is not None:
                result = json.loads(stored)
                result["replayed"] = True
                return result
            started = conn.execute(
                text("SELECT COUNT(*) FROM idempotency_keys WHERE key = :k"), {"k": key}).scalar()
            if started:
                before = conn.execute(
                    text("SELECT before_rowid FROM idempotency_keys WHERE key = :k"), {"k": key}).scalar()
                adopted_row = conn.execute(text(
                    "SELECT MIN(rowid) FROM transactions WHERE rowid > :before AND item_name = :item "
                    "AND transaction_type = :type AND units = :units AND ABS(price - :price) < 0.005 "
                    "AND transaction_date = :date"
                ), {"before": before, "item": item_name, "type": transaction_type, "units": int(units),
                    "price": float(price), "date": date}).scalar()
            else:
                denial = guard(conn)
                if denial:
                    return {"status": "denied", "reason": denial}
                before = conn.execute(text("SELECT COALESCE(MAX(rowid), 0) FROM transactions")).scalar()
                conn.execute(text(
                    "INSERT INTO idempotency_keys (key, kind, before_rowid, created_at) "
                    "VALUES (:k, :kind, :before, :created)"
                ), {"k": key, "kind": kind, "before": before, "created": datetime.now().isoformat()})
        if adopted_row is None:
            writer(item_name=item_name, transaction_type=transaction_type, quantity=int(units),
                   price=float(price), date=date)
        with db_engine.begin() as conn:
            # create_transaction's own return value comes from last_insert_rowid() on a different
            # connection, so it can be 0. The newest row, under our lock, is the one just written.
            transaction_id = adopted_row if adopted_row is not None else conn.execute(
                text("SELECT MAX(rowid) FROM transactions")).scalar()
            result = {"status": "committed", "transaction_id": int(transaction_id), **(meta or {})}
            conn.execute(text("UPDATE idempotency_keys SET transaction_id = :tx, result = :result WHERE key = :k"),
                         {"tx": result["transaction_id"], "result": json.dumps(result), "k": key})
        if adopted_row is not None:
            result["replayed"] = True       # an earlier attempt already wrote it
        return result

def _retry(fn, attempts: Optional[int] = None, base: Optional[float] = None):
    """Bounded retry with exponential backoff. Reads only, or idempotent writes."""
    attempts = config.DB_RETRIES if attempts is None else attempts   # read at call time so knobs are tunable
    base = config.BACKOFF_S if base is None else base
    last = None
    for attempt_number in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - deliberately broad at an I/O boundary
            last = exc
            if attempt_number < attempts - 1:
                time.sleep(base * (2 ** attempt_number))
    raise last
