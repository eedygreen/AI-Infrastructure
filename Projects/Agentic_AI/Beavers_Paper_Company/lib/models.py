"""The per-request ledger and the parsed-request types."""

import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from utils import logger


# ---------------------------------------------------------------------------
# Data model: the per-request ledger
# ---------------------------------------------------------------------------
@dataclass
class LineItem:
    """One product the customer wants, with its quantity."""
    item_name: str
    quantity: int


@dataclass
class ParsedRequest:
    """What the customer asked for, after parsing and validation.

    The flags say what could not be understood (unmatched products, an unconfirmed deadline, an unclear
    quantity, a failed parse), so the reply can ask the customer instead of guessing."""
    raw: str
    needed_by: Optional[str]
    intents: List[str]
    lines: List[LineItem]
    unmatched: List[str]
    notes: str = ""
    deadline_unconfirmed: Optional[str] = None     # a deadline before the request date: flagged, never guessed
    failed: bool = False                           # parsing itself failed (system problem): never blame the customer
    unclear_quantity: List[str] = field(default_factory=list)   # items whose quantity is not in the customer's own words: we ask, never guess

    def record_order(self, item_name: str, order: dict) -> dict:
        """Store the order result for an item and return what is on record.

        A confirmation already on record is never replaced by a repeat of the same sale or by a failure. A model
        can call the sale tool twice for one item. The second call is recognised as a repeat and must not turn
        a new order into "already placed"."""
        with self._lock:
            current = self.orders.get(item_name)
            if current and current["status"] == "confirmed" and (order["status"] != "confirmed" or order.get("replayed")):
                return current
            self.orders[item_name] = order
            return order

@dataclass
class RequestContext:
    """Everything known about one request: the parsed request, the results of each phase, and the log.

    Tools write into it and the orchestrator reads only from it, never from the model's text."""
    request_id: int
    request_date: str
    run_id: str
    iteration: int = 1
    parsed: Optional[ParsedRequest] = None
    stock: Dict[str, dict] = field(default_factory=dict)        # Phase 1 (Inventory)
    history: List[dict] = field(default_factory=list)           # Phase 1 (Quote prefetch)
    pricing_ctx: Dict[str, dict] = field(default_factory=dict)  # Phase 2 internal policy
    quotes: Dict[str, dict] = field(default_factory=dict)       # Phase 2 result
    decisions: Dict[str, dict] = field(default_factory=dict)    # Phase 3 gate (Orchestrator)
    restocks: Dict[str, dict] = field(default_factory=dict)     # Phase 3 inline restock
    orders: Dict[str, dict] = field(default_factory=dict)       # Phase 3 sales
    background: List[dict] = field(default_factory=list)        # replenishment report
    events: List[str] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def log(self, msg: str) -> None:
        """Record an event for this request: kept on the context and sent to the log."""
        line = f"[req {self.request_id}.{self.iteration}] {msg}"
        with self._lock:
            self.events.append(line)
        logger.info(line)

    def line(self, item_name: str) -> Optional[LineItem]:
        """The item with this name in the parsed request, or None."""
        if not self.parsed:
            return None
        return next((line for line in self.parsed.lines if line.item_name == item_name), None)

    def key(self, kind: str, item_name: str, quantity: int) -> str:
        # Stable across retries AND loop iterations, so a re-run replays instead of re-writing.
        """The idempotency key for one write. It is built from the run, the request, the kind of write, the
        item and the quantity, so a repeat of the same write gets the same key."""
        return f"{self.run_id}:{self.request_id}:{kind}:{item_name}:{int(quantity)}"
