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
    item_name: str
    quantity: int


@dataclass
class ParsedRequest:
    raw: str
    needed_by: Optional[str]
    intents: List[str]
    lines: List[LineItem]
    unmatched: List[str]
    notes: str = ""
    deadline_unconfirmed: Optional[str] = None     # a deadline before the request date: flagged, never guessed
    failed: bool = False                           # parsing itself failed (system problem): never blame the customer

@dataclass
class RequestContext:
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
        line = f"[req {self.request_id}.{self.iteration}] {msg}"
        with self._lock:
            self.events.append(line)
        logger.info(line)

    def line(self, item_name: str) -> Optional[LineItem]:
        if not self.parsed:
            return None
        return next((l for l in self.parsed.lines if l.item_name == item_name), None)

    def key(self, kind: str, item_name: str, quantity: int) -> str:
        # Stable across retries AND loop iterations, so a re-run replays instead of re-writing.
        return f"{self.run_id}:{self.request_id}:{kind}:{item_name}:{int(quantity)}"
