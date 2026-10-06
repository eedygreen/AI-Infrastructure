"""Pure functions (no I/O): pricing policy, fulfilment gate, parsing and validation."""

import difflib
import json
import re
from typing import Dict, List, Optional

from lib import config
from lib.database import CATALOG
from lib.models import LineItem, ParsedRequest


# ---------------------------------------------------------------------------
# Pure policy functions (no I/O: unit-testable)
# ---------------------------------------------------------------------------
def list_unit_price(unit_price: float) -> float:
    return unit_price * (1 + config.LIST_MARKUP)


def discount_cap(quantity: int, shortfall: int, high_demand: bool) -> float:
    """Max discount allowed. Stock we must buy in first is never discounted."""
    if shortfall > 0:
        return 0.0
    cap = next(rate for threshold, rate in config.DISCOUNT_TIERS if quantity >= threshold)
    return cap * config.HIGH_DEMAND_CAP_FACTOR if high_demand else cap


def apply_price_policy(unit_price: float, quantity: int, requested: float, cap: float) -> dict:
    """Clamp a requested discount to the cap and the margin floor, then price the line."""
    list_unit = list_unit_price(unit_price)
    applied = min(max(requested, 0.0), cap)
    margin_limit = max(0.0, 1 - (unit_price * (1 + config.MIN_MARGIN)) / list_unit)
    applied = min(applied, margin_limit)
    return {
        "unit_list_price": round(list_unit, 4),
        "discount": applied,
        "clamped": applied + 1e-9 < max(requested, 0.0),
        "line_total": round(quantity * list_unit * (1 - applied), 2),
    }

def decide_fulfillment(request_date: str, needed_by: Optional[str], stock: Optional[dict],
                       deadline_unconfirmed: bool = False) -> dict:
    """Phase-3 gate for one line. Returns action: ship | restock_then_ship | skip."""
    if not stock or stock.get("error"):
        return {"action": "skip", "reason": "availability_unknown"}
    if stock["shortfall"] == 0:
        return {"action": "ship", "delivery_date": request_date}
    if deadline_unconfirmed:
        return {"action": "skip", "reason": "deadline_unconfirmed", "eta": stock.get("eta")}
    eta = stock.get("eta")
    if not eta:
        return {"action": "skip", "reason": "eta_unknown"}
    if needed_by and eta > needed_by:
        return {"action": "skip", "reason": "eta_after_deadline", "eta": eta}
    return {"action": "restock_then_ship", "delivery_date": eta, "restock_qty": stock["shortfall"]}

_DISCOUNT_PATTERNS = [
    re.compile(r"(\d{1,2}(?:\.\d+)?)\s*%\s*(?:\w+\s+){0,2}discount", re.I),
    re.compile(r"discount\s+of\s+(\d{1,2}(?:\.\d+)?)\s*%", re.I),
]

def extract_discount_hint(text_value: str) -> Optional[float]:
    for pattern in _DISCOUNT_PATTERNS:
        match = pattern.search(text_value or "")
        if match:
            return float(match.group(1)) / 100.0
    return None

def coerce_json(answer) -> Optional[dict]:
    """The LLM's final answer may be a dict, a JSON string, or JSON inside prose/fences."""
    if isinstance(answer, dict):
        return answer
    if not isinstance(answer, str):
        return None
    candidate = answer.strip()
    candidate = re.sub(r"^```(?:json)?|```$", "", candidate, flags=re.M).strip()
    try:
        value = json.loads(candidate)
        return value if isinstance(value, dict) else None
    except ValueError:
        pass
    match = re.search(r"\{.*\}", candidate, re.S)
    if match:
        try:
            value = json.loads(match.group(0))
            return value if isinstance(value, dict) else None
        except ValueError:
            return None
    return None

_MONTH = (r"(?:january|february|march|april|may|june|july|august|september|october|november|december"
          r"|jan|feb|mar|apr|jun|jul|aug|sept|sep|oct|nov|dec)\b")
_DATE_PATTERNS = [
    re.compile(r"\(Date of request:[^)]*\)", re.I),                                           # the date the harness adds
    re.compile(rf"\b{_MONTH}\.?\s+\d{{1,2}}(?!\d)(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?", re.I),            # April 15, 2025
    re.compile(rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?{_MONTH}(?:,?\s+\d{{4}})?", re.I),    # 15 April 2025
    re.compile(r"\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}"),                                  # 2025-04-15, 4/15/25
]

def quantity_appears_in_request(quantity: int, raw: str) -> bool:
    """True if `quantity` is a number the customer wrote (a count of reams counts as 500 sheets each).

    The model turns the customer's words into a number. This checks that number against the words, so a
    slip such as reading "500 sheets" as 1 ream can never become a sale. Dates are removed first, so a
    day or a year in the message cannot vouch for a quantity."""
    text = raw
    for pattern in _DATE_PATTERNS:
        text = pattern.sub(" ", text)
    written = {int(n.replace(",", "")) for n in re.findall(r"\d[\d,]*", text) if n.replace(",", "").isdigit()}
    if re.search(r"\bream", text, re.I):
        written |= {n * config.SHEETS_PER_REAM for n in written}
    return quantity in written

def validate_parsed(raw: str, payload: dict, request_date: str) -> ParsedRequest:
    """Never trust the LLM's structure: validate names against the catalog, coerce numbers."""
    names_lower = {n.lower(): n for n in CATALOG}
    merged: Dict[str, int] = {}
    unmatched = [str(u) for u in (payload.get("unmatched") or [])]
    unclear: List[str] = []                   # items whose quantity cannot be traced to the customer's words
    for entry in payload.get("lines") or []:
        raw_name = str((entry or {}).get("item_name", "")).strip()
        name = names_lower.get(raw_name.lower())
        if not name:
            close = difflib.get_close_matches(raw_name.lower(), list(names_lower), n=1, cutoff=0.85)
            name = names_lower[close[0]] if close else None
        try:
            quantity = int(float(str((entry or {}).get("quantity", 0)).replace(",", "")))
        except ValueError:
            quantity = 0
        if not name or quantity <= 0:
            unmatched.append(raw_name or "(unnamed item)")
            continue
        if not quantity_appears_in_request(quantity, raw):      # the model's arithmetic is never trusted
            if name not in unclear:
                unclear.append(name)
            continue
        merged[name] = merged.get(name, 0) + quantity
    needed_by = payload.get("needed_by")
    if not (isinstance(needed_by, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", needed_by)):
        needed_by = None
    deadline_unconfirmed = None
    if needed_by and needed_by < request_date:
        deadline_unconfirmed, needed_by = needed_by, None
    intents = [i for i in (payload.get("intents") or []) if i in {"inventory", "quote", "order"}]
    if not intents:
        intents = ["quote"]   # ambiguous -> the recoverable choice (a wrong order is a side effect)
    return ParsedRequest(
        raw=raw, needed_by=needed_by, intents=intents,
        lines=[LineItem(n, q) for n, q in merged.items()],
        unmatched=unmatched, unclear_quantity=unclear, notes=str(payload.get("notes") or ""),
        deadline_unconfirmed=deadline_unconfirmed,
    )
