"""All tool factories. Built per request (closures over the ledger) and scoped per phase.

Each make_*_tools(ctx) returns the tools one task type may use, so a read-only task is never
handed a write tool."""

import difflib
import json
import pandas as pd
from smolagents import tool
from typing import List, Optional

from lib import config
from lib.database import CATALOG, _cash_in_conn, _commit_transaction, _retry, _stock_in_conn
from lib.models import RequestContext
from lib.policy import apply_price_policy, discount_cap, extract_discount_hint, list_unit_price
from lib.starter_utils import (
    create_transaction, db_engine, 
    generate_financial_report, 
    get_all_inventory,
    get_cash_balance, get_stock_level,
    get_supplier_delivery_date,
    search_quote_history
)


# Tools for inventory agent
# ---------------------------------------------------------------------------
# Shared inventory helpers (deterministic, no LLM)
# ---------------------------------------------------------------------------
def _low_stock_items(request_date: str, stock: Optional[dict] = None) -> List[dict]:
    """Deterministic reorder scan: items below min_stock_level, with a top-up quantity.
    `stock` is the full inventory if the caller already read it (the tools do, with get_all_inventory).
    """
    inventory = pd.read_sql("SELECT item_name, min_stock_level FROM inventory", db_engine)
    if stock is None:
        stock = get_all_inventory(as_of_date=request_date)
    low = []
    for _, row in inventory.iterrows():
        name, minimum = row["item_name"], int(row["min_stock_level"])
        current = int(stock.get(name, 0))
        if current < minimum:
            low.append({"item_name": name, "current_stock": current, "reorder_point": minimum,
                        "suggested_quantity": max(1, minimum * config.REORDER_TARGET_MULT - current)})
    return low


def _place_restock(ctx: RequestContext, item_name: str, quantity: int,
                   purpose: str, cash_available: float, writer=create_transaction) -> dict:
    """Guarded stock purchase. 
    The calling tool reads the cash balance with get_cash_balance and passes it in: an early refusal that
    costs nothing. The same check is repeated at the moment of writing, in case cash changed meanwhile.
    `writer` is the starter's create_transaction.
    """
    item = CATALOG.get(item_name)
    if not item or quantity <= 0:
        return {"item_name": item_name, "status": "denied", "reason": "invalid item or quantity"}
    cost = round(item["unit_price"] * quantity, 2)
    try:
        eta = _retry(lambda: get_supplier_delivery_date(input_date_str=ctx.request_date, quantity=quantity))
    except Exception:  # noqa: BLE001
        eta = None
    if cash_available - cost < config.MIN_CASH_RESERVE:         # # get_cash_balance returns 0.0 on error: refuses
        return {
            "item_name": item_name,
            "quantity": quantity,
            "eta": eta,
            "status": "denied",
            "reason": "insufficient funds"
        }

    def guard(conn):
        """The final cash check, run just before the write: refuse if the purchase would take cash below the
        reserve. Returns a reason to refuse, or None to go ahead."""
        if _cash_in_conn(conn, ctx.request_date) - cost < config.MIN_CASH_RESERVE:
            return "insufficient funds"
        return None

    try:
        outcome = _retry(lambda: _commit_transaction(
            key=ctx.key(f"restock-{purpose}", item_name, quantity), kind=f"restock-{purpose}",
            item_name=item_name, transaction_type="stock_orders", units=quantity,
            price=cost, date=ctx.request_date, guard=guard, writer=writer))
    except Exception as exc:  # noqa: BLE001
        ctx.log(f"restock write failed for {item_name}: {exc}")
        return {"item_name": item_name, "quantity": quantity, "status": "denied", "reason": "write failed"}
    placed = outcome["status"] == "committed" or outcome.get("replayed")
    return {"item_name": item_name, "quantity": quantity, "eta": eta,
            "status": "placed" if placed else "denied", "reason": outcome.get("reason")}


# Tool sets, one per kind of delegated task. Only the first is read-only.
def make_inventory_read_tools(ctx: RequestContext) -> list:
    """The tool for checking stock. It only looks things up, so a check can never change anything."""
    @tool
    def check_item_stock(item_name: str) -> str:
        """Check on-hand stock for one requested item and, if short, the supplier delivery date for the missing units.

        Args:
            item_name: Exact catalog name of an item in this request.
        """
        line = ctx.line(item_name)
        if line is None:
            return json.dumps({"error": f"'{item_name}' is not part of this request"})
        try:
            on_hand = _retry(lambda: int(
                get_stock_level(item_name=item_name, as_of_date=ctx.request_date)["current_stock"].iloc[0]))
        except Exception as exc:  # noqa: BLE001
            ctx.stock[item_name] = {"error": "database unavailable"}
            ctx.log(f"stock check failed for {item_name}: {exc}")
            return json.dumps({"item_name": item_name, "error": "database unavailable"})
        shortfall = max(0, line.quantity - on_hand)
        eta = None
        if shortfall:
            try:
                eta = _retry(lambda: get_supplier_delivery_date(
                    input_date_str=ctx.request_date, quantity=shortfall))
            except Exception as exc:  # noqa: BLE001
                ctx.log(f"supplier ETA unavailable for {item_name}: {exc}")
        result = {"item_name": item_name, "requested": line.quantity, "on_hand": on_hand,
                  "shortfall": shortfall, "eta": eta}
        ctx.stock[item_name] = result
        return json.dumps(result)

    return [check_item_stock]


def make_inventory_restock_tools(ctx: RequestContext) -> list:
    """The one tool that may buy stock for an approved order."""
    @tool
    def restock_for_order(item_name: str) -> str:
        """Buy the missing units of one item from the supplier so an approved customer order can be fulfilled.

        Args:
            item_name: Exact catalog name of an item the orchestrator approved for restocking.
        """
        decision = ctx.decisions.get(item_name)
        if not decision or decision.get("action") != "restock_then_ship":
            return json.dumps({"item_name": item_name, "status": "denied", "reason": "not approved"})
        cash_available = get_cash_balance(as_of_date=ctx.request_date)
        result = _place_restock(ctx, item_name,
                                int(decision["restock_qty"]), "shortfall",
                                cash_available=cash_available,
                                writer=create_transaction
                            )
        ctx.restocks[item_name] = result
        return json.dumps({field: result[field] for field in ("item_name", "status", "eta") if field in result})

    return [restock_for_order]


def make_inventory_replenish_tools(ctx: RequestContext) -> list:
    """The tools for routine replenishment: find the items below their reorder point, and top one up."""
    @tool
    def find_low_stock() -> str:
        """List stocked items that are below their reorder point, with the suggested top-up quantity."""
        stock = get_all_inventory(as_of_date=ctx.request_date)
        return json.dumps(_low_stock_items(ctx.request_date, stock=stock))

    @tool
    def restock_for_replenishment(item_name: str) -> str:
        """Top up one low-stock item to its target level. Quantity is computed by the system.

        Args:
            item_name: Exact catalog name of an item returned by find_low_stock.
        """
        match = next((entry for entry in _low_stock_items(ctx.request_date) if entry["item_name"] == item_name), None)
        if not match:
            return json.dumps({"item_name": item_name, "status": "skipped", "reason": "not below reorder point"})
        cash_available = get_cash_balance(as_of_date=ctx.request_date)
        result = _place_restock(ctx, item_name,
                                match["suggested_quantity"], 
                                "replenishment",
                                cash_available=cash_available,
                                writer=create_transaction
                            )
        ctx.background.append(result)
        return json.dumps({field: result[field] for field in ("item_name", "status") if field in result})

    return [find_low_stock, restock_for_replenishment]


# Tools for quoting agent
def make_quote_history_tools(ctx: RequestContext) -> list:
    """The tool that finds similar past quotes."""
    @tool
    def search_history(search_terms: List[str]) -> str:
        """Search past quotes that match ANY of the given terms (item names, event types).

        Args:
            search_terms: Two to five short search terms.
        """
        seen, merged = set(), []
        for term in [str(raw_term) for raw_term in search_terms][:5]:
            try:
                rows = _retry(lambda current_term=term: search_quote_history(search_terms=[current_term], limit=3))
            except Exception as exc:  # noqa: BLE001
                ctx.log(f"quote history unavailable for '{term}': {exc}")
                continue
            for row in rows:
                ident = (row.get("original_request"), row.get("order_date"), row.get("total_amount"))
                if ident not in seen:
                    seen.add(ident)
                    merged.append(row)
        ctx.history = merged[:5]
        summary = [{"total_amount": past.get("total_amount"), "order_size": past.get("order_size"),
                    "event_type": past.get("event_type"),
                    "discount_hint_pct": (lambda hint: None if hint is None else round(hint * 100, 1))(
                        extract_discount_hint(past.get("quote_explanation", "")))}
                   for past in ctx.history]
        return json.dumps(summary)

    return [search_history]


def make_quote_pricing_tools(ctx: RequestContext) -> list:
    """The tools for pricing: the allowed discount range for each item, and pricing one item within it."""
    @tool
    def get_pricing_context() -> str:
        """Get list prices and the allowed discount range for every line of this request.
        Call this once before pricing."""
        high_demand: set = set()
        report_ok = True
        try:
            report = _retry(lambda: generate_financial_report(as_of_date=ctx.request_date))
            high_demand = {product["item_name"] for product in report["top_selling_products"]}
        except Exception as exc:  # noqa: BLE001
            report_ok = False # fail closed: without the demand signal, treat every item as a top seller
            ctx.log(f"financial report unavailable, halving every discount cap: {exc}")
        hints = [found for found in (extract_discount_hint(past.get("quote_explanation", "")) for past in ctx.history)
                 if found is not None]
        hint = sorted(hints)[len(hints) // 2] if hints else None
        out = {}
        for line in ctx.parsed.lines:
            shortfall = ctx.stock.get(line.item_name, {}).get("shortfall", 0)
            cap = discount_cap(line.quantity, shortfall, (not report_ok) or line.item_name in high_demand)
            recommended = min(cap, hint) if hint is not None else cap
            ctx.pricing_ctx[line.item_name] = {"cap": cap, "recommended": recommended}
            out[line.item_name] = {
                "quantity": line.quantity,
                "unit_list_price": round(list_unit_price(CATALOG[line.item_name]["unit_price"]), 4),
                "max_discount_pct": round(cap * 100, 1),
                "recommended_discount_pct": round(recommended * 100, 1),
            }
        return json.dumps(out)

    @tool
    def price_line(item_name: str, discount_pct: float) -> str:
        """Price one line of the request. The discount is clamped to the allowed maximum.

        Args:
            item_name: Exact catalog name of an item in this request.
            discount_pct: Discount percent you choose (0 to 100), normally the recommended value.
        """
        line = ctx.line(item_name)
        if line is None:
            return json.dumps({"error": f"'{item_name}' is not part of this request"})
        stock = ctx.stock.get(item_name)
        if not stock or stock.get("error"):
            return json.dumps({"item_name": item_name, "error": "availability unknown, cannot price safely"})
        cap = ctx.pricing_ctx.get(item_name, {}).get(
            "cap", discount_cap(line.quantity, stock.get("shortfall", 0), False))
        priced = apply_price_policy(CATALOG[item_name]["unit_price"], line.quantity,
                                    float(discount_pct) / 100.0, cap)
        result = {"item_name": item_name, "quantity": line.quantity,
                  "unit_list_price": priced["unit_list_price"],
                  "discount_applied_pct": round(priced["discount"] * 100, 1),
                  "discount_was_clamped": priced["clamped"], "line_total": priced["line_total"]}
        ctx.quotes[item_name] = result
        return json.dumps(result)

    return [get_pricing_context, price_line]


# Tools for ordering agent
def make_order_tools(ctx: RequestContext) -> list:
    """The one tool that records a sale."""
    @tool
    def finalize_sale(item_name: str) -> str:
        """Record the sale of one approved line. Price and quantity come from the stored quote.

        Args:
            item_name: Exact catalog name of an approved line item.
        """
        decision, quote = ctx.decisions.get(item_name), ctx.quotes.get(item_name)
        if not decision or decision.get("action") not in {"ship", "restock_then_ship"} \
                or not decision.get("approved"):
            return json.dumps({"item_name": item_name, "status": "denied", "reason": "not approved"})
        if not quote:
            return json.dumps({"item_name": item_name, "status": "denied", "reason": "no quote on file"})
        quantity = quote["quantity"]

        def guard(conn):   # checked just before the write, under the write lock
            """The final stock check, run just before the write: refuse if the stock is no longer there.
            Returns a reason to refuse, or None to go ahead."""
            if _stock_in_conn(conn, item_name, ctx.request_date) < quantity:
                return "stock changed"
            return None

        try:
            outcome = _retry(lambda: _commit_transaction(
                key=ctx.key("sale", item_name, quantity), kind="sale", item_name=item_name,
                transaction_type="sales", units=quantity, price=quote["line_total"],
                date=ctx.request_date, guard=guard, writer=create_transaction,
                meta={"delivery_date": decision["delivery_date"], "total": quote["line_total"]}
                )
            )
        except Exception as exc:  # noqa: BLE001
            ctx.log(f"sale write failed for {item_name}: {exc}")
            ctx.orders[item_name] = {"item_name": item_name, "status": "denied", "reason": "write failed"}
            return json.dumps(ctx.orders[item_name])
        confirmed = outcome["status"] == "committed" or outcome.get("replayed")
        ctx.orders[item_name] = {
            "item_name": item_name, "status": "confirmed" if confirmed else "denied",
            "reason": outcome.get("reason"),
            "delivery_date": outcome.get("delivery_date", decision["delivery_date"]),
            "total": outcome.get("total", quote["line_total"]), "replayed": bool(outcome.get("replayed")),
        }
        return json.dumps({field: value for field, value in ctx.orders[item_name].items() if field != "total"})

    return [finalize_sale]


# Tools for customer support agent
def make_customer_support_tools() -> list:
    """The tool the customer support agent uses to match a customer's wording to catalog names."""
    @tool
    def lookup_catalog_item(query: str) -> str:
        """Find catalog items whose name best matches a customer's wording.

        Args:
            query: The product wording used by the customer, e.g. 'glossy A4 sheets'.
        """
        wording = query.lower().strip()
        names = list(CATALOG)
        hits = [name for name in names if wording in name.lower() or name.lower() in wording]
        for close_match in difflib.get_close_matches(wording, [name.lower() for name in names], n=3, cutoff=0.4):
            original = next(name for name in names if name.lower() == close_match)
            if original not in hits:
                hits.append(original)
        return json.dumps(hits[:5])

    return [lookup_catalog_item]
