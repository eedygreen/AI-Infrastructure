"""The five agents of the plan: Customer Support, Orchestrator, Inventory, Quote and Order."""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from smolagents import OpenAIServerModel, ToolCallingAgent
from sqlalchemy.sql import text
from typing import Callable, List, Optional

from lib import config
from lib.database import CATALOG
from lib.models import LineItem, ParsedRequest, RequestContext
from lib.policy import coerce_json, decide_fulfillment, validate_parsed
from lib.starter_utils import db_engine
from lib.tools import _low_stock_items, make_customer_support_tools, make_inventory_read_tools, make_inventory_replenish_tools, make_inventory_restock_tools, make_order_tools, make_quote_history_tools, make_quote_pricing_tools


# Set up and load your env parameters and instantiate your model.
@lru_cache(maxsize=1)
def get_model():
    """Create the model client on first use, so importing this package never needs an API key."""
    return OpenAIServerModel(
        model_id='gpt-4o-mini',
        api_base='https://openai.vocareum.com/v1',
        api_key=os.getenv('OPENAI_API_KEY'),
    )



# ---------------------------------------------------------------------------
# Agent plumbing
# ---------------------------------------------------------------------------
def build_agent(name: str, description: str, tools: list) -> ToolCallingAgent:
    return ToolCallingAgent(
        tools=tools,
        model=get_model(),
        max_steps=config.AGENT_MAX_STEPS,
        name=name,
        description=description
    )


def run_agent(agent: ToolCallingAgent, task: str, timeout: Optional[float] = None):
    """Run an agent with a wall-clock timeout. A timed-out run keeps going in its thread
    (Python cannot kill it), which is safe only because every write is idempotent."""
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        return pool.submit(agent.run, task).result(timeout=config.AGENT_TIMEOUT_S if timeout is None else timeout)
    finally:
        pool.shutdown(wait=False)


def run_until_covered(ctx: RequestContext, label: str, attempt, missing) -> bool:
    """Delegate until the LEDGER covers every line (bounded). Returns True if covered.

    `attempt(todo)` performs ONE specialist pass.
    `missing()` inspects the ledger, so an LLM that skips a tool or crashes mid-run is re-run for only the lines still missing.
    Args:
        ctx (RequesteContext): request context
        label ():
        attempt: No of tried
        missing: the skipped

    Return:
        True or False
    """
    for n in range(config.AGENT_RETRIES + 1):
        todo = missing()
        if not todo:
            return True
        try:
            attempt(todo)
        except Exception as exc:  # noqa: BLE001
            ctx.log(f"{label} attempt {n + 1} failed: {type(exc).__name__}: {exc}")
        time.sleep(config.BACKOFF_S * n)
    still = missing()
    if still:
        ctx.log(f"{label} incomplete after retries: {[l.item_name for l in still]}")
    return not still


def lines_json(lines: List[LineItem]) -> str:
    return json.dumps([{"item_name": l.item_name, "quantity": l.quantity} for l in lines])


class SpecialistAgent:
    """Base class for the plan's agents. A subclass owns ONE role: its name, its prompts and its
    tools. Every delegated task builds a fresh ToolCallingAgent with a SCOPED tool set
    (least privilege: a read-only task is never handed a write tool)."""
    role: str = ""
    description: str = ""

    def _run(self, tools: list, task: str):
        return run_agent(build_agent(self.role, self.description, tools), task)


# ---------------------------------------------------------------------------
# InventoryAgent  (plan: stock + supplier ETA reads, guarded restock, replenishment)
# ---------------------------------------------------------------------------
class InventoryAgent(SpecialistAgent):
    role = "inventory_agent"
    description = "Checks stock and supplier lead times, buys missing stock, keeps stock above reorder points."

    # Tasks the Orchestrator can delegate (each is one message in the sequence diagram).
    def check_stock(self, ctx: RequestContext, lines: List[LineItem]):
        """Phase 1, read-only."""
        return self._run(make_inventory_read_tools(ctx), (
            "You are InventoryAgent (read-only). For EVERY line below call check_item_stock "
            "exactly once with its item_name. Do not invent items. When done, call final_answer "
            f"with one short sentence.\nLINES_JSON: {lines_json(lines)}"))

    def restock(self, ctx: RequestContext, lines: List[LineItem]):
        """Phase 3, guarded write: buy the shortfall for lines the Orchestrator approved."""
        return self._run(make_inventory_restock_tools(ctx), (
            "You are InventoryAgent (purchasing). For EVERY line below call restock_for_order "
            "exactly once with its item_name, then call final_answer with one sentence.\n"
            f"LINES_JSON: {lines_json(lines)}"))

    def replenish(self, ctx: RequestContext):
        """Background, off the request path."""
        return self._run(make_inventory_replenish_tools(ctx), (
            "You are InventoryAgent (replenishment). Call find_low_stock, then call "
            "restock_for_replenishment once for EACH item it returns, then call final_answer "
            "with one sentence."))


# ---------------------------------------------------------------------------
# QuoteAgent  (plan: quote history prefetch, then pricing with bulk discounts)
# ---------------------------------------------------------------------------
class QuoteAgent(SpecialistAgent):
    role = "quote_agent"
    description = "Looks up similar past quotes and prices request lines with strategic bulk discounts."

    def prefetch_history(self, ctx: RequestContext):
        """Phase 1, read-only, independent of stock."""
        return self._run(make_quote_history_tools(ctx), (
            "You are QuoteAgent. Call search_history ONCE with 2 to 5 short search terms (item names "
            "and the event type) relevant to this request, then call final_answer with one sentence.\n"
            f"REQUEST: {ctx.parsed.raw}\nLINES_JSON: {lines_json(ctx.parsed.lines)}"))

    def price(self, ctx: RequestContext, lines: List[LineItem]):
        """Phase 2: needs the Phase 1 stock position and history."""
        return self._run(make_quote_pricing_tools(ctx), (
            "You are QuoteAgent. 1) Call get_pricing_context once. 2) For EVERY line below call "
            "price_line(item_name, discount_pct). Use the recommended_discount_pct unless you have "
            "a strategic reason to go lower; never above max_discount_pct. 3) Call final_answer "
            f"with one sentence. Never mention internal figures.\nLINES_JSON: {lines_json(lines)}"))


# ---------------------------------------------------------------------------
# OrderAgent  (plan: finalize sales; the only agent that writes a sale)
# ---------------------------------------------------------------------------
class OrderAgent(SpecialistAgent):
    role = "order_agent"
    description = "Finalizes approved sales transactions."

    def place_orders(self, ctx: RequestContext, lines: List[LineItem]):
        """Phase 3: only lines the Orchestrator approved."""
        return self._run(make_order_tools(ctx), (
            "You are OrderAgent. For EVERY line below call finalize_sale exactly once with its "
            "item_name, then call final_answer with one sentence.\n"
            f"LINES_JSON: {lines_json(lines)}"))


# an orchestration agent that will manage them.
# ---------------------------------------------------------------------------
# OrchestratorAgent (deterministic coordinator; delegates to the three specialists)
# ---------------------------------------------------------------------------
CUSTOMER_REASONS = {
    "availability_unknown": "we could not verify availability right now",
    "eta_unknown": "we could not confirm a delivery date right now",
    "eta_after_deadline": "we cannot deliver by your requested date",
    "restock_denied": "we could not secure additional stock at this time",
    "stock changed": "stock changed while we were processing",
    "write failed": "we hit a temporary system problem",
    "no_quote": "we could not price this item right now",
    "deadline_unconfirmed": "please confirm the delivery date you need so we can check supplier timing",
    "already_ordered_other_quantity": "this item is already ordered under this request, so a colleague will help change the quantity",
    "held_partial": "it was held because another item in your order could not be fulfilled",
}

PARSE_FAILED_MESSAGE = ("We could not process your message because of a temporary problem on our side. "
                        "Please try again in a few minutes.")

class OrchestratorAgent:
    def __init__(self, run_id: str, inventory: InventoryAgent, quote: QuoteAgent, order: OrderAgent):
        self.run_id = run_id
        self.inventory, self.quote, self.order = inventory, quote, order
        self._background = ThreadPoolExecutor(max_workers=1)
        self._pending: list = []

    # -- Plan ---------------------------------------------------------------
    @staticmethod
    def plan(parsed: ParsedRequest) -> set:
        needs = set(parsed.intents)
        if "order" in needs:
            needs |= {"quote", "inventory"}   # an order is only safe with price + stock
        if "quote" in needs:
            needs.add("inventory")            # pricing depends on stock position
        return needs

    # -- Main entry ----------------------------------------------------------
    def process(self, ctx: RequestContext, parsed: ParsedRequest) -> str:
        ctx.parsed = parsed
        if not parsed.lines:
            ctx.log("parsing failed (system problem); not blaming the customer" if parsed.failed
                    else "no recognisable items; asking the customer to clarify")
            return self.render(ctx, set())
        needs = self.plan(parsed)
        ctx.log(f"plan: {sorted(needs)} for {[l.item_name for l in parsed.lines]}")
        self._phase1(ctx, needs)
        if "quote" in needs:
            self._phase2(ctx)
        if "order" in needs:
            self._phase3(ctx)
        return self.render(ctx, needs)

    # -- Phase 1: independent reads, in parallel ------------------------------
    def _phase1(self, ctx: RequestContext, needs: set) -> None:
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = []
            if "inventory" in needs:
                jobs.append(pool.submit(self._inventory_check, ctx))
            if "quote" in needs:
                jobs.append(pool.submit(self._history_prefetch, ctx))
            for job in jobs:
                try:
                    job.result()
                except Exception as exc:  # noqa: BLE001 - phase helpers should not raise
                    ctx.log(f"phase 1 helper crashed: {exc}")

    def _inventory_check(self, ctx: RequestContext) -> bool:
        return run_until_covered(
            ctx, "inventory check", lambda todo: self.inventory.check_stock(ctx, todo),
            lambda: [l for l in ctx.parsed.lines if l.item_name not in ctx.stock])

    def _history_prefetch(self, ctx: RequestContext) -> bool:
        try:
            self.quote.prefetch_history(ctx)
        except Exception as exc:  # noqa: BLE001
            ctx.log(f"quote history prefetch failed (advisory, continuing): {exc}")   # non-fatal
            return False
        return True

    # -- Phase 2: pricing (needs Phase 1) --------------------------------------
    def _phase2(self, ctx: RequestContext) -> None:
        priceable = [l for l in ctx.parsed.lines
                     if ctx.stock.get(l.item_name) and not ctx.stock[l.item_name].get("error")]
        if not priceable:
            ctx.log("phase 2 skipped: no line has verified availability")
            return

        run_until_covered(
            ctx, "pricing", lambda todo: self.quote.price(ctx, todo),
            lambda: [l for l in priceable if l.item_name not in ctx.quotes])

    # -- Phase 3: side effects, gated --------------------------------------------
    def _phase3(self, ctx: RequestContext) -> None:
        lines = ctx.parsed.lines
        for line in lines:
            prior = self._prior_sale(ctx, line)
            if prior:   # idempotency must cover the DECISION too, or a repeat would buy stock again
                ctx.orders[line.item_name] = {"item_name": line.item_name, "status": "confirmed", "reason": None,
                                              "delivery_date": prior.get("delivery_date", ctx.request_date),
                                              "total": prior.get("total", 0.0), "replayed": True}
                ctx.decisions[line.item_name] = {"action": "ship", "approved": True, "already_ordered": True,
                                                 "delivery_date": ctx.orders[line.item_name]["delivery_date"]}
                ctx.log(f"{line.item_name}: already ordered earlier, reporting existing order")
                continue
            other = self._other_quantity_sold(ctx, line)
            if other is not None:                       # a follow-up changed the quantity of an item already sold: never sell it twice
                ctx.decisions[line.item_name] = {"action": "skip",
                                                 "reason": "write failed" if other < 0 else "already_ordered_other_quantity"}
                ctx.log(f"{line.item_name}: not placing a second other (already sold this request: {other})")
                continue
            decision = decide_fulfillment(ctx.request_date, ctx.parsed.needed_by, ctx.stock.get(line.item_name),
                                          bool(ctx.parsed.deadline_unconfirmed))
            if decision["action"] != "skip" and line.item_name not in ctx.quotes:
                decision = {"action": "skip", "reason": "no_quote"}
            ctx.decisions[line.item_name] = decision

        if not config.ALLOW_PARTIAL_FULFILLMENT and (
                ctx.parsed.unmatched or ctx.parsed.unclear_quantity
                or any(d["action"] == "skip" for d in ctx.decisions.values())):
            for name, d in ctx.decisions.items():
                if d["action"] != "skip":
                    ctx.decisions[name] = {"action": "skip", "reason": "held_partial"}
            ctx.log("all-or-nothing policy: order held, no side effects")
            return

        restock_lines = [l for l in lines if ctx.decisions[l.item_name]["action"] == "restock_then_ship"]
        if restock_lines:
            run_until_covered(
                ctx, "restock", lambda todo: self.inventory.restock(ctx, todo),
                lambda: [l for l in restock_lines if l.item_name not in ctx.restocks])
            for line in restock_lines:
                outcome = ctx.restocks.get(line.item_name)
                if not outcome or outcome["status"] != "placed":
                    ctx.decisions[line.item_name] = {"action": "skip", "reason": "restock_denied"}

        approved = [l for l in lines if ctx.decisions[l.item_name]["action"] in {"ship", "restock_then_ship"}]
        for line in approved:
            ctx.decisions[line.item_name]["approved"] = True
        if not approved:
            ctx.log("no approved lines: skipping order, no side effects")
            return

        run_until_covered(
            ctx, "order", lambda todo: self.order.place_orders(ctx, todo),
            lambda: [l for l in approved if l.item_name not in ctx.orders])

    @staticmethod
    def _prior_sale(ctx: RequestContext, line: LineItem) -> Optional[dict]:
        try:
            with db_engine.connect() as conn:
                stored = conn.execute(text("SELECT result FROM idempotency_keys WHERE key = :k"),
                                      {"k": ctx.key("sale", line.item_name, line.quantity)}).scalar()
            return json.loads(stored) if stored else None
        except Exception as exc:  # noqa: BLE001 - fall through to the normal (idempotent) path
            ctx.log(f"prior-sale lookup failed for {line.item_name}: {exc}")
            return None

    @staticmethod
    def _other_quantity_sold(ctx: RequestContext, line: LineItem) -> Optional[int]:
        """Quantity already sold for this item under this request when it differs from the one asked for now.
        Returns None if there is no such sale, and -1 if the lookup itself failed (callers must then fail closed)."""
        same = ctx.key("sale", line.item_name, line.quantity)
        prefix = ctx.key("sale", line.item_name, 0)[:-1]       # "<run>:<request>:sale:<item>:", built from ctx.key itself
        try:
            with db_engine.connect() as conn:
                other = conn.execute(
                    text("SELECT key FROM idempotency_keys WHERE substr(key, 1, :n) = :p AND key <> :same LIMIT 1"),
                    {"n": len(prefix), "p": prefix, "same": same}).scalar()
            return int(other[len(prefix):]) if other else None
        except Exception as exc:  # noqa: BLE001
            ctx.log(f"other-quantity lookup failed for {line.item_name}: {exc}")
            return -1


    # -- Synthesis: template-rendered, so internal data cannot leak ---------------
    @staticmethod
    def _discount_note(quote: dict, stock: Optional[dict], quantity: int) -> str:
        """Say why the price has the discount it has (or none), so the customer is never left guessing."""
        if quote["discount_applied_pct"]:
            return f", {quote['discount_applied_pct']:g}% bulk discount"
        first_discount_quantity = min(minimum for minimum, rate in config.DISCOUNT_TIERS if rate > 0)
        if (stock or {}).get("shortfall") and quantity >= first_discount_quantity:
            return ", no bulk discount because we must restock this item"
        return ""

    def render(self, ctx: RequestContext, needs: set) -> str:
        p = ctx.parsed
        out = [f"Thank you for your request (reference #{ctx.request_id})."]
        if p.failed:
            out.append(PARSE_FAILED_MESSAGE)
            return "\n".join(out)
        if p.unmatched:
            out.append("We could not match these to products we sell: " + "; ".join(p.unmatched) + ".")
        if p.unclear_quantity:
            out.append("We were not sure how many you need of: " + "; ".join(p.unclear_quantity)
                       + ". Please confirm the quantities and we will happily help.")
        if p.deadline_unconfirmed:
            out.append(f"The delivery date you gave ({p.deadline_unconfirmed}) is earlier than your request date. "
                       "Could you confirm the date you need?"
                       + (" Items already in stock are not affected." if "order" in needs else ""))
        if not p.lines:
            out.append("Please tell us which products and quantities you need and we will happily help.")
            return "\n".join(out)

        quoted_total = confirmed_total = 0.0
        for line in p.lines:
            stock, quote = ctx.stock.get(line.item_name), ctx.quotes.get(line.item_name)
            decision, order = ctx.decisions.get(line.item_name), ctx.orders.get(line.item_name)
            head = f"- {line.quantity:,} x {line.item_name}"
            if "quote" in needs and order and order.get("replayed") and order["status"] == "confirmed":
                quoted_total += order["total"]
                head += f": ${order['total']:,.2f}"                    # the amount already charged
            elif "quote" in needs and quote:
                quoted_total += quote["line_total"]
                head += f": ${quote['line_total']:,.2f}{self._discount_note(quote, stock, line.quantity)}"
            elif "quote" in needs:
                head += ": price unavailable right now"
            if "order" in needs:
                if order and order["status"] == "confirmed":
                    confirmed_total += order["total"]
                    restocked = bool(decision and decision.get("action") == "restock_then_ship")
                    head += (" | CONFIRMED" + (" (already placed)" if order.get("replayed") else "")
                             + f", delivery by {order['delivery_date']}"
                             + (" (includes time to restock this item)" if restocked else ""))
                else:
                    reason = (order or {}).get("reason") or (decision or {}).get("reason") or "no_quote"
                    detail = CUSTOMER_REASONS.get(reason, "we could not complete it right now")
                    eta = (decision or {}).get("eta")
                    head += f" | NOT FULFILLED: {detail}" + (f" (earliest delivery {eta})" if eta else "")
            elif "inventory" in needs and "quote" not in needs:
                if stock and not stock.get("error"):
                    head += (": available now" if stock["shortfall"] == 0 else
                             f": {stock['on_hand']:,} in stock, remainder can be supplied by {stock.get('eta') or 'a date we cannot confirm yet'}")
                else:
                    head += ": availability unavailable right now"
            out.append(head)
        if "quote" in needs and quoted_total:
            out.append(f"Quote total: ${quoted_total:,.2f}")
        if "order" in needs and confirmed_total:
            out.append(f"Order total confirmed: ${confirmed_total:,.2f}")
        return "\n".join(out)

    # -- Background replenishment (off the request path) --------------------------
    def after_reply(self, ctx: RequestContext) -> None:
        """Call AFTER the reply has gone to the customer. Never affects their outcome."""
        self._pending.append(self._background.submit(self._replenish, ctx))

    def _replenish(self, ctx: RequestContext) -> None:
        try:
            low = _low_stock_items(ctx.request_date)   # deterministic pre-check: no LLM call if nothing is low
            if not low:
                return
            ctx.log(f"replenishment: {[x['item_name'] for x in low]} below reorder point")
            self.inventory.replenish(ctx)
            ctx.log("replenishment report: " + json.dumps(ctx.background))   # logged, never customer-facing
        except Exception as exc:  # noqa: BLE001
            ctx.log(f"replenishment failed (flag for ops review): {exc}")

    def drain_background(self) -> None:
        """Test harness only: wait so the next request sees deterministic books."""
        for future in self._pending:
            future.result()
        self._pending.clear()


# CustomerSupportAgent interfacing customers
# ---------------------------------------------------------------------------
# CustomerSupportAgent (front door + bounded customer loop)
# ---------------------------------------------------------------------------
class CustomerSupportAgent(SpecialistAgent):
    role = "customer_support_agent"
    description = "Understands customer requests, categorizes them, and runs the bounded customer loop."

    def __init__(self, orchestrator: OrchestratorAgent, run_id: str):
        self.orchestrator, self.run_id = orchestrator, run_id
        self.last_ctx: Optional[RequestContext] = None

    def categorize(self, ctx: RequestContext, raw: str) -> ParsedRequest:
        task = (
            "You are CustomerSupportAgent for Munder Difflin Paper Company. Convert the customer message "
            "into structured data.\nRules:\n"
            "- Map every requested product to an EXACT catalog name (use lookup_catalog_item when unsure). "
            "Only exact names may appear in 'lines'. Products you cannot map go in 'unmatched' using the customer's wording.\n"
            f"- quantity: positive integer count of catalog units. Every catalog item is counted in single sheets or pieces, "
            "so a number of sheets or pieces stays exactly as the customer wrote it (500 sheets means 500, never 1). "
            f"Only reams are converted: 1 ream = {config.SHEETS_PER_REAM} sheets. \n"
            "Keep other units (boxes, packs) as the stated number and mention them in 'notes'.\n"
            "- needed_by: delivery deadline as YYYY-MM-DD or null. Relative dates are relative to the request date.\n"
            "- intents: subset of [\"inventory\",\"quote\",\"order\"]. 'order' ONLY if the customer clearly wants to buy/place/ship goods; "
            "'quote' if they ask for a price; 'inventory' if they only ask about availability.\n"
            "Finish by calling final_answer with ONE argument named 'answer', whose value is the JSON object "
            "below written as a string. Do NOT pass lines, unmatched, needed_by, intents or notes as separate "
            "arguments: final_answer accepts only 'answer'. The JSON object: "
            "{\"lines\":[{\"item_name\":str,\"quantity\":int}],\"unmatched\":[str],\"needed_by\":str|null,"
            "\"intents\":[str],\"notes\":str}\n"
            f"Request date: {ctx.request_date}\nCATALOG: {json.dumps(list(CATALOG))}\n"
            f"CUSTOMER MESSAGE: {raw}")
        last_error = None
        for attempt in range(config.AGENT_RETRIES + 1):
            try:
                answer = self._run(make_customer_support_tools(), task)
                payload = coerce_json(answer)
                if payload is not None:
                    return validate_parsed(raw, payload, ctx.request_date)
                last_error = f"unparseable answer: {str(answer)[:120]}"
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
            ctx.log(f"categorize attempt {attempt + 1} failed: {last_error}")
        return ParsedRequest(raw=raw, needed_by=None, intents=[], lines=[], unmatched=[raw[:80]], failed=True)

    def handle(self, raw_request: str, request_date: str, request_id: int,
               followup: Optional[Callable[[str, int], Optional[str]]] = None) -> str:
        """Bounded loop. `followup(reply, iteration)` returns None when the customer is satisfied,
        otherwise their follow-up text. In batch tests there is no live customer, so one pass."""
        text_in, reply = raw_request, ""
        for iteration in range(1, config.MAX_ITERATIONS + 1):
            ctx = RequestContext(request_id=request_id, request_date=request_date,
                                 run_id=self.run_id, iteration=iteration)
            self.last_ctx = ctx
            parsed = self.categorize(ctx, text_in)
            reply = self.orchestrator.process(ctx, parsed)
            self.orchestrator.after_reply(ctx)           # background work starts only after the reply exists
            feedback = followup(reply, iteration) if followup else None
            if feedback is None:
                return reply
            text_in = feedback
        return reply + "\n\nWe want to get this right, so a human colleague will follow up with you shortly."
