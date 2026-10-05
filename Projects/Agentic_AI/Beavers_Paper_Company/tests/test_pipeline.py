import os, sys, json, time, shutil, tempfile, traceback, logging
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "shims"))            # scripted stand-in for smolagents (no LLM calls)
sys.path.insert(0, os.path.dirname(HERE))                   # project root: where lib/, workflow.py, main.py and utils/ live

# Load the stand-in smolagents BY FILE PATH instead of trusting import-path order. If a `smolagents` folder sits in the project
# root, or the real library comes earlier on the path, the real one would win and 33 tests would die with "no attribute HOOKS".
import importlib.util
_SHIM = os.path.join(HERE, "shims", "smolagents", "__init__.py")
assert os.path.isfile(_SHIM), f"test stand-in missing: expected {_SHIM}"
_spec = importlib.util.spec_from_file_location("smolagents", _SHIM)
_fake = importlib.util.module_from_spec(_spec)
sys.modules["smolagents"] = _fake
_spec.loader.exec_module(_fake)
assert hasattr(_fake, "HOOKS"), f"{_SHIM} is not the test stand-in (it has no HOOKS): is the file empty or an old copy?"

WORK = tempfile.mkdtemp(prefix="munder_")
os.chdir(WORK)
import pandas as pd

pd.DataFrame({
    "response": [f"Need {n} of paper for event {i}" for i, n in enumerate([100, 500, 2000, 50, 300, 800])],
    "job": ["planner"] * 6, "need_size": ["small", "medium", "large", "small", "medium", "large"],
    "event": ["party"] * 6,
}).to_csv("quote_requests.csv", index=False)
pd.DataFrame({
    "total_amount": [120.0, 560.0, 2100.0, 60.0, 340.0, 900.0],
    "quote_explanation": ["Applied a 10% discount for the bulk order.", "Standard pricing.",
                          "We offered a discount of 15% on this large order.", "No discount.",
                          "5% bulk discount applied.", "10% volume discount."],
    "request_metadata": [str({"job_type": "planner", "order_size": s, "event_type": "party"})
                         for s in ["small", "medium", "large", "small", "medium", "large"]],
}).to_csv("quotes.csv", index=False)

import smolagents
import importlib


class _Namespace:
    """One handle (`ps`) over the split modules, so the tests read as they did for the single file.
    Reads search the modules in order. Writes are allowed only for policy knobs and go to lib.config."""
    _ORDER = ["lib.agents", "lib.tools", "lib.policy", "lib.database", "lib.models", "lib.starter_utils",
              "workflow", "lib.config"]

    def __init__(self):
        object.__setattr__(self, "_mods", [importlib.import_module(m) for m in self._ORDER])

    def __getattr__(self, name):
        for m in self._mods:
            if hasattr(m, name):
                return getattr(m, name)
        raise AttributeError(name)

    def __setattr__(self, name, value):
        config = importlib.import_module("lib.config")
        if not hasattr(config, name):
            raise AttributeError(f"{name} is not a policy knob in lib.config")
        setattr(config, name, value)


ps = _Namespace()
ps.BACKOFF_S = 0.0
ps.AGENT_TIMEOUT_S = 20

RUN = {"n": 0}
RESULTS = []


def fresh():
    ps.init_database(ps.db_engine)
    ps.ensure_runtime_tables(reset=True)
    h = smolagents.HOOKS
    h["fail"].clear(); h["skip"].clear(); h["calls"].clear(); h["delay"].clear(); h["discount"] = None
    h["last_task"].clear(); h["parse_garbage"] = False; h["parse_via_final_answer"] = False
    ps.ALLOW_PARTIAL_FULFILLMENT = True
    RUN["n"] += 1
    orch = ps.OrchestratorAgent(f"run{RUN['n']}", ps.InventoryAgent(), ps.QuoteAgent(), ps.OrderAgent())
    return orch, ps.CustomerSupportAgent(orch, f"run{RUN['n']}")


def go(csa, orch, text, date="2025-04-01", rid=1, followup=None):
    reply = csa.handle(f"{text} (Date of request: {date})", date, rid, followup)
    orch.drain_background()
    return reply, csa.last_ctx


def inv():
    return pd.read_sql("SELECT * FROM inventory", ps.db_engine)


def stock(item, date="2999-01-01"):
    return int(ps.get_stock_level(item_name=item, as_of_date=date)["current_stock"].iloc[0])


def cash(date="2999-01-01"):
    return ps.get_cash_balance(as_of_date=date)


def rows(item=None, kind=None):
    q = "SELECT * FROM transactions WHERE item_name IS NOT NULL"
    df = pd.read_sql(q, ps.db_engine)
    if item: df = df[df.item_name == item]
    if kind: df = df[df.transaction_type == kind]
    return df


def test(fn):
    try:
        fn(); RESULTS.append((fn.__name__, True, ""))
    except Exception:
        RESULTS.append((fn.__name__, False, traceback.format_exc(limit=3)))
    return fn


def stocked(min_stock=0):
    d = inv()
    for _, r in d.iterrows():
        if stock(r.item_name, "2025-04-01") >= min_stock and r.item_name:
            return r.item_name, int(r.min_stock_level), float(r.unit_price)
    raise RuntimeError("no suitable stocked item")


# ------------------------------------------------------------------ pure logic
@test
def pure_policy():
    assert ps.discount_cap(50, 0, False) == 0.0
    assert ps.discount_cap(100, 0, False) == 0.05 and ps.discount_cap(500, 0, False) == 0.10
    assert ps.discount_cap(5000, 0, False) == 0.15
    assert ps.discount_cap(5000, 10, False) == 0.0           # stock we must buy in: never discounted
    assert ps.discount_cap(5000, 0, True) == 0.075            # high demand halves the cap
    p = ps.apply_price_policy(0.10, 1000, 0.90, 0.15)          # asked 90%, cap 15%
    assert abs(p["discount"] - 0.15) < 1e-9 and p["clamped"]
    assert p["line_total"] == round(1000 * 0.10 * 1.3 * 0.85, 2)
    old = ps.LIST_MARKUP; ps.LIST_MARKUP = 0.12                 # margin floor (10%) must bind below the cap
    try:
        p = ps.apply_price_policy(1.0, 100, 0.15, 0.15)
        assert p["line_total"] >= 100 * 1.0 * 1.10 - 0.01, p
    finally: ps.LIST_MARKUP = old
    assert ps.apply_price_policy(1.0, 10, -5, 0.15)["discount"] == 0.0   # negative discount ignored


@test
def pure_gate():
    d = ps.decide_fulfillment
    assert d("2025-04-01", None, {"shortfall": 0})["action"] == "ship"
    assert d("2025-04-01", "2025-04-02", {"shortfall": 5, "eta": "2025-04-05"})["reason"] == "eta_after_deadline"
    assert d("2025-04-01", "2025-04-09", {"shortfall": 5, "eta": "2025-04-05"})["action"] == "restock_then_ship"
    assert d("2025-04-01", None, {"shortfall": 5, "eta": "2025-04-05"})["action"] == "restock_then_ship"
    assert d("2025-04-01", None, {"shortfall": 5, "eta": None})["reason"] == "eta_unknown"
    assert d("2025-04-01", None, {"error": "x"})["reason"] == "availability_unknown"
    assert d("2025-04-01", None, None)["reason"] == "availability_unknown"
    assert d("2025-04-01", None, {"shortfall": 0}, deadline_unconfirmed=True)["action"] == "ship"   # in stock: unaffected
    flagged = d("2025-04-01", None, {"shortfall": 5, "eta": "2025-04-05"}, deadline_unconfirmed=True)
    assert flagged["action"] == "skip" and flagged["reason"] == "deadline_unconfirmed"


@test
def pure_parsing():
    assert ps.coerce_json({"a": 1}) == {"a": 1}
    assert ps.coerce_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert ps.coerce_json('Sure! Here: {"a": 2} hope it helps') == {"a": 2}
    assert ps.coerce_json("no json") is None and ps.coerce_json(None) is None and ps.coerce_json("[1]") is None
    p = ps.validate_parsed("raw", {"lines": [{"item_name": "a4 PAPER", "quantity": "1,000"},
                                             {"item_name": "A4 paper", "quantity": 500},
                                             {"item_name": "Cardstok", "quantity": 5},
                                             {"item_name": "Balloons", "quantity": 5},
                                             {"item_name": "Glossy paper", "quantity": -3},
                                             {"item_name": "Glossy paper", "quantity": "abc"}],
                                   "needed_by": "April 5", "intents": ["order", "hack"]}, "2025-04-01")
    got = {l.item_name: l.quantity for l in p.lines}
    assert got == {"A4 paper": 1500, "Cardstock": 5}, got       # merged, fuzzy-fixed, validated
    assert set(p.unmatched) == {"Balloons", "Glossy paper"}, p.unmatched
    assert p.needed_by is None and p.intents == ["order"]
    assert ps.validate_parsed("r", {}, "2025-04-01").intents == ["quote"]   # ambiguous -> recoverable default
    past = ps.validate_parsed("r", {"lines": [], "needed_by": "2024-04-15"}, "2025-04-01")
    assert past.needed_by is None and past.deadline_unconfirmed == "2024-04-15"        # flagged, not guessed
    same = ps.validate_parsed("r", {"lines": [], "needed_by": "2025-04-01"}, "2025-04-01")
    assert same.needed_by == "2025-04-01" and same.deadline_unconfirmed is None         # same-day is valid
    assert ps.extract_discount_hint("We offered a 12.5% bulk discount") == 0.125
    assert ps.extract_discount_hint("discount of 7%") == 0.07 and ps.extract_discount_hint("none") is None


# ------------------------------------------------------------------ pipeline
@test
def happy_path_in_stock():
    orch, csa = fresh()
    item, mn, price = stocked(60)
    s0, c0 = stock(item), cash()
    reply, ctx = go(csa, orch, f"I need 50 {item} by 2025-04-10")
    exp = round(50 * price * 1.3, 2)
    assert "CONFIRMED" in reply and "(already placed)" not in reply, reply
    sale = rows(item, "sales")
    assert len(sale) == 1 and abs(float(sale.price.iloc[0]) - exp) < 0.01
    topups = rows(item, "stock_orders").iloc[1:]                    # row 0 is the seed purchase
    bought_units = int(sum(float(u) for u in topups.units)); bought_cost = sum(float(p) for p in topups.price)
    assert stock(item) == s0 - 50 + bought_units, (stock(item), s0, bought_units)
    assert abs(cash() - (c0 + exp - bought_cost)) < 0.01, (cash(), c0, exp, bought_cost)
    assert not ctx.restocks and ctx.orders[item]["delivery_date"] == "2025-04-01"


@test
def bulk_discount_and_clamp():
    orch, csa = fresh()
    item, mn, price = stocked(130)
    reply, ctx = go(csa, orch, f"Please order 120 {item} by 2025-04-10")
    q = ctx.quotes[item]
    assert q["discount_applied_pct"] == 5.0, q                      # tier cap 5%, history hint 10% -> min
    assert abs(q["line_total"] - round(120 * price * 1.3 * 0.95, 2)) < 0.01
    orch, csa = fresh()
    smolagents.HOOKS["discount"] = 90.0                              # LLM tries to give away the store
    reply, ctx = go(csa, orch, f"Please order 120 {item} by 2025-04-10")
    q = ctx.quotes[item]
    assert q["discount_applied_pct"] == 5.0 and q["discount_was_clamped"], q


@test
def shortfall_restock_then_sale_then_background_topup():
    orch, csa = fresh()
    item, mn, price = stocked(10)
    on_hand = stock(item, "2025-04-01")
    qty = on_hand + 300
    reply, ctx = go(csa, orch, f"I need {qty} {item} by 2025-05-30")
    assert "CONFIRMED, delivery by 2025-04-05" in reply, reply        # 300 units -> 4 day lead time
    assert ctx.restocks[item]["status"] == "placed"
    assert ctx.quotes[item]["discount_applied_pct"] == 0.0            # shortfall lines are never discounted
    r = rows(item, "stock_orders")
    assert any(int(float(u)) == 300 for u in r.units), r              # exactly the shortfall was bought
    # background replenishment ran AFTER the reply and topped up to target (5 x reorder point)
    assert ctx.background and stock(item) == mn * ps.REORDER_TARGET_MULT, (stock(item), mn)


@test
def shortfall_eta_after_deadline_has_no_side_effects():
    orch, csa = fresh()
    item, mn, price = stocked(10)
    on_hand = stock(item, "2025-04-01")
    before = len(pd.read_sql("SELECT * FROM transactions", ps.db_engine))
    reply, ctx = go(csa, orch, f"I need {on_hand + 300} {item} by 2025-04-02")
    after = len(pd.read_sql("SELECT * FROM transactions", ps.db_engine))
    assert "NOT FULFILLED" in reply and "cannot deliver by your requested date (earliest delivery 2025-04-05)" in reply, reply
    assert before == after, "a blocked order must write nothing"


@test
def insufficient_cash_denies_restock():
    orch, csa = fresh()
    c0 = cash()
    reply, ctx = go(csa, orch, "I need 2000000 Standard copy paper by 2026-01-01")
    assert ctx.restocks["Standard copy paper"]["status"] == "denied"
    assert "could not secure additional stock" in reply and len(rows("Standard copy paper", "sales")) == 0
    assert abs(cash() - c0) < 0.01 and cash() >= ps.MIN_CASH_RESERVE


@test
def idempotent_replay_does_not_double_sell():
    orch, csa = fresh()
    item, mn, price = stocked(100)
    go(csa, orch, f"I need 40 {item} by 2025-04-10", rid=7)
    reply, ctx = go(csa, orch, f"I need 40 {item} by 2025-04-10", rid=7)    # same request id again
    assert len(rows(item, "sales")) == 1 and ctx.orders[item]["replayed"], ctx.orders
    assert "already placed" in reply


@test
def repeat_of_a_restocked_order_does_not_buy_stock_again():
    orch, csa = fresh()
    orch.after_reply = lambda ctx: None                         # no background top-up: stock stays at 0 after the sale
    item, mn, price = stocked(10)
    qty = stock(item, "2025-04-01") + 300
    first, _ = go(csa, orch, f"I need {qty} {item} by 2025-05-30", rid=9)
    buys, sells = len(rows(item, "stock_orders")), len(rows(item, "sales"))
    assert stock(item) == 0
    second, ctx = go(csa, orch, f"I need {qty} {item} by 2025-05-30", rid=9)  # customer repeats the same line
    assert len(rows(item, "stock_orders")) == buys, "repeat must NOT trigger another purchase"
    assert len(rows(item, "sales")) == sells == 1
    assert "already placed" in second and "delivery by 2025-04-05" in second, second   # original delivery date preserved


@test
def transient_llm_failure_is_retried():
    orch, csa = fresh()
    item, mn, price = stocked(60)
    smolagents.HOOKS["fail"]["inventory_agent"] = 1
    reply, ctx = go(csa, orch, f"I need 50 {item} by 2025-04-10")
    assert "CONFIRMED" in reply and smolagents.HOOKS["calls"].count("inventory_agent:check") == 2


@test
def persistent_llm_failure_degrades_without_side_effects():
    orch, csa = fresh()
    item, mn, price = stocked(60)
    smolagents.HOOKS["fail"]["inventory_agent"] = 99
    c0 = cash(); before = len(pd.read_sql("SELECT * FROM transactions", ps.db_engine))
    reply, ctx = go(csa, orch, f"I need 50 {item} by 2025-04-10")
    assert "NOT FULFILLED: we could not verify availability" in reply, reply
    assert len(pd.read_sql("SELECT * FROM transactions", ps.db_engine)) == before and abs(cash() - c0) < 0.01
    assert smolagents.HOOKS["calls"].count("inventory_agent:check") == ps.AGENT_RETRIES + 1   # bounded


@test
def llm_skipping_a_line_is_detected_by_ledger():
    orch, csa = fresh()
    d = inv(); a, b = d.item_name.iloc[0], d.item_name.iloc[1]
    smolagents.HOOKS["skip"]["inventory_agent"] = {b}
    reply, ctx = go(csa, orch, f"I need 10 {a} and 10 {b} by 2025-04-10")
    assert ctx.orders.get(a, {}).get("status") == "confirmed" and b not in ctx.orders, ctx.orders
    assert "could not verify availability" in reply


@test
def partial_vs_all_or_nothing():
    orch, csa = fresh()
    d = inv(); a = d.item_name.iloc[0]; b = d.item_name.iloc[1]
    big = stock(b, "2025-04-01") + 300
    txt = f"I need 10 {a} and {big} {b} by 2025-04-02"
    reply, ctx = go(csa, orch, txt)
    assert ctx.orders[a]["status"] == "confirmed" and b not in ctx.orders, reply
    orch, csa = fresh(); ps.ALLOW_PARTIAL_FULFILLMENT = False
    before = len(pd.read_sql("SELECT * FROM transactions", ps.db_engine))
    reply, ctx = go(csa, orch, txt)
    assert not ctx.orders and not ctx.restocks and "held because another item" in reply, reply
    assert len(pd.read_sql("SELECT * FROM transactions", ps.db_engine)) == before


@test
def loop_is_bounded_and_escalates():
    orch, csa = fresh()
    item, mn, price = stocked(10000 // 1000)
    reply, ctx = go(csa, orch, f"How much is 10 {item}?", followup=lambda r, i: f"Still not happy, how much is 10 {item}?")
    assert smolagents.HOOKS["calls"].count("customer_support_agent:parse") == ps.MAX_ITERATIONS
    assert "human colleague" in reply


@test
def satisfied_customer_exits_loop_after_one_pass():
    orch, csa = fresh()
    item, mn, price = stocked(10)
    reply, ctx = go(csa, orch, f"How much is 10 {item}?", followup=lambda r, i: None)
    assert smolagents.HOOKS["calls"].count("customer_support_agent:parse") == 1 and "Quote total" in reply
    assert rows(item, "sales").empty, "a quote-only request must not sell anything"


@test
def inventory_only_request_skips_quote_and_order():
    orch, csa = fresh()
    item, mn, price = stocked(10)
    reply, ctx = go(csa, orch, f"Do you have 5 {item} in stock?")
    assert "available now" in reply and not ctx.quotes
    assert not any(c.startswith(("quote_agent", "order_agent")) for c in smolagents.HOOKS["calls"]), smolagents.HOOKS["calls"]


@test
def unknown_product_asks_to_clarify_and_calls_no_specialists():
    orch, csa = fresh()
    reply, ctx = go(csa, orch, "I need 10 unobtainium by 2025-04-10")
    assert "could not match" in reply and "unobtainium" in reply
    assert set(smolagents.HOOKS["calls"]) == {"customer_support_agent:parse"}


@test
def customer_reply_never_leaks_internals():
    orch, csa = fresh()
    d = inv(); a = d.item_name.iloc[0]
    secret = f"{cash():,.2f}"
    reply, ctx = go(csa, orch, f"I need {stock(a, '2025-04-01') + 300} {a} by 2025-05-30")
    low = reply.lower()
    for word in ["cash", "margin", "supplier cost", "reserve", "balance", "unit_price", secret]:
        assert word not in low, f"leaked: {word}"


@test
def phase1_runs_in_parallel():
    orch, csa = fresh()
    item, mn, price = stocked(10)
    smolagents.HOOKS["delay"]["inventory_agent"] = 0.6
    smolagents.HOOKS["delay"]["quote_agent"] = 0.6
    smolagents.HOOKS["delay"]["customer_support_agent"] = 0
    ctx = ps.RequestContext(1, "2025-04-01", "p1")
    ctx.parsed = ps.ParsedRequest("r", None, ["quote"], [ps.LineItem(item, 5)], [])
    t0 = time.time(); orch._phase1(ctx, {"inventory", "quote"}); dt = time.time() - t0
    assert 0.55 < dt < 1.1, f"expected ~0.6s if parallel, got {dt:.2f}s"


@test
def order_tool_refuses_unapproved_and_quoteless_lines():
    orch, csa = fresh()
    item, mn, price = stocked(10)
    ctx = ps.RequestContext(1, "2025-04-01", "t")
    ctx.parsed = ps.ParsedRequest("r", None, ["order"], [ps.LineItem(item, 5)], [])
    fin = ps.make_order_tools(ctx)[0]
    assert json.loads(fin(item_name=item))["reason"] == "not approved"
    ctx.decisions[item] = {"action": "ship", "approved": True, "delivery_date": "2025-04-01"}
    assert json.loads(fin(item_name=item))["reason"] == "no quote on file"
    assert rows(item, "sales").empty


@test
def agent_roster_matches_the_plan():
    orch, csa = fresh()
    # the plan's five agents exist as first-class components, and the orchestrator delegates THROUGH them
    assert isinstance(orch.inventory, ps.InventoryAgent) and isinstance(orch.quote, ps.QuoteAgent)
    assert isinstance(orch.order, ps.OrderAgent) and isinstance(csa, ps.CustomerSupportAgent)
    assert csa.orchestrator is orch
    roles = {c.role for c in (ps.InventoryAgent, ps.QuoteAgent, ps.OrderAgent, ps.CustomerSupportAgent)}
    assert roles == {"inventory_agent", "quote_agent", "order_agent", "customer_support_agent"}
    # run every code path (read, price, restock, sell, replenish) and check no OTHER agent name ever appears
    d = inv(); a, b = d.item_name.iloc[0], d.item_name.iloc[1]
    go(csa, orch, f"I need 120 {a} and {stock(b, '2025-04-01') + 300} {b} by 2025-05-30")
    seen = {c.split(":")[0] for c in smolagents.HOOKS["calls"]}
    modes = {c.split(":")[1] for c in smolagents.HOOKS["calls"]}
    assert seen == roles, seen
    assert {"parse", "check", "history", "pricing", "restock", "order"} <= modes, modes


@test
def least_privilege_tool_scoping():
    ctx = ps.RequestContext(1, "2025-04-01", "t")
    names = lambda tools: {t.name for t in tools}
    assert names(ps.make_inventory_read_tools(ctx)) == {"check_item_stock"}                     # Phase-1 agent cannot write
    assert names(ps.make_inventory_restock_tools(ctx)) == {"restock_for_order"}
    assert names(ps.make_inventory_replenish_tools(ctx)) == {"find_low_stock", "restock_for_replenishment"}
    assert names(ps.make_quote_history_tools(ctx)) == {"search_history"}
    assert names(ps.make_quote_pricing_tools(ctx)) == {"get_pricing_context", "price_line"}     # no DB writes at all
    assert names(ps.make_order_tools(ctx)) == {"finalize_sale"}                                 # the ONLY sale writer
    assert names(ps.make_customer_support_tools()) == {"lookup_catalog_item"}


@test
def past_deadline_is_flagged_not_guessed():
    orch, csa = fresh()
    d = inv(); a, b = d.item_name.iloc[0], d.item_name.iloc[1]
    big = stock(b, "2025-04-01") + 300
    reply, ctx = go(csa, orch, f"I need 10 {a} and {big} {b} by 2024-12-01")
    assert ctx.parsed.deadline_unconfirmed == "2024-12-01" and ctx.parsed.needed_by is None
    assert "earlier than your request date" in reply and "confirm the date you need" in reply, reply
    assert ctx.orders[a]["status"] == "confirmed"                       # in stock: does not depend on the deadline
    assert b not in ctx.orders and b not in ctx.restocks                # the deadline-dependent line: no purchase, no sale
    assert "please confirm the delivery date you need" in reply
    assert rows(b, "sales").empty


@test
def customer_confirms_date_and_order_completes():
    orch, csa = fresh()
    d = inv(); b = d.item_name.iloc[1]
    big = stock(b, "2025-04-01") + 300
    answers = iter([f"Sorry, I meant by 2025-05-30. I need {big} {b}."])
    reply, ctx = go(csa, orch, f"I need {big} {b} by 2024-12-01", followup=lambda r, i: next(answers, None))
    assert smolagents.HOOKS["calls"].count("customer_support_agent:parse") == 2
    assert "CONFIRMED, delivery by 2025-04-05" in reply and ctx.orders[b]["status"] == "confirmed", reply
    assert len(rows(b, "sales")) == 1


@test
def write_level_replay_returns_same_result_without_inserting():
    orch, csa = fresh()
    item, mn, price = stocked(20)
    kw = dict(key="k1", kind="sale", item_name=item, transaction_type="sales", units=5, price=9.99,
              date="2025-04-01", guard=lambda c: None, meta={"delivery_date": "2025-04-01", "total": 9.99})
    first = ps._commit_transaction(**kw)
    second = ps._commit_transaction(**kw)
    assert first["status"] == "committed" and not first.get("replayed")
    assert second["status"] == "committed" and second["replayed"] and second["transaction_id"] == first["transaction_id"]
    assert second["total"] == 9.99 and len(rows(item, "sales")) == 1
    denied = ps._commit_transaction(**{**kw, "key": "k2", "guard": lambda c: "nope"})
    assert denied == {"status": "denied", "reason": "nope"} and len(rows(item, "sales")) == 1


@test
def decided_but_unapproved_line_cannot_be_sold():
    orch, csa = fresh()
    item, mn, price = stocked(10)
    ctx = ps.RequestContext(1, "2025-04-01", "t")
    ctx.parsed = ps.ParsedRequest("r", None, ["order"], [ps.LineItem(item, 5)], [])
    ctx.quotes[item] = {"quantity": 5, "line_total": 1.0}
    fin = ps.make_order_tools(ctx)[0]
    for decision in ({"action": "restock_then_ship", "delivery_date": "2025-04-05"},       # restock not confirmed yet
                     {"action": "ship", "delivery_date": "2025-04-01"},                     # decided, never approved
                     {"action": "skip", "reason": "eta_after_deadline", "approved": True}):  # approved flag cannot override skip
        ctx.decisions[item] = decision
        assert json.loads(fin(item_name=item))["reason"] == "not approved", decision
    assert rows(item, "sales").empty


@test
def csa_fails_cleanly_when_neither_path_yields_data():
    orch, csa = fresh()
    item, mn, price = stocked(10)
    smolagents.HOOKS["parse_garbage"] = True
    before = len(pd.read_sql("SELECT * FROM transactions", ps.db_engine))
    reply, ctx = go(csa, orch, f"I need 10 {item} by 2025-04-10")
    assert smolagents.HOOKS["calls"].count("customer_support_agent:parse") == ps.AGENT_RETRIES + 1   # bounded
    assert set(smolagents.HOOKS["calls"]) == {"customer_support_agent:parse"}                         # nothing downstream ran
    assert len(pd.read_sql("SELECT * FROM transactions", ps.db_engine)) == before


@test
def csa_prompt_does_not_invite_flattened_final_answer():
    """Regression for: final_answer(lines=..., unmatched=..., ...) -> 'Argument lines is not in the tool's input schema'."""
    import inspect, re
    orch, csa = fresh()
    item, mn, price = stocked(10)
    go(csa, orch, f"How much is 10 {item}?")
    prompt = smolagents.HOOKS["last_task"]["customer_support_agent:parse"]
    assert "ONE argument named 'answer'" in prompt, "prompt must name final_answer's single argument"
    assert "Do NOT pass lines, unmatched, needed_by, intents or notes as separate arguments" in prompt
    assert "ONLY this JSON object" not in prompt, "the old wording that led the model to spread the object into arguments"
    # any final_answer(...) call spelled out in the prompt may only use the real schema (a single `answer`)
    for call in re.findall(r"final_answer\(([^)]*)\)", prompt):
        assert set(re.findall(r"(\w+)\s*=", call)) <= {"answer"}, call


@test
def csa_recovers_when_a_parse_run_fails_once():
    """The failure in the log surfaces as a failed run. categorize must retry, boundedly, and then succeed."""
    orch, csa = fresh()
    item, mn, price = stocked(10)
    smolagents.HOOKS["fail"]["customer_support_agent"] = 1
    reply, ctx = go(csa, orch, f"How much is 10 {item}?")
    assert smolagents.HOOKS["calls"].count("customer_support_agent:parse") == 2 and "Quote total" in reply, reply


@test
def multi_item_request_parses_every_line():
    """The log's failing case: several items in one message."""
    orch, csa = fresh()
    d = inv(); a, b, c = d.item_name.iloc[0], d.item_name.iloc[1], d.item_name.iloc[2]
    reply, ctx = go(csa, orch, f"How much are 10 {a}, 20 {b} and 30 {c}?")
    assert {l.item_name: l.quantity for l in ctx.parsed.lines} == {a: 10, b: 20, c: 30}, ctx.parsed.lines
    assert reply.count(" x ") == 3 and "Quote total" in reply


def _write_sample_csv():
    pd.DataFrame({
        "request": ["I need 20 A4 paper by 2025-04-20", "How much is 10 Cardstock?", "I need 5 unobtainium"],
        "job": ["a", "b", "c"], "event": ["x", "y", "z"],
        "request_date": ["04/03/25", "04/01/25", "04/02/25"],
    }).to_csv("quote_requests_sample.csv", index=False)


@test
def importing_the_package_needs_no_api_key():
    """Regression for: OpenAIError at import time when OPENAI_API_KEY is not set."""
    import subprocess
    env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    code = ("import lib.agents as a, workflow, main; "
            "assert a.get_model.cache_info().currsize == 0, 'model was built at import time'; print('lazy-ok')")
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, cwd=WORK)
    assert r.returncode == 0 and "lazy-ok" in r.stdout, r.stderr[-400:]


@test
def limit_and_no_sleep_flags():
    import main as entry
    _write_sample_csv(); fresh()
    all_sleeps = []
    old_sleep = ps.time.sleep; ps.time.sleep = lambda s: all_sleeps.append(s)
    pauses = lambda: [s for s in all_sleeps if s == 1]          # the harness pause; retry backoff sleeps are 0.0 here
    try:
        out = entry.main(["--limit", "2", "--no-sleep"])
        assert [r["request_id"] for r in out] == [1, 2] and pauses() == [], (len(out), all_sleeps)   # earliest two, no pause
        fresh(); all_sleeps.clear()
        out = entry.main(["--limit", "2"])
        assert len(out) == 2 and len(pauses()) == 2, all_sleeps                                      # pause kept by default
        fresh(); all_sleeps.clear()
        assert len(entry.main(["--no-sleep"])) == 3                                                  # no limit = all requests
    finally:
        ps.time.sleep = old_sleep


@test
def limit_must_be_positive():
    import main as entry
    for bad in (["--limit", "0"], ["--limit", "-3"], ["--limit", "x"]):
        try:
            entry.parse_args(bad)
        except SystemExit as exc:
            assert exc.code == 2, bad
        else:
            raise AssertionError(f"{bad} should be rejected")


@test
def project_starter_wrapper_runs_the_same_entry_point():
    import main as entry, project_starter
    assert project_starter.main is entry.main
    src = open(project_starter.__file__).read()
    assert src.count("\n") < 12 and "def " not in src, "the wrapper must stay a thin shim"


@test
def cash_in_conn_matches_starter_get_cash_balance():
    orch, csa = fresh()
    item, mn, price = stocked(50)
    for key, ttype, units, amount, date in [("a", "sales", 5, 123.45, "2025-04-01"), ("b", "stock_orders", 40, 77.77, "2025-04-02"),
                                            ("c", "sales", 9, 310.10, "2025-04-03"), ("d", "stock_orders", 3, 5.05, "2025-04-03")]:
        ps._commit_transaction(key=key, kind=ttype, item_name=item, transaction_type=ttype, units=units,
                               price=amount, date=date, guard=lambda c: None)
    with ps.db_engine.connect() as conn:
        for day in ["2025-01-01", "2025-03-31", "2025-04-01", "2025-04-02", "2025-04-03", "2999-01-01"]:
            assert abs(ps._cash_in_conn(conn, day) - ps.get_cash_balance(as_of_date=day)) < 1e-6, day


@test
def atomic_stock_guard_catches_a_race():
    orch, csa = fresh()
    item, mn, price = stocked(50)
    ctx = ps.RequestContext(1, "2025-04-01", "t")
    ctx.parsed = ps.ParsedRequest("r", None, ["order"], [ps.LineItem(item, 40)], [])
    ctx.stock[item] = {"shortfall": 0}
    ctx.quotes[item] = {"quantity": 40, "line_total": 10.0}
    ctx.decisions[item] = {"action": "ship", "approved": True, "delivery_date": "2025-04-01"}
    # another customer buys almost everything between our check and our write
    ps._commit_transaction(key="rival", kind="sale", item_name=item, transaction_type="sales",
                           units=stock(item, "2025-04-01") - 5, price=1.0, date="2025-04-01", guard=lambda c: None)
    out = json.loads(ps.make_order_tools(ctx)[0](item_name=item))
    assert out["status"] == "denied" and out["reason"] == "stock changed", out
    assert len(rows(item, "sales")) == 1                                   # only the rival's sale exists


@test
def replenishment_is_idempotent_and_only_when_low():
    orch, csa = fresh()
    ctx = ps.RequestContext(1, "2025-04-01", "r1")
    assert ps._low_stock_items("2025-04-01") == []                         # seed data starts above reorder points
    item, mn, price = stocked(10)
    ps._commit_transaction(key="drain", kind="sale", item_name=item, transaction_type="sales",
                           units=stock(item, "2025-04-01") - 1, price=1.0, date="2025-04-01", guard=lambda c: None)
    assert [x["item_name"] for x in ps._low_stock_items("2025-04-01")] == [item]
    orch._replenish(ctx); n = len(rows(item, "stock_orders"))
    orch._replenish(ps.RequestContext(1, "2025-04-01", "r1"))               # second scan: nothing low, nothing bought
    assert len(rows(item, "stock_orders")) == n and stock(item) == mn * ps.REORDER_TARGET_MULT


@test
def replenishment_respects_cash_reserve():
    orch, csa = fresh()
    item, mn, price = stocked(10)
    ps._commit_transaction(key="drain", kind="sale", item_name=item, transaction_type="sales",
                           units=stock(item, "2025-04-01") - 1, price=1.0, date="2025-04-01", guard=lambda c: None)
    old = ps.MIN_CASH_RESERVE; ps.MIN_CASH_RESERVE = 10 ** 9
    try:
        ctx = ps.RequestContext(1, "2025-04-01", "r")
        orch._replenish(ctx)
        assert ctx.background and ctx.background[0]["status"] == "denied"
    finally: ps.MIN_CASH_RESERVE = old


@test
def full_harness_runs_and_writes_results():
    pd.DataFrame({
        "request": ["I need 20 A4 paper by 2025-04-20", "How much is 10 Cardstock?", "I need 5 unobtainium"],
        "job": ["a", "b", "c"], "event": ["x", "y", "z"],
        "request_date": ["04/03/25", "04/01/25", "04/02/25"],
    }).to_csv("quote_requests_sample.csv", index=False)
    fresh(); old_sleep = ps.time.sleep; ps.time.sleep = lambda s: None
    try: out = ps.run_test_scenarios()
    finally: ps.time.sleep = old_sleep
    assert [r["request_id"] for r in out] == [1, 2, 3] and [r["request_date"] for r in out] == ["2025-04-01", "2025-04-02", "2025-04-03"]
    df = pd.read_csv("test_results.csv")
    assert list(df.columns) == ["request_id", "request_date", "cash_balance", "inventory_value", "response"]
    assert "could not match" in df.response.iloc[1]


if __name__ == "__main__":
    ok = sum(1 for _, p, _ in RESULTS if p)
    for name, passed, err in RESULTS:
        print(("PASS " if passed else "FAIL ") + name)
        if not passed: print(err)
    print(f"\n{ok}/{len(RESULTS)} passed")
    shutil.rmtree(WORK, ignore_errors=True)
    sys.exit(0 if ok == len(RESULTS) else 1)