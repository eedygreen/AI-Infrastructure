"""TEST-ONLY stand-in for smolagents. Mimics the @tool validation rules that matter
(docstring Args for every parameter, return type hint) and replaces the LLM with a
scripted policy that tests can sabotage via HOOKS."""
import inspect, json, re, threading

HOOKS = {"fail": {}, "skip": {}, "discount": None, "calls": [], "lock": threading.Lock(), "delay": {},
         "last_task": {}, "parse_garbage": False, "parse_via_final_answer": False}

class StubTool:
    def __init__(self, func): self.func, self.name = func, func.__name__
    def __call__(self, *a, **kw): return self.func(*a, **kw)

def tool(func):
    sig, doc = inspect.signature(func), inspect.getdoc(func) or ""
    if "return" not in func.__annotations__:
        raise TypeError(f"Tool return type not found: add a return type hint to {func.__name__}")
    if not doc.split("\n")[0].strip():
        raise ValueError(f"{func.__name__}: missing description")
    args_block = doc.split("Args:")[1] if "Args:" in doc else ""
    for p in sig.parameters:
        if p not in ("self",) and not re.search(rf"^\s*{p}\s*(\(.*?\))?\s*:", args_block, re.M):
            raise ValueError(f"{func.__name__}: parameter '{p}' not described in docstring Args")
        if sig.parameters[p].annotation is inspect._empty:
            raise TypeError(f"{func.__name__}: parameter '{p}' has no type hint")
    return StubTool(func)

class OpenAIServerModel:
    def __init__(self, **kw): self.kw = kw

class ToolCallingAgent:
    def __init__(self, tools, model, max_steps=20, name=None, description=None):
        self.tools, self.name = {t.name: t for t in tools}, name
    def run(self, task):
        with HOOKS["lock"]:
            HOOKS["calls"].append(f"{self.name}:{_mode(self)}")
            HOOKS["last_task"][f"{self.name}:{_mode(self)}"] = task
        d = HOOKS["delay"].get(self.name)
        if d: import time; time.sleep(d)
        n = HOOKS["fail"].get(self.name, 0)
        if n:
            HOOKS["fail"][self.name] = n - 1
            raise RuntimeError(f"simulated LLM/API failure in {self.name}")
        return globals()["_policy"](self, task)

def _lines(task):
    m = re.search(r"LINES_JSON: (\[.*\])", task, re.S)
    return json.loads(m.group(1)) if m else []

MODES = {"check_item_stock": "check", "restock_for_order": "restock", "find_low_stock": "replenish",
         "search_history": "history", "get_pricing_context": "pricing", "finalize_sale": "order",
         "lookup_catalog_item": "parse"}

def _mode(agent):
    return next(m for t, m in MODES.items() if t in agent.tools)

def _policy(agent, task):
    skip = HOOKS["skip"].get(agent.name, set())
    t, mode = agent.tools, _mode(agent)
    if mode == "parse":
        if HOOKS["parse_garbage"]:
            return "I am sorry, I could not do that."               # model neither calls the tool nor returns JSON
        payload = json.loads(_fake_parse(task))
        return json.dumps(payload)                                  # final_answer(answer=<JSON string>)
    if mode == "check":
        for l in _lines(task):
            if l["item_name"] not in skip: t["check_item_stock"](item_name=l["item_name"])
    elif mode == "history":
        t["search_history"](search_terms=[l["item_name"] for l in _lines(task)][:3] or ["paper"])
    elif mode == "pricing":
        ctx = json.loads(t["get_pricing_context"]())
        for l in _lines(task):
            if l["item_name"] in skip: continue
            pct = HOOKS["discount"] if HOOKS["discount"] is not None else ctx[l["item_name"]]["recommended_discount_pct"]
            t["price_line"](item_name=l["item_name"], discount_pct=pct)
    elif mode == "restock":
        for l in _lines(task):
            if l["item_name"] not in skip: t["restock_for_order"](item_name=l["item_name"])
    elif mode == "order":
        for l in _lines(task):
            if l["item_name"] not in skip: t["finalize_sale"](item_name=l["item_name"])
    elif mode == "replenish":
        for it in json.loads(t["find_low_stock"]()):
            t["restock_for_replenishment"](item_name=it["item_name"])
    return "done"

def _fake_parse(task):
    from lib.database import CATALOG
    msg = task.split("CUSTOMER MESSAGE:", 1)[1]
    lines, unmatched = [], []
    for name in CATALOG:
        m = re.search(rf"([\d,]+)\s*(reams?\s+of\s+|sheets\s+of\s+|units\s+of\s+)?{re.escape(name)}", msg, re.I)
        if m:
            q = int(m.group(1).replace(",", ""))
            if m.group(2) and m.group(2).lower().startswith("ream"): q *= 500
            lines.append({"item_name": name, "quantity": q})
    if "unobtainium" in msg.lower(): unmatched.append("unobtainium")
    nb = re.search(r"by (\d{4}-\d{2}-\d{2})", msg)
    low = msg.lower()
    intents = ["inventory"] if ("do you have" in low or "in stock" in low) else \
              ["quote"] if ("how much" in low or "price" in low) else ["order"]
    return json.dumps({"lines": lines, "unmatched": unmatched, "needed_by": nb.group(1) if nb else None,
                       "intents": intents, "notes": ""})
