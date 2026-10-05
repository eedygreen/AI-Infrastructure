"""The test harness: runs the sample requests through the agents and writes test_results.csv."""

import pandas as pd
import time, os
from datetime import datetime
from typing import Optional
from utils import logger
from lib import config
from lib.agents import (
    PARSE_FAILED_MESSAGE,
    CustomerSupportAgent,
    InventoryAgent,
    OrchestratorAgent,
    OrderAgent,
    QuoteAgent
)
from lib.database import (
    ensure_runtime_tables,
    load_run_id,
    save_run_id
)
from lib.starter_utils import (
    db_engine,
    generate_financial_report,
    init_database
)

RESULT_COLUMNS = ["request_id", "request_date", "cash_balance", "inventory_value", "response"]
CRASH_MESSAGE = "We could not process this request right now"      # the reply recorded when a request raises


def _is_failed_row(response) -> bool:
    """True for a saved row that records a failure to process a request (model down, or a crash), not a real outcome."""
    response = str(response)
    return PARSE_FAILED_MESSAGE in response or response.startswith(CRASH_MESSAGE)

def _resume_state(sample):
    """Read what a stopped run left behind, so --resume can carry on from it.

    Returns (finished_rows, run_id), or (None, None) when no request was really finished. Refuses (SystemExit) when what
    is on disk does not fit together: stopping is better than carrying on with wrong books."""
    def refuse(why):
        raise SystemExit(f"Cannot resume: {why}. Run without --resume to start over.")

    if not os.path.exists("test_results.csv"):
        return None, None
    saved = pd.read_csv("test_results.csv")
    if list(saved.columns) != RESULT_COLUMNS:
        refuse("test_results.csv does not have the expected columns")
    rows = saved.to_dict("records")
    while rows and _is_failed_row(rows[-1]["response"]):      # never really processed: they will be redone
        rows.pop()
    if not rows:
        return None, None
    run_id = load_run_id()
    if not run_id:
        refuse("the database holds no record of the earlier run")
    dates = [d.strftime("%Y-%m-%d") for d in sample["request_date"]]
    if [r["request_date"] for r in rows] != dates[:len(rows)]:
        refuse("the saved results do not match quote_requests_sample.csv")
    last = rows[-1]
    report = generate_financial_report(last["request_date"])
    if (abs(report["cash_balance"] - last["cash_balance"]) > 0.01
            or abs(report["inventory_value"] - last["inventory_value"]) > 0.01):
        refuse(f"the database does not match the last saved result (cash {report['cash_balance']:.2f} in the database "
               f"vs {last['cash_balance']:.2f} saved; inventory {report['inventory_value']:.2f} vs "
               f"{last['inventory_value']:.2f})")
    return rows, run_id

def _save_results(results):
    """Write test_results.csv. Called after every request, so a stop or a crash keeps everything finished so far.
    Written to a temporary file and then renamed, so the file is never left half-written."""
    pd.DataFrame(results).to_csv("test_results.csv.tmp", index=False)
    os.replace("test_results.csv.tmp", "test_results.csv")

def run_test_scenarios(limit: Optional[int] = None, no_sleep: bool = False, resume: bool = False):
    try:
        quote_requests_sample = pd.read_csv("quote_requests_sample.csv")
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"], format="%m/%d/%y", errors="coerce"
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date")
    except Exception as e:
        logger.error(f"FATAL: Error loading test data: {e}")
        return

    all_requests = quote_requests_sample            # the full, date-ordered list: --resume checks the saved rows against it
    if limit:
        quote_requests_sample = quote_requests_sample.head(limit)

    finished, run_id = _resume_state(all_requests) if resume else (None, None)
    if finished is None:
        if resume:
            logger.info("Nothing to resume: starting from the beginning.")
        logger.info("Initializing Database...")
        init_database(db_engine=db_engine)
        ensure_runtime_tables(reset=True)
        run_id = datetime.now().strftime("%Y%m%d%H%M%S")
        save_run_id(run_id=run_id)
        finished = []
    else:
        logger.info("Resuming run %s after request %s", run_id, len(finished))
        logger.info(f"Resumging run {run_id}: requests 1 to {len(finished)} are already done.")
        ensure_runtime_tables()

    # Get initial state
    if finished:
        current_cash, current_inventory = finished[-1]["cash_balance"], finished[-1]["inventory_value"]
    else:
        initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
        report = generate_financial_report(initial_date)
        current_cash = report["cash_balance"]
        current_inventory = report["inventory_value"]

    inventory_agent = InventoryAgent()
    quote_agent = QuoteAgent()
    order_agent = OrderAgent()
    orchestrator = OrchestratorAgent(
        run_id,
        inventory_agent,
        quote_agent,
        order_agent
    )
    customer_agent = CustomerSupportAgent(
        orchestrator=orchestrator,
        run_id=run_id
    )

    results = list(finished)
    parse_failures_in_a_row = 0
    for request_number, (_, row) in enumerate(quote_requests_sample.iterrows(), start=1):
        if request_number <= len(finished):
            continue                        # done by the run we are resuming

        request_date = row["request_date"].strftime("%Y-%m-%d")

        logger.info(f"\n=== Request {request_number} ===")
        logger.info(f"Context: {row['job']} organizing {row['event']}")
        logger.info(f"Request Date: {request_date}")
        logger.info(f"Cash Balance: ${current_cash:.2f}")
        logger.info(f"Inventory Value: ${current_inventory:.2f}")

        # Process request
        request_with_date = f"{row['request']} (Date of request: {request_date})"

        try:
            response = customer_agent.handle(request_with_date, request_date, request_number)
            parse_failed = customer_agent.last_ctx.parsed.failed
        except Exception as e:
            logger.error("request %s crashed", request_number)
            response = f"{CRASH_MESSAGE}({type(e).__name__})."
            parse_failed = False
        orchestrator.drain_background()     # test harness only: keep the books determinstic

        # A rejected key or an unreachable model fails EVERY request the same way: stop instead of writing a run of apologies.
        parse_failures_in_a_row = parse_failures_in_a_row + 1 if parse_failed else 0
        stop_run = parse_failures_in_a_row >= config.MAX_CONSECUTIVE_PARSE_FAILURES

        # Update state
        report = generate_financial_report(request_date)
        current_cash = report["cash_balance"]
        current_inventory = report["inventory_value"]

        logger.info(f"Response: {response}")
        logger.info(f"Updated Cash: ${current_cash:.2f}")
        logger.info(f"Updated Inventory: ${current_inventory:.2f}")

        results.append(
            {
                "request_id": request_number,
                "request_date": request_date,
                "cash_balance": current_cash,
                "inventory_value": current_inventory,
                "response": response,
            }
        )
        # Save results
        _save_results(results=results)
        if stop_run:
            raise SystemExit(f"Stopping: {parse_failures_in_a_row} requests in a row could not be understood because "
                             "the model call failed. Check OPENAI_API_KEY and network access, then rerun. "
                             "Results so far were saved to test_results.csv")
        if not no_sleep:
            time.sleep(1)

    # Final report
    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    logger.info("\n===== FINAL FINANCIAL REPORT =====")
    logger.info(f"Final Cash: ${final_report['cash_balance']:.2f}")
    logger.info(f"Final Inventory: ${final_report['inventory_value']:.2f}")

    return results
