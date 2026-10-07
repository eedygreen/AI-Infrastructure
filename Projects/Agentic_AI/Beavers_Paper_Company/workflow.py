"""The test harness: runs the sample requests through the agents and writes test_results.csv."""

import pandas as pd
import time
from typing import Optional
from utils import logger
from lib.agents import (
    CustomerSupportAgent,
    InventoryAgent,
    OrchestratorAgent,
    OrderAgent,
    QuoteAgent
)
from lib.StateMachine import CRASH_MESSAGE, StateMachine
from lib.starter_utils import generate_financial_report


def run_test_scenarios(limit: Optional[int] = None, no_sleep: bool = False, resume: bool = False):
    """Run the sample requests through the agents, one at a time, and write test_results.csv.

    `limit` handles only the first N requests (by date). `no_sleep` skips the pause between requests.
    `resume` continues a stopped run instead of starting over. Returns the list of result rows."""
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

    state = StateMachine(all_requests, resume=resume)

    inventory_agent = InventoryAgent()
    quote_agent = QuoteAgent()
    order_agent = OrderAgent()
    orchestrator = OrchestratorAgent(
        state.run_id,
        inventory_agent,
        quote_agent,
        order_agent
    )
    customer_agent = CustomerSupportAgent(
        orchestrator=orchestrator,
        run_id=state.run_id
    )

    for request_number, (_, row) in enumerate(quote_requests_sample.iterrows(), start=1):
        if state.is_finished(request_number):
            continue                        # done by the run we are resuming

        request_date = row["request_date"].strftime("%Y-%m-%d")

        logger.info(f"\n=== Request {request_number} ===")
        logger.info(f"Context: {row['job']} organizing {row['event']}")
        logger.info(f"Request Date: {request_date}")
        logger.info(f"Cash Balance: ${state.cash:.2f}")
        logger.info(f"Inventory Value: ${state.inventory:.2f}")

        # Process request
        request_with_date = f"{row['request']} (Date of request: {request_date})"

        try:
            response = customer_agent.handle(request_with_date, request_date, request_number)
            parse_failed = customer_agent.last_ctx.parsed.failed
        except Exception as e:
            logger.error("request %s crashed: %s: %s", request_number, type(e).__name__, e)   # the detail goes to the log
            response = f"{CRASH_MESSAGE}."                                                      # the customer sees none of it
            parse_failed = False
        orchestrator.drain_background()     # test harness only: keep the books determinstic

        # Update state
        state.update_books(request_date)

        logger.info(f"Response: {response}")
        logger.info(f"Updated Cash: ${state.cash:.2f}")
        logger.info(f"Updated Inventory: ${state.inventory:.2f}")

        # Save results (the state also counts the requests in a row the model could not understand)
        state.record(request_number, request_date, response, parse_failed)
        if state.must_stop():
            raise SystemExit(state.stop_message())
        if not no_sleep:
            time.sleep(1)

    # Final report
    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    logger.info("\n===== FINAL FINANCIAL REPORT =====")
    logger.info(f"Final Cash: ${final_report['cash_balance']:.2f}")
    logger.info(f"Final Inventory: ${final_report['inventory_value']:.2f}")

    return state.results
