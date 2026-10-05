"""The test harness: runs the sample requests through the agents and writes test_results.csv."""

import pandas as pd
import time
from datetime import datetime
from typing import Optional
from utils import logger

from lib.agents import CustomerSupportAgent, InventoryAgent, OrchestratorAgent, OrderAgent, QuoteAgent
from lib.database import ensure_runtime_tables
from lib.starter_utils import db_engine, generate_financial_report, init_database


# Run your test scenarios by writing them here. Make sure to keep track of them.

def run_test_scenarios(limit: Optional[int] = None, no_sleep: bool = False):
    
    logger.info("Initializing Database...")
    init_database(db_engine)
    ensure_runtime_tables(reset=True)
    try:
        quote_requests_sample = pd.read_csv("quote_requests_sample.csv")
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"], format="%m/%d/%y", errors="coerce"
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date")
        if limit:
            quote_requests_sample = quote_requests_sample.head(limit)
    except Exception as e:
        print(f"FATAL: Error loading test data: {e}")
        return

    # Get initial state
    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]

    run_id = datetime.now().strftime("%Y%m%d%H%M%S")
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

    results = []
    for request_number, (_, row) in enumerate(quote_requests_sample.iterrows(), start=1):
        request_date = row["request_date"].strftime("%Y-%m-%d")

        print(f"\n=== Request {request_number} ===")
        print(f"Context: {row['job']} organizing {row['event']}")
        print(f"Request Date: {request_date}")
        print(f"Cash Balance: ${current_cash:.2f}")
        print(f"Inventory Value: ${current_inventory:.2f}")

        # Process request
        request_with_date = f"{row['request']} (Date of request: {request_date})"

        try:
            response = customer_agent.handle(request_with_date, request_date, request_number)
        except Exception as e:
            logger.error("request %s crashed", request_number)
            response = f"We could not process this request right now({type(e).__name__})."
        orchestrator.drain_background()     # test harness only: keep the books determinstic

        # Update state
        report = generate_financial_report(request_date)
        current_cash = report["cash_balance"]
        current_inventory = report["inventory_value"]

        print(f"Response: {response}")
        print(f"Updated Cash: ${current_cash:.2f}")
        print(f"Updated Inventory: ${current_inventory:.2f}")

        results.append(
            {
                "request_id": request_number,
                "request_date": request_date,
                "cash_balance": current_cash,
                "inventory_value": current_inventory,
                "response": response,
            }
        )

        if not no_sleep:
            time.sleep(1)

    # Final report
    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    print("\n===== FINAL FINANCIAL REPORT =====")
    print(f"Final Cash: ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory: ${final_report['inventory_value']:.2f}")

    # Save results
    pd.DataFrame(results).to_csv("test_results.csv", index=False)
    return results
