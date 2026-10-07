"""The state of one test run: what is finished, the run id, the books, and when the run must stop."""

import os
from datetime import datetime

import pandas as pd

from lib import config
from lib.agents import PARSE_FAILED_MESSAGE
from lib.database import ensure_runtime_tables, load_run_id, save_run_id
from lib.starter_utils import db_engine, generate_financial_report, init_database
from utils import logger

RESULT_COLUMNS = ["request_id", "request_date", "cash_balance", "inventory_value", "response"]
CRASH_MESSAGE = "We could not process this request right now"      # the reply recorded when a request raises


class StateMachine:
    """The state of one test run.

    It starts a fresh run, or picks up a stopped one when asked to resume. It then keeps the run id, the rows
    finished so far, the current cash and inventory value, and how many requests in a row the model could not
    understand. It saves test_results.csv after every request, and says when the run must stop."""

    def __init__(self, all_requests, resume: bool = False):
        """Start a fresh run, or resume a stopped one (this prepares the database either way).

        `all_requests` is the full, date-ordered list of sample requests: a resumed run checks its saved rows
        against it."""
        finished, run_id = self.read_saved_run(all_requests) if resume else (None, None)
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
            ensure_runtime_tables()

        self.run_id = run_id
        self.results = list(finished)               # one row per finished request
        self.finished_count = len(finished)         # how many of them came from the run being resumed
        self.parse_failures_in_a_row = 0
        # Get initial state
        if finished:
            self.cash, self.inventory = finished[-1]["cash_balance"], finished[-1]["inventory_value"]
        else:
            initial_date = all_requests["request_date"].min().strftime("%Y-%m-%d")
            report = generate_financial_report(initial_date)
            self.cash = report["cash_balance"]
            self.inventory = report["inventory_value"]

    @staticmethod
    def is_failed_row(response) -> bool:
        """True for a saved row that records a failure to process a request (model down, or a crash), not a real outcome."""
        response = str(response)
        return PARSE_FAILED_MESSAGE in response or response.startswith(CRASH_MESSAGE)

    @staticmethod
    def read_saved_run(sample):
        """Read what a stopped run left behind, so --resume can carry on from it.

        Returns (finished_rows, run_id), or (None, None) when no request was really finished. Refuses (SystemExit) when what
        is on disk does not fit together: stopping is better than carrying on with wrong books."""
        def refuse(why):
            """Stop with a clear message instead of resuming on books that do not fit together."""
            raise SystemExit(f"Cannot resume: {why}. Run without --resume to start over.")

        if not os.path.exists("test_results.csv"):
            return None, None
        saved = pd.read_csv("test_results.csv")
        if list(saved.columns) != RESULT_COLUMNS:
            refuse("test_results.csv does not have the expected columns")
        rows = saved.to_dict("records")
        while rows and StateMachine.is_failed_row(rows[-1]["response"]):      # never really processed: they will be redone
            rows.pop()
        if not rows:
            return None, None
        run_id = load_run_id()
        if not run_id:
            refuse("the database holds no record of the earlier run")
        dates = [day.strftime("%Y-%m-%d") for day in sample["request_date"]]
        if [saved["request_date"] for saved in rows] != dates[:len(rows)]:
            refuse("the saved results do not match quote_requests_sample.csv")
        last = rows[-1]
        report = generate_financial_report(last["request_date"])
        if (abs(report["cash_balance"] - last["cash_balance"]) > 0.01
                or abs(report["inventory_value"] - last["inventory_value"]) > 0.01):
            refuse(f"the database does not match the last saved result (cash {report['cash_balance']:.2f} in the database "
                   f"vs {last['cash_balance']:.2f} saved; inventory {report['inventory_value']:.2f} vs "
                   f"{last['inventory_value']:.2f})")
        return rows, run_id

    def is_finished(self, request_number: int) -> bool:
        """True if the run being resumed already finished this request."""
        return request_number <= self.finished_count

    def update_books(self, request_date: str) -> None:
        """Read cash and inventory value as of a request's date (the 'Update state' step)."""
        report = generate_financial_report(request_date)
        self.cash = report["cash_balance"]
        self.inventory = report["inventory_value"]

    def record(self, request_number: int, request_date: str, response: str, parse_failed: bool) -> None:
        """Add a finished request to the results, and save them at once.

        Call update_books first, so the row carries the books after this request."""
        self.parse_failures_in_a_row = self.parse_failures_in_a_row + 1 if parse_failed else 0
        self.results.append(
            {
                "request_id": request_number,
                "request_date": request_date,
                "cash_balance": self.cash,
                "inventory_value": self.inventory,
                "response": response,
            }
        )
        self.save()                         # keep everything finished so far, even if the run stops or crashes next

    def save(self) -> None:
        """Write test_results.csv. Called after every request, so a stop or a crash keeps everything finished so far.
        Written to a temporary file and then renamed, so the file is never left half-written."""
        pd.DataFrame(self.results).to_csv("test_results.csv.tmp", index=False)
        os.replace("test_results.csv.tmp", "test_results.csv")

    def must_stop(self) -> bool:
        """True when too many requests in a row failed because the model could not be used.

        A rejected key or an unreachable model fails EVERY request the same way: stop instead of writing a run of apologies."""
        return self.parse_failures_in_a_row >= config.MAX_CONSECUTIVE_PARSE_FAILURES

    def stop_message(self) -> str:
        """What to tell the person running the tests when the run stops."""
        return (f"Stopping: {self.parse_failures_in_a_row} requests in a row could not be understood because "
                "the model call failed. Check OPENAI_API_KEY and network access, then rerun. "
                "Results so far were saved to test_results.csv")
