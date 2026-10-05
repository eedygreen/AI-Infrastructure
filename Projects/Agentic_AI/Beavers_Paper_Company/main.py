"""Entry point: python main.py [--limit N] [--no-sleep]

Also reachable as `python project_starter.py` (same flags).
"""
import argparse

from lib import config
from workflow import run_test_scenarios


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run the Munder Difflin multi-agent test scenarios.")
    parser.add_argument("--limit", type=int, default=None, metavar="N",
                        help="process only the first N requests (by date); each request costs many model calls")
    parser.add_argument("--no-sleep", action="store_true",
                        help="skip the 1-second pause between requests")
    parser.add_argument("--resume", action="store_true",
                        help="continue a stopped run from test_results.csv and the existing database " \
                        "(Without this flag a run always starts over)")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be 1 or more")
    return args


def main(argv=None):
    args = parse_args(argv)
    config.load_env()
    return run_test_scenarios(limit=args.limit, no_sleep=args.no_sleep, resume=args.resume)


if __name__ == "__main__":
    main()
