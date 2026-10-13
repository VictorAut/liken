"""pyinstrument sampling run of one liken workload.

Runs the workload phase under a sampling profiler, prints a console
summary, and always writes the full HTML call stack to `results/` — that
HTML is the artifact worth opening in a browser.
"""

from __future__ import annotations

import argparse
import logging
import time

from common import add_base_args
from common import load_workload
from common import prepare
from common import results_path
from common import run
from common import source_name


try:
    from pyinstrument import Profiler
except ImportError as err:
    raise SystemExit(
        "error: pyinstrument is not installed; install the dev group on a supported Python (python_full_version < '3.15')"
    ) from err


logger = logging.getLogger(__name__)


# MAIN:


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="pyinstrument sampling run of one liken workload.")
    add_base_args(parser)
    parser.add_argument("--interval", type=float, default=0.001, help="Sampling interval in seconds. Default 0.001.")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Profile one workload run with pyinstrument and write the HTML."""

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    args = parse_args(argv)

    try:
        df, entry = load_workload(args)
    except ValueError as err:
        raise SystemExit(f"error: {err}") from err

    built = prepare(df, entry, phase=args.phase)

    profiler = Profiler(interval=args.interval)
    start = time.perf_counter()
    profiler.start()
    run(df, entry, built, phase=args.phase)
    profiler.stop()
    elapsed = time.perf_counter() - start

    destination = results_path("instrument", entry.name, args.size, source_name(args.csv), "html", output=args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(profiler.output_html(), encoding="utf-8")

    logger.info("pyinstrument, %.3f seconds wall clock", elapsed)
    logger.info("\n%s", profiler.output_text(show_all=False))
    logger.info("HTML call stack written to %s", destination)


if __name__ == "__main__":
    main()
