"""line_profiler drill-down of named functions in one liken workload.

Instruments the functions named with `--function`, runs the workload phase,
prints the per-line report, and saves it to `results/`. Line profiling is
the third step of the workflow: use it after `profile.py` or `instrument.py`
has named a hot function.
"""

from __future__ import annotations

import argparse
import importlib
import io
import logging
import time
from typing import Any

from common import add_base_args
from common import load_workload
from common import prepare
from common import results_path
from common import run
from common import source_name


try:
    from line_profiler import LineProfiler
except ImportError as err:
    raise SystemExit(
        "error: line_profiler is not installed; install the dev group on a supported Python (python_full_version < '3.15')"
    ) from err


logger = logging.getLogger(__name__)


# HELPERS:


def resolve_function(dotted: str) -> Any:
    """Import the module part of a dotted path and walk its attributes.

    Args:
        dotted: A path like `liken.core.deduper.Dedupe._deduplicate`.

    Returns:
        The resolved object.

    Raises:
        ValueError: With the reason the path did not resolve.
    """

    parts = dotted.split(".")

    if len(parts) < 2:
        raise ValueError(f"--function {dotted!r}: need at least a module and an attribute")

    for split in range(len(parts) - 1, 0, -1):
        module_name = ".".join(parts[:split])
        try:
            obj: Any = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue

        for attr in parts[split:]:
            try:
                obj = getattr(obj, attr)
            except AttributeError as err:
                raise ValueError(f"--function {dotted!r}: {module_name} has no attribute {attr!r}") from err

        return obj

    raise ValueError(f"--function {dotted!r}: no module in the path could be imported")


# MAIN:


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="line_profiler drill-down of named functions in one liken workload.")
    add_base_args(parser)
    parser.add_argument(
        "--function",
        action="append",
        required=True,
        help="Dotted path of a function to instrument, e.g. liken.core.deduper.Dedupe._deduplicate. Repeatable.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Line-profile the named functions over one workload run."""

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    args = parse_args(argv)

    try:
        targets = [resolve_function(dotted) for dotted in args.function]
        df, entry = load_workload(args)
    except ValueError as err:
        raise SystemExit(f"error: {err}") from err

    profiler = LineProfiler()
    for target in targets:
        profiler.add_function(target)

    built = prepare(df, entry, phase=args.phase)

    start = time.perf_counter()
    profiler.enable_by_count()
    run(df, entry, built, phase=args.phase)
    profiler.disable_by_count()
    elapsed = time.perf_counter() - start

    report = io.StringIO()
    profiler.print_stats(stream=report)
    text = report.getvalue()

    logger.info("line_profiler, %.3f seconds wall clock\n%s", elapsed, text)

    destination = results_path("lineprofile", entry.name, args.size, source_name(args.csv), "txt", output=args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")
    logger.info("Report written to %s", destination)


if __name__ == "__main__":
    main()
