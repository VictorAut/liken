"""cProfile/pstats first look at one liken run.

Profiles the workload phase under `cProfile`, prints a ranked function
report, and persists one JSON summary (plus, with `--dump`, a raw `.pstats`
file for interactive analysis with `python -m pstats`).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


# This script shares its name with the stdlib `profile` module, which
# cProfile imports. Drop this script's directory from sys.path so the
# stdlib wins, then restore it for the flat `common` import.
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if Path(p).resolve() != Path(_SCRIPT_DIR)]

import argparse  # noqa: E402
import cProfile  # noqa: E402
import importlib.metadata  # noqa: E402
import logging  # noqa: E402
import platform  # noqa: E402
import pstats  # noqa: E402
import time  # noqa: E402


sys.path.insert(0, _SCRIPT_DIR)

from common import add_base_args  # noqa: E402
from common import git_metadata  # noqa: E402
from common import load_workload  # noqa: E402
from common import now_iso  # noqa: E402
from common import prepare  # noqa: E402
from common import results_path  # noqa: E402
from common import run  # noqa: E402
from common import source_name  # noqa: E402
from common import write_results  # noqa: E402


logger = logging.getLogger(__name__)


# HELPERS:


def function_rows(profiler: cProfile.Profile, sort: str, limit: int) -> list[dict[str, Any]]:
    """Extract the profile's function rows, sorted and limited.

    Args:
        profiler: The finished `cProfile.Profile`.
        sort: Metric to rank by: tottime, cumtime, calls or name.
        limit: Maximum rows. 0 keeps all rows.

    Returns:
        One row per function, keyed `path:line(funcname)`.
    """

    stats = pstats.Stats(profiler).stats  # type: ignore[attr-defined]  # runtime attribute, absent from typeshed

    rows: list[dict[str, Any]] = []
    for (filename, lineno, funcname), (primitive, calls, tottime, cumtime, _callers) in stats.items():
        rows.append(
            {
                "key": f"{filename}:{lineno}({funcname})",
                "primitive_calls": primitive,
                "calls": calls,
                "tottime": tottime,
                "cumtime": cumtime,
            }
        )

    reverse = sort != "name"
    metric = "key" if sort == "name" else sort
    rows.sort(key=lambda row: row[metric], reverse=reverse)

    return rows if limit == 0 else rows[:limit]


def print_report(profiler: cProfile.Profile, sort: str, limit: int) -> None:
    """Print the pstats console report, full paths, ranked by `sort`."""

    stats = pstats.Stats(profiler)
    stats.sort_stats(sort)
    if limit == 0:
        stats.print_stats()
    else:
        stats.print_stats(limit)


def package_version(name: str) -> str | None:
    """Return the installed version of `name`, or None when absent."""

    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


# MAIN:


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="cProfile first look at one liken run.")
    add_base_args(parser)
    parser.add_argument(
        "--sort",
        choices=("tottime", "cumtime", "calls", "name"),
        default="tottime",
        help="Ranking metric. Default tottime (self time).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=40,
        help="Rows in the console report and the JSON. 0 keeps all rows. Default 40.",
    )
    parser.add_argument("--dump", action="store_true", help="Also write a raw .pstats file next to the JSON.")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Profile one workload run and persist the report."""

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    args = parse_args(argv)

    try:
        df, entry = load_workload(args)
    except ValueError as err:
        raise SystemExit(f"error: {err}") from err

    built = prepare(df, entry, phase=args.phase)

    profiler = cProfile.Profile()
    start = time.perf_counter()
    profiler.enable()
    run(df, entry, built, phase=args.phase)
    profiler.disable()
    elapsed = time.perf_counter() - start

    print_report(profiler, args.sort, args.limit)

    source = source_name(args.csv)
    destination = results_path("profile", entry.name, args.size, source, "json", output=args.output)

    payload: dict[str, Any] = {
        "script": "profile",
        "created": now_iso(),
        "workload": {
            "deduper": entry.name,
            "size": args.size,
            "seed": args.seed,
            "source": source,
            "phase": args.phase,
        },
        "git": git_metadata(),
        "versions": {
            "python": platform.python_version(),
            "liken": package_version("liken"),
        },
        "platform": platform.platform(),
        "sort": args.sort,
        "elapsed_seconds": elapsed,
        "functions": function_rows(profiler, args.sort, args.limit),
    }

    written = write_results(destination, payload)
    logger.info("Profile JSON written to %s", written)

    if args.dump:
        dump_path = results_path("profile", entry.name, args.size, source, "pstats", output=None)
        profiler.dump_stats(str(dump_path))
        logger.info("pstats dump written to %s", dump_path)


if __name__ == "__main__":
    main()
