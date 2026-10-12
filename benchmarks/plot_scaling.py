"""Renders normalised scaling curves from gathered scaling results.

Gathering never reruns the benchmark: input is a results file, a directory,
or a glob, optionally narrowed with `--deduper` and `--after`. Each deduper's
curve is normalised so its earliest measured size maps to 1. Raw timings were
persisted by `scaling.py`; normalisation happens here.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

import matplotlib.pyplot as plt
from common import default_png_path
from common import gather_files
from common import merge_payloads
from common import parse_after
from common import unique_path


if TYPE_CHECKING:
    from collections.abc import Sequence


logger = logging.getLogger(__name__)


# HELPERS:


def curve_for(runs: Sequence[dict[str, Any]], name: str) -> tuple[list[int], list[float]]:
    """Extract one deduper's (sizes, normalised times) from merged runs.

    Args:
        runs: Merged measurements.
        name: The deduper label.

    Returns:
        Sizes ascending, and times divided by the earliest size's time.
    """

    points = sorted((int(run["size"]), float(run["seconds"])) for run in runs if str(run["deduper"]) == name)

    sizes = [size for size, _ in points]
    times = [seconds for _, seconds in points]

    if times and times[0] > 0:
        times = [seconds / times[0] for seconds in times]

    return sizes, times


def run_chart(runs: Sequence[dict[str, Any]], output: Path) -> None:
    """Draw the normalised scaling curves and save it.

    Args:
        runs: Merged measurements, one per (deduper, columns, size).
        output: PNG destination.
    """

    names = list(dict.fromkeys(str(run["deduper"]) for run in runs))
    start = min(int(run["size"]) for run in runs)

    plt.figure(figsize=(10, 6))
    cmap = plt.get_cmap("tab20")

    for i, name in enumerate(names):
        sizes, times = curve_for(runs, name)
        plt.plot(sizes, times, linewidth=2, label=name, color=cmap(i / max(len(names) - 1, 1)))

    plt.xticks([])
    plt.yticks([])
    plt.xlim(left=start)
    plt.ylim(bottom=0)

    plt.xlabel("Dataset Size", fontsize=12)
    plt.ylabel("Execution Time", fontsize=12)

    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()

    logger.info("Scaling chart saved to %s", output)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Plot scaling results as normalised curves.")
    parser.add_argument("--input", required=True, help="Results file, directory, or glob to gather.")
    parser.add_argument("--deduper", nargs="+", help="Keep only these dedupers.")
    parser.add_argument("--after", help="Keep only results created after this timestamp.")
    parser.add_argument("--output", help="PNG destination. Default is timestamped under benchmarks/results/.")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Gather results files, merge them, and plot one PNG."""

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    args = parse_args(argv)

    paths = gather_files(args.input)
    runs = merge_payloads(
        paths, script="scaling", dedupers=args.deduper, after=parse_after(args.after) if args.after else None
    )

    if not runs:
        raise ValueError("No scaling measurements to plot after filters")

    output = unique_path(Path(args.output)) if args.output else default_png_path("plot_scaling")
    run_chart(runs, output)


if __name__ == "__main__":
    main()
