"""Renders a grouped bar chart PNG from gathered benchmark results.

Gathering never reruns the benchmark: input is a results file, a directory,
or a glob, optionally narrowed with `--deduper` and `--after`. Overlapping
(deduper, columns, size) keys resolve to the newest run with a warning.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

import matplotlib.pyplot as plt
from common import REGISTRY
from common import default_png_path
from common import gather_files
from common import merge_payloads
from common import parse_after
from common import unique_path
from matplotlib.ticker import FuncFormatter


if TYPE_CHECKING:
    from collections.abc import Sequence


logger = logging.getLogger(__name__)


# HELPERS:


def ordered_labels(runs: Sequence[dict[str, Any]]) -> list[str]:
    """Deduper labels in registry order, then any unknown names."""

    present = {str(run["deduper"]) for run in runs}

    known = [entry.name for entry in REGISTRY if entry.name in present]
    unknown = [name for name in dict.fromkeys(str(run["deduper"]) for run in runs) if name not in known]

    return known + unknown


def run_chart(runs: Sequence[dict[str, Any]], output: Path) -> None:
    """Draw the grouped horizontal bar chart and save it.

    Args:
        runs: Merged measurements, one per (deduper, columns, size).
        output: PNG destination.
    """

    labels = ordered_labels(runs)
    sizes = sorted({int(run["size"]) for run in runs})

    seconds: dict[tuple[str, int], float] = {}
    for run in runs:
        seconds[(str(run["deduper"]), int(run["size"]))] = float(run["seconds"])

    y = range(len(labels))
    height = 0.8 / len(sizes)
    cmap = plt.get_cmap("viridis")

    plt.figure(figsize=(10, 6))

    for i, size in enumerate(sizes):
        offset = (i - len(sizes) / 2) * height + height / 2
        plt.barh(
            [position + offset for position in y],
            [seconds.get((label, size), float("nan")) for label in labels],
            height=height,
            label=str(size),
            color=cmap(i / max(len(sizes) - 1, 1)),
        )

    plt.gca().invert_yaxis()
    plt.xscale("log")

    plt.axvline(60, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    plt.axvline(3600, color="red", linestyle="--", linewidth=1, alpha=0.7)
    plt.text(40, -1.2, "1 min", color="gray")
    plt.text(2000, -1.2, "1 hour", color="red")

    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))

    plt.yticks(list(y), labels)
    plt.ylabel("Deduper", fontsize=12)
    plt.xlabel("Execution Time (seconds)", fontsize=12)
    plt.legend(title="Dataset Size", loc="lower right")

    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()

    logger.info("Benchmark chart saved to %s", output)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Plot benchmark results as a grouped bar chart.")
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
        paths, script="benchmark", dedupers=args.deduper, after=parse_after(args.after) if args.after else None
    )

    if not runs:
        raise ValueError("No benchmark measurements to plot after filters")

    output = unique_path(Path(args.output)) if args.output else default_png_path("plot_benchmark")
    run_chart(runs, output)


if __name__ == "__main__":
    main()
