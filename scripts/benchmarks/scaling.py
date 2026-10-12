"""Times `canonicalize()` per deduper over a size sweep.

Writes one JSON results file per run under `benchmarks/results/`. Raw timings
are persisted; normalisation to a start-at-1 curve happens in the plot script
(`plot_scaling.py`), which never reruns the measurement.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

from common import DEFAULT_SEED
from common import REGISTRY
from common import SCALING_PRESETS
from common import DeduperEntry
from common import load_data
from common import now_iso
from common import results_file
from common import run_timed
from common import sample_data
from common import select_entries
from common import validate_columns
from common import write_results


logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    import pandas as pd


# HELPERS:


def run_scaling(
    entries: tuple[DeduperEntry, ...],
    sizes: tuple[int, ...],
    base_data: pd.DataFrame,
    *,
    seed: int,
) -> list[dict[str, Any]]:
    """Time `canonicalize()` for each entry over each size.

    Args:
        entries: Resolved registry entries to measure.
        sizes: Sweep sizes; the base data is sampled to each in turn.
        base_data: The loaded base dataframe.
        seed: Sampling seed, so runs are reproducible.

    Returns:
        One measurement per (deduper, size), ready for persistence.
    """

    samples = sample_data(base_data, sizes, seed=seed)
    measurements: list[dict[str, Any]] = []

    for entry in entries:
        logger.info("Scaling %s", entry.name)

        for size in sizes:
            seconds = run_timed(samples[size], entry)
            logger.info("%s: %d rows -> %.6f seconds", entry.name, size, seconds)
            measurements.append(
                {
                    "deduper": entry.name,
                    "columns": list(entry.columns),
                    "size": size,
                    "seconds": seconds,
                }
            )

    return measurements


def resolve_sizes(args: argparse.Namespace) -> tuple[int, ...]:
    """Return the size sweep: explicit flags win over the preset."""

    if args.sizes is not None:
        return tuple(args.sizes)

    return SCALING_PRESETS[args.preset].sizes


def resolve_entries(args: argparse.Namespace) -> tuple[DeduperEntry, ...]:
    """Return the deduper entries: explicit flags win over the preset."""

    if args.deduper is not None:
        return select_entries(args.deduper)

    if args.preset is not None:
        return select_entries(SCALING_PRESETS[args.preset].dedupers)

    return select_entries([entry.name for entry in REGISTRY])


def resolve_preset_name(args: argparse.Namespace) -> str:
    """Return the preset name for the results file, or "custom" on override."""

    if args.preset is not None and args.sizes is None and args.deduper is None:
        return args.preset

    return "custom"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Measure canonicalize() scaling per deduper over a size sweep.")
    parser.add_argument("--preset", choices=sorted(SCALING_PRESETS), help="Named preset of sizes and dedupers.")
    parser.add_argument("--csv", help="Load a local CSV instead of synthetic data.")
    parser.add_argument("--sizes", nargs="+", type=int, help="Sweep sizes, e.g. 1000 2000 4000. Overrides the preset.")
    parser.add_argument("--deduper", nargs="+", type=str, help="Run only these dedupers. Narrows the preset.")
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help=f"Seed for synthetic data. Default {DEFAULT_SEED}."
    )
    parser.add_argument("--output", help="JSON destination. Default is timestamped under benchmarks/results/.")

    args = parser.parse_args(argv)

    if args.preset is None and args.sizes is None:
        parser.error("Provide --preset or --sizes")

    return args


def main(argv: list[str] | None = None) -> None:
    """Run the scaling sweep and persist one JSON results file."""

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    args = parse_args(argv)
    sizes = resolve_sizes(args)
    entries = resolve_entries(args)
    preset_name = resolve_preset_name(args)

    base_data = load_data(args.csv, seed=args.seed, n_rows=max(sizes))
    validate_columns(base_data, entries)

    measurements = run_scaling(entries, sizes, base_data, seed=args.seed)

    payload = {
        "script": "scaling",
        "source": {"kind": "synthetic" if args.csv is None else "csv", "csv": args.csv},
        "created": now_iso(),
        "seed": args.seed,
        "runs": measurements,
    }

    source_name = "synthetic" if args.csv is None else Path(args.csv).stem
    destination = results_file("scaling", preset_name, source_name, output=args.output)
    write_results(Path(destination), payload)


if __name__ == "__main__":
    main()
