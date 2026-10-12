"""Shared machinery for the profiling scripts.

Bootstraps the import of `benchmarks.common` so the workload (deduper
registry, data loading, validation, pipeline build) is defined once, then
provides the base CLI arguments, workload loading, results-path helper and
git metadata the profiling scripts share.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any


# The profiling scripts' directory is on sys.path already; append its parent
# so `benchmarks.common` is importable as an implicit namespace package.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(_SCRIPTS_DIR))

from benchmarks.common import DEFAULT_SEED  # noqa: E402
from benchmarks.common import DeduperEntry  # noqa: E402
from benchmarks.common import build_run  # noqa: E402
from benchmarks.common import load_data  # noqa: E402
from benchmarks.common import now_iso  # noqa: E402
from benchmarks.common import sample_data  # noqa: E402
from benchmarks.common import select_entries  # noqa: E402
from benchmarks.common import unique_path  # noqa: E402
from benchmarks.common import validate_columns  # noqa: E402
from benchmarks.common import write_results  # noqa: E402


if TYPE_CHECKING:
    import argparse

    import pandas as pd

    from liken import Dedupe


__all__ = [
    "DEFAULT_SEED",
    "RESULTS_DIR",
    "DeduperEntry",
    "add_base_args",
    "build_run",
    "git_metadata",
    "load_data",
    "load_workload",
    "now_iso",
    "prepare",
    "results_path",
    "run",
    "run_full",
    "sample_data",
    "select_entries",
    "source_name",
    "unique_path",
    "validate_columns",
    "write_results",
]


logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"


# CLI:


def add_base_args(parser: argparse.ArgumentParser) -> None:
    """Add the base workload arguments shared by the profiling scripts.

    Args:
        parser: The script's argument parser, extended in place.
    """

    parser.add_argument("--deduper", required=True, help="Deduper to profile. One of the benchmark registry names.")
    parser.add_argument("--size", type=int, default=10_000, help="Row count of the workload. Default 10000.")
    parser.add_argument("--csv", help="Load a local CSV instead of synthetic data.")
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help=f"Seed for synthetic data. Default {DEFAULT_SEED}."
    )
    parser.add_argument(
        "--phase",
        choices=("canonicalize", "all"),
        default="canonicalize",
        help="Region under test: canonicalize() only, or build + apply + canonicalize. Default canonicalize.",
    )
    parser.add_argument("--output", help="Explicit destination. Default is timestamped under profiling/results/.")


def resolve_entry(args: argparse.Namespace) -> DeduperEntry:
    """Return the single registry entry named by `--deduper`.

    Raises:
        ValueError: If the name is not in the registry. The message lists
            the known names.
    """

    return select_entries([args.deduper])[0]


def load_workload(args: argparse.Namespace) -> tuple[pd.DataFrame, DeduperEntry]:
    """Load, validate and sample the workload data for one profiling run.

    Args:
        args: Parsed CLI arguments carrying deduper, size, csv and seed.

    Returns:
        The sampled dataframe and the resolved registry entry.
    """

    entry = resolve_entry(args)
    base = load_data(args.csv, seed=args.seed, n_rows=args.size)
    validate_columns(base, [entry])
    df = sample_data(base, [args.size], seed=args.seed)[args.size]

    return df, entry


def source_name(csv_path: str | None) -> str:
    """Return "synthetic", or the CSV file stem for the results file name."""

    return "synthetic" if csv_path is None else Path(csv_path).stem


# PHASE EXECUTION:


def prepare(df: pd.DataFrame, entry: DeduperEntry, *, phase: str) -> Dedupe | None:
    """Build the pipeline run when the build is outside the profiled region.

    Args:
        df: The sampled workload dataframe.
        entry: The registry entry to run.
        phase: "canonicalize" builds now so only `canonicalize()` is
            profiled; "all" returns None so the build is profiled too.

    Returns:
        The built `Dedupe` object, or None for the "all" phase.
    """

    return build_run(df, entry) if phase == "canonicalize" else None


def run(df: pd.DataFrame, entry: DeduperEntry, built: Dedupe | None, *, phase: str) -> Dedupe:
    """Execute the phase under test; the caller wraps this in a profiler.

    Args:
        df: The sampled workload dataframe.
        entry: The registry entry to run.
        built: The object from `prepare`, or None for the "all" phase.
        phase: "canonicalize" runs `canonicalize()` on `built`; "all"
            builds, applies and canonicalizes.

    Returns:
        The `Dedupe` object after the phase ran.

    Raises:
        ValueError: If `phase` is not a known phase, or `built` is None
            for the "canonicalize" phase.
    """

    if phase == "all":
        return run_full(df, entry)

    if phase != "canonicalize":
        raise ValueError(f"Unknown phase {phase!r}")

    if built is None:
        raise ValueError("canonicalize phase requires the object from prepare()")

    built.canonicalize()

    return built


def run_full(df: pd.DataFrame, entry: DeduperEntry) -> Dedupe:
    """Build, apply, then canonicalize — the whole pipeline run.

    Args:
        df: The sampled workload dataframe.
        entry: The registry entry to run.

    Returns:
        The built `Dedupe` object after `canonicalize()`.
    """

    lk = build_run(df, entry)
    lk.canonicalize()

    return lk


# RESULTS:


def display_size(size: int) -> str:
    """Render a row count compactly for file names: 10000 -> "10k"."""

    if size % 1_000_000 == 0 and size >= 1_000_000:
        return f"{size // 1_000_000}m"
    if size % 1_000 == 0 and size >= 1_000:
        return f"{size // 1_000}k"

    return str(size)


def results_path(script: str, deduper: str, size: int, source: str, ext: str, *, output: str | None) -> Path:
    """Timestamped results path, never overwriting an existing file.

    Pattern: `{script}_{deduper}_{size}_{source}_{timestamp}.{ext}`.

    Args:
        script: The profiling script's name, e.g. "profile".
        deduper: The deduper name.
        size: The workload row count, rendered compactly.
        source: "synthetic" or the CSV file stem.
        ext: File extension without the dot, e.g. "json" or "html".
        output: Explicit `--output` path, or None for the default.

    Returns:
        The destination path under `profiling/results/` unless `output`.
    """

    if output is not None:
        return unique_path(Path(output))

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    name = f"{script}_{deduper}_{display_size(size)}_{source}_{timestamp}.{ext}"

    return unique_path(RESULTS_DIR / name)


# GIT METADATA:


def git_metadata() -> dict[str, Any]:
    """Commit hash and dirty flag for the current checkout.

    Returns:
        `{"commit": <hash>, "dirty": <bool>}`, or both None when git is
        unavailable, so persisted profiles still load outside a repo.
    """

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=_SCRIPTS_DIR.parent,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            cwd=_SCRIPTS_DIR.parent,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}

    return {"commit": commit, "dirty": bool(status.strip())}
