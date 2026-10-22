"""Shared machinery for the benchmark and scaling scripts.

Holds the deduper registry, named presets, data loading, the timed
`canonicalize()` runner, and JSON persistence.
"""

from __future__ import annotations

import glob
import json
import logging
import random
import time
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

import pandas as pd
from faker import Faker

from liken import Dedupe
from liken.collections.pipelines import Pipeline
from liken.collections.pipelines import col


if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence


logger = logging.getLogger(__name__)


# HELPERS:


def _linspace(start: int, end: int, steps: int) -> tuple[int, ...]:
    """Evenly spaced integer sizes from `start` to `end`, both included."""

    if steps < 2:
        raise ValueError(f"steps must be at least 2, got {steps}")

    step_size = (end - start) / (steps - 1)
    return tuple(round(start + i * step_size) for i in range(steps))


# CONSTANTS:


RESULTS_DIR = Path(__file__).parent / "results"

DEFAULT_SEED = 1234

SYNTHETIC_SCHEMA = [
    "id",
    "first_name",
    "last_name",
    "email",
    "phone",
    "address",
    "city",
    "country",
    "company",
    "job",
    "date_of_birth",
    "signup_date",
    "salary",
    "is_active",
    # Extra columns so the full deduper registry works without a CSV.
    "probability",
    "house_number",
    "height",
    "continent",
]

CONTINENTS = (
    "Africa",
    "Antarctica",
    "Asia",
    "Europe",
    "North America",
    "Oceania",
    "South America",
)


# TYPES:


@dataclass(frozen=True)
class DeduperEntry:
    """One benchmarkable deduper configuration.

    Attributes:
        name: Short label used in CLI, results files and charts.
        columns: Columns the deduper reads; validated against the data.
        method: Chainable method name on `Col` that builds the deduper.
        args: Positional arguments for that method.
        kwargs: Keyword arguments for that method.
    """

    name: str
    columns: tuple[str, ...]
    method: str
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Preset:
    """A named preset of dataset sizes and dedupers."""

    sizes: tuple[int, ...]
    dedupers: tuple[str, ...]


# REGISTRY (data, not code paths):


REGISTRY: tuple[DeduperEntry, ...] = (
    DeduperEntry("exact", ("address",), "exact"),
    DeduperEntry("fuzzy", ("address",), "fuzzy"),
    DeduperEntry("fuzzy_topn", ("address",), "fuzzy", kwargs={"threshold": 0.95, "topn": 5}),
    DeduperEntry(
        "sorted_neighbourhood",
        ("address",),
        "sorted_neighbourhood",
        kwargs={"threshold": 0.95, "window": 5},
    ),
    DeduperEntry("tfidf", ("address",), "tfidf"),
    DeduperEntry("lsh", ("address",), "lsh"),
    DeduperEntry("jaccard", ("city", "country", "job"), "jaccard"),
    DeduperEntry("cosine", ("probability", "house_number", "height"), "cosine"),
    DeduperEntry(
        "knn",
        ("probability", "house_number", "height"),
        "knn",
        kwargs={"threshold": 0.95, "k": 5, "metric": "cosine"},
    ),
    DeduperEntry("isna", ("address",), "isna"),
    DeduperEntry("isin", ("country",), "isin", args=(["Germany"],)),
    DeduperEntry("str_contains", ("address",), "str_contains", args=("street",)),
    DeduperEntry("str_startswith", ("address",), "str_startswith", args=("5",)),
    DeduperEntry("str_endswith", ("address",), "str_endswith", args=("5",)),
    DeduperEntry("str_len", ("address",), "str_len", kwargs={"min_len": 2, "max_len": 25}),
)

FAST_DEDUPERS = ("exact", "isna", "isin", "str_contains", "str_len")

BENCHMARK_PRESETS: dict[str, Preset] = {
    "quick": Preset(sizes=(10_000, 50_000), dedupers=FAST_DEDUPERS),
    "full": Preset(
        sizes=(10_000, 50_000, 100_000, 250_000, 1_000_000),
        dedupers=tuple(entry.name for entry in REGISTRY),
    ),
}

SCALING_PRESETS: dict[str, Preset] = {
    "quick": Preset(sizes=_linspace(1_000, 5_000, 6), dedupers=FAST_DEDUPERS),
    "full": Preset(sizes=_linspace(1_000, 12_000, 12), dedupers=tuple(entry.name for entry in REGISTRY)),
}


# HELPERS:


def now_iso() -> str:
    """Current local time, second resolution, ISO 8601."""

    return datetime.now().isoformat(timespec="seconds")


def parse_size(token: str) -> int:
    """Parse a size token like "10K", "1M" or "100000" into a row count."""

    text = token.strip().upper()

    if text.endswith("K"):
        return int(text[:-1]) * 1_000
    if text.endswith("M"):
        return int(text[:-1]) * 1_000_000

    try:
        return int(text)
    except ValueError as err:
        raise ValueError(f"Cannot parse size {token!r}") from err


def registry_entry(name: str) -> DeduperEntry:
    """Return the registry entry for `name`, failing loudly if absent."""

    for entry in REGISTRY:
        if entry.name == name:
            return entry

    known = ", ".join(entry.name for entry in REGISTRY)
    raise ValueError(f"Unknown deduper {name!r}. Known dedupers: {known}")


def select_entries(names: Sequence[str]) -> tuple[DeduperEntry, ...]:
    """Resolve deduper names to registry entries, preserving order."""

    return tuple(registry_entry(name) for name in names)


# DATA LOADING:


def _maybe_null(value: Any, p: float) -> Any:
    """Return `value`, or None with probability `p`, matching `fake_row()`."""

    return value if random.random() > p else None


def _synthetic_rows(n_rows: int, *, seed: int) -> list[tuple[Any, ...]]:
    """Generate `n_rows` synthetic rows covering `SYNTHETIC_SCHEMA`.

    Mirrors `liken.datasets.fake_row()`: the nullable fields carry 2% nulls
    so the `isna` deduper exercises null handling on the default source.
    """

    random.seed(seed)
    fake = Faker()
    fake.seed_instance(seed)

    rows: list[tuple[Any, ...]] = []

    for _ in range(n_rows):
        row = (
            fake.uuid4(),
            _maybe_null(fake.first_name(), 0.02),
            _maybe_null(fake.last_name(), 0.02),
            _maybe_null(fake.email(), 0.02),
            _maybe_null(fake.phone_number().replace("x", ", "), 0.02),
            _maybe_null(fake.address().replace("\n", ", "), 0.02),
            _maybe_null(fake.city(), 0.02),
            _maybe_null(fake.country(), 0.02),
            _maybe_null(fake.company(), 0.02),
            _maybe_null(fake.job(), 0.02),
            _maybe_null(fake.date_of_birth(minimum_age=18, maximum_age=80), 0.02),
            _maybe_null(fake.date_time_this_decade(), 0.02),
            _maybe_null(round(random.uniform(30_000, 150_000), 2), 0.02),
            random.choice([True, False]),
            round(random.random(), 6),
            random.randint(1, 999),
            round(random.uniform(1.5, 2.1), 2),
            random.choice(CONTINENTS),
        )
        rows.append(row)

    return rows


def load_data(
    csv_path: str | None,
    *,
    seed: int,
    n_rows: int,
) -> pd.DataFrame:
    """Load the base data: a local CSV, or synthetic data at `n_rows`.

    Args:
        csv_path: Path to a local CSV, or None for synthetic data.
        seed: Seed for synthetic generation.
        n_rows: Row count of the synthetic base. Ignored for CSV input.

    Returns:
        The base dataframe, ready to be sampled to the requested sizes.
    """

    if csv_path is None:
        logger.info("Synthesizing %d rows (seed=%d)...", n_rows, seed)
        return pd.DataFrame(_synthetic_rows(n_rows, seed=seed), columns=SYNTHETIC_SCHEMA)

    return pd.read_csv(csv_path)


def sample_data(
    base: pd.DataFrame,
    sizes: Sequence[int],
    *,
    seed: int,
) -> dict[int, pd.DataFrame]:
    """Sample the base data to each requested size.

    Args:
        base: The loaded base dataframe.
        sizes: Requested row counts. Must not exceed the base size.
        seed: Sampling seed, so runs are reproducible.

    Returns:
        Mapping of size to the sampled dataframe.
    """

    base_rows = len(base)

    if base_rows < max(sizes):
        raise ValueError(f"Data has {base_rows} rows; largest requested size is {max(sizes)}")

    return {size: base.sample(n=size, random_state=seed) for size in sizes}


def validate_columns(df: pd.DataFrame, entries: Sequence[DeduperEntry]) -> None:
    """Check every entry's columns exist in `df`, failing loudly if not.

    Raises:
        ValueError: If any entry requires columns missing from the data.
    """

    missing: dict[str, list[str]] = {}

    for entry in entries:
        absent = [column for column in entry.columns if column not in df.columns]
        if absent:
            missing[entry.name] = absent

    if missing:
        details = "; ".join(f"{name}: {', '.join(cols)}" for name, cols in missing.items())
        raise ValueError(f"Data is missing columns required by dedupers - {details}")


# TIMED RUNNER:


def build_run(df: pd.DataFrame, entry: DeduperEntry) -> Dedupe:
    """Build one pipeline run: the `Dedupe` object with the entry applied.

    Shared by `run_timed` and the profiling scripts, so the build is
    reusable around a profiler.

    Args:
        df: The dataframe to deduplicate.
        entry: The registry entry to run.

    Returns:
        The `Dedupe` object with the pipeline applied, ready to canonicalize.
    """

    lk = Dedupe(df)

    # Single-column dedupers validate against a plain string, not a 1-tuple.
    columns: str | tuple[str, ...] = entry.columns[0] if len(entry.columns) == 1 else entry.columns

    lk.apply(Pipeline().step(getattr(col(columns), entry.method)(*entry.args, **entry.kwargs)))

    return lk


def run_timed(df: pd.DataFrame, entry: DeduperEntry) -> float:
    """Build and run one pipeline, timing `canonicalize()` only.

    Matches the old scripts' semantics: `apply()` runs first, then only
    `canonicalize()` is timed.

    Args:
        df: The dataframe to deduplicate.
        entry: The registry entry to run.

    Returns:
        Elapsed seconds for `canonicalize()`.
    """

    lk = build_run(df, entry)

    start = time.perf_counter()
    lk.canonicalize()

    return time.perf_counter() - start


# JSON PERSISTENCE:


def unique_path(path: Path) -> Path:
    """Return `path`, suffixed `_1`, `_2`, ... if it already exists.

    Runs never overwrite an existing results file.
    """

    if not path.exists():
        return path

    stem, suffix = path.stem, path.suffix

    counter = 1
    while True:
        candidate = path.with_name(f"{stem}_{counter}{suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def write_results(path: Path, payload: Mapping[str, Any]) -> Path:
    """Write a results payload as JSON, never overwriting.

    Args:
        path: Desired destination; a suffix is appended if taken.
        payload: Results payload, serialized as JSON.

    Returns:
        The path actually written.
    """

    path = unique_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    logger.info("Results written to %s", path)

    return path


def results_file(
    script: str,
    preset_name: str,
    source_name: str,
    *,
    output: str | None,
) -> Path:
    """Timestamped results path: `{script}_{preset}_{source}_{ts}.json`.

    Args:
        script: "benchmark" or "scaling".
        preset_name: Preset name, or "custom" when flags overrode a preset.
        source_name: "synthetic" or the CSV file stem.
        output: Explicit `--output` path, or None for the default.

    Returns:
        The destination path, defaulting under `benchmarks/results/`.
    """

    if output is not None:
        return Path(output)

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    return RESULTS_DIR / f"{script}_{preset_name}_{source_name}_{timestamp}.json"


def default_png_path(name: str) -> Path:
    """Timestamped PNG path under `benchmarks/results/`, never overwriting."""

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    return unique_path(RESULTS_DIR / f"{name}_{timestamp}.png")


# GATHERING AND MERGING (plot scripts):


def gather_files(input_spec: str) -> list[Path]:
    """Resolve `--input` (file, directory, or glob) to results files.

    Args:
        input_spec: A JSON results file path, a directory (all `*.json`
            inside), or a glob pattern such as "benchmarks/results/scaling_*.json".

    Returns:
        The matching files, sorted by name.

    Raises:
        ValueError: If nothing matches. Lists what was searched.
    """

    if "*" in input_spec or "?" in input_spec:
        found = [Path(p) for p in glob.glob(input_spec)]
        searched = f"glob {input_spec!r}"
    else:
        path = Path(input_spec)
        if path.is_dir():
            found = sorted(path.glob("*.json"))
            searched = f"directory {path}"
        elif path.is_file():
            found = [path]
            searched = f"file {path}"
        else:
            raise ValueError(f"No results file at {path}")

    if not found:
        raise ValueError(f"No results files matched {searched}")

    return sorted(set(found))


def parse_after(token: str) -> datetime:
    """Parse an `--after` timestamp (ISO 8601, or the compact file format)."""

    text = token.strip()

    if len(text) == 15 and text[8] == "T" and text.replace("T", "").isdigit():
        text = f"{text[:4]}-{text[4:6]}-{text[6:8]}T{text[9:11]}:{text[11:13]}:{text[13:15]}"

    try:
        return datetime.fromisoformat(text)
    except ValueError as err:
        raise ValueError(f"Cannot parse --after timestamp {token!r}") from err


def load_payload(path: Path, *, script: str) -> dict[str, Any]:
    """Load one results file for a plot script.

    Args:
        path: Path to a JSON results file.
        script: Expected "script" value ("benchmark" or "scaling"); files
            from the other script are skipped with a warning.

    Returns:
        The payload dict, or an empty dict when the file was skipped.
    """

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as err:
        logger.warning("Skipping unreadable results file %s: %s", path, err)
        return {}

    if not isinstance(payload, dict) or payload.get("script") != script:
        logger.warning("Skipping %s: not a %s results file", path, script)
        return {}

    return payload


def merge_payloads(
    paths: Sequence[Path],
    *,
    script: str,
    dedupers: Sequence[str] | None = None,
    after: datetime | None = None,
) -> list[dict[str, Any]]:
    """Merge gathered results files into one measurement set.

    Files are processed oldest first. Measurements are keyed by
    (deduper, columns, size); when several files contain the same key the
    newest run wins and the superseded file names are logged. Files that
    differ in seed or data source trigger a warning that the chart mixes
    origins.

    Args:
        paths: Gathered results files.
        script: Expected "script" value; other files are skipped.
        dedupers: Keep only these dedupers, or None for all.
        after: Keep only files created after this timestamp, or None.

    Returns:
        One measurement dict per (deduper, columns, size), in first-seen
        key order across the gathered files.
    """

    payloads: list[tuple[Path, dict[str, Any]]] = []

    for path in paths:
        payload = load_payload(path, script=script)
        if not payload:
            continue
        if after is not None and not _created_after(payload, path, after):
            continue
        payloads.append((path, payload))

    if not payloads:
        raise ValueError(f"No usable {script} results files among {', '.join(str(p) for p in paths)}")

    origins = {_origin(payload) for _, payload in payloads}
    if len(origins) > 1:
        logger.warning("Gathered files mix data origins (source/seed): %s", sorted(origins))

    payloads.sort(key=lambda item: (str(item[1].get("created", "")), item[0].name))

    merged: dict[tuple[str, tuple[str, ...], int], tuple[str, dict[str, Any]]] = {}
    superseded: dict[tuple[str, tuple[str, ...], int], set[str]] = {}

    for path, payload in payloads:
        runs = payload.get("runs")
        if not isinstance(runs, list):
            logger.warning("Skipping %s: no runs recorded", path)
            continue

        for run in runs:
            key = _run_key(run, path)
            if key is None:
                continue
            name = key[0]
            if dedupers is not None and name not in dedupers:
                continue
            if key in merged:
                superseded.setdefault(key, set()).add(merged[key][0])
            merged[key] = (path.name, dict(run))

    if superseded:
        for key, files in sorted(superseded.items()):
            logger.warning(
                "Key %s appears in multiple gathered files; newest run wins, superseded: %s", key, sorted(files)
            )

    return [run for _, run in merged.values()]


def _created_after(payload: dict[str, Any], path: Path, after: datetime) -> bool:
    """Check a payload's `created` timestamp against the `--after` cutoff.

    Files with a missing or unparsable timestamp are skipped with a warning
    when an `--after` filter is active, instead of failing the merge.
    """

    raw = payload.get("created")

    if isinstance(raw, str) and raw:
        try:
            return parse_after(raw) > after
        except ValueError:
            pass

    logger.warning("Skipping %s: unparsable created timestamp %r (--after is active)", path, raw)
    return False


def _origin(payload: dict[str, Any]) -> tuple[str, str, Any]:
    """Extract the (kind, csv, seed) origin triple for the mix warning."""

    source = payload.get("source")

    if not isinstance(source, dict):
        return ("", "", payload.get("seed"))

    return (str(source.get("kind") or ""), str(source.get("csv")), payload.get("seed"))


def _run_key(run: Any, path: Path) -> tuple[str, tuple[str, ...], int] | None:
    """Build the (deduper, columns, size) merge key from one run record.

    Returns None (with a warning naming the file) for records that are not
    usable, so one malformed run does not abort the whole merge.
    """

    if not isinstance(run, dict):
        logger.warning("Skipping malformed run in %s: %r", path, run)
        return None

    name = str(run.get("deduper"))

    try:
        size = int(run["size"])
    except (KeyError, TypeError, ValueError):
        logger.warning("Skipping run %r in %s: unusable size %r", name, path, run.get("size"))
        return None

    columns = run.get("columns")
    if not isinstance(columns, (list, tuple)) or not columns:
        logger.warning("Skipping run %r in %s: unusable columns %r", name, path, columns)
        return None

    return (name, tuple(str(c) for c in columns), size)
