"""Profile-level A/B for liken runs.

Diffs two or more profile JSONs produced by `profile.py` of the same
workload: one row per function key in the union, showing which functions
gained or lost time. Answers "did the cost move"; "did the run get faster"
belongs to the benchmark scripts.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from common import now_iso
from common import write_results


logger = logging.getLogger(__name__)


WORKLOAD_KEYS = ("deduper", "size", "seed", "source", "phase")

METRICS = ("tottime", "cumtime")


# HELPERS:


def load_profile(path: Path) -> dict[str, Any]:
    """Load one profile JSON written by `profile.py`.

    Args:
        path: Path to the JSON file.

    Returns:
        The payload dict.

    Raises:
        ValueError: If the file is unreadable, not JSON, or not a profile
            results file.
    """

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as err:
        raise ValueError(f"Cannot read profile file {path}: {err}") from err

    if not isinstance(payload, dict) or payload.get("script") != "profile":
        raise ValueError(f"{path} is not a profile results file")

    return payload


def check_workloads_match(payloads: list[tuple[Path, dict[str, Any]]]) -> dict[str, Any]:
    """Check all payloads share the same workload keys.

    Args:
        payloads: Loaded (path, payload) pairs, at least two.

    Returns:
        The shared workload dict.

    Raises:
        ValueError: Naming the first workload key that differs.
    """

    workload = payloads[0][1]["workload"]

    for path, payload in payloads[1:]:
        for key in WORKLOAD_KEYS:
            if payload["workload"].get(key) != workload.get(key):
                raise ValueError(
                    f"Workloads differ on {key!r}: "
                    f"{payloads[0][0]} has {workload.get(key)!r}, {path} has {payload['workload'].get(key)!r}. "
                    "These are different workloads, not variations of one."
                )

    return workload


def compare_rows(
    base: dict[str, Any],
    other: dict[str, Any],
    *,
    metric: str,
    limit: int,
    keep_identical: bool,
) -> list[dict[str, Any]]:
    """Build the delta rows between two profile payloads.

    Args:
        base: The baseline profile payload (side A).
        other: The comparison profile payload (side B).
        metric: "tottime" or "cumtime".
        limit: Maximum rows. 0 keeps all.
        keep_identical: Keep rows where both sides have the same time.

    Returns:
        Rows sorted by absolute delta of the metric, descending.
    """

    functions_a = {row["key"]: row for row in base["functions"]}
    functions_b = {row["key"]: row for row in other["functions"]}

    rows: list[dict[str, Any]] = []
    for key in dict.fromkeys([*functions_a, *functions_b]):
        row_a = functions_a.get(key)
        row_b = functions_b.get(key)
        time_a = row_a[metric] if row_a is not None else None
        time_b = row_b[metric] if row_b is not None else None

        if time_a is not None and time_b is not None:
            delta = time_b - time_a
            delta_pct = (delta / time_a * 100) if time_a > 0 else None
        else:
            delta = None
            delta_pct = None

        if not keep_identical and time_a == time_b:
            continue

        rows.append(
            {
                "key": key,
                f"{metric}_a": time_a,
                f"{metric}_b": time_b,
                "delta": delta,
                "delta_pct": delta_pct,
            }
        )

    def order(row: dict[str, Any]) -> float:
        """Sort key: absolute delta, new/removed rows last."""

        value = row["delta"]
        return abs(value) if value is not None else float("-inf")

    rows.sort(key=order, reverse=True)

    return rows if limit == 0 else rows[:limit]


def render_table(rows: list[dict[str, Any]], *, metric: str) -> str:
    """Render the delta rows as a fixed-width console table.

    Args:
        rows: Delta rows from `compare_rows`.
        metric: The compared metric, for the column headers.

    Returns:
        The table as text.
    """

    def cell(value: float | None, fmt: str) -> str:
        return "-" if value is None else fmt.format(value)

    headers = (f"{metric} A", f"{metric} B", "delta", "delta %", "function")
    body = [
        (
            cell(row[f"{metric}_a"], "{:.6f}"),
            cell(row[f"{metric}_b"], "{:.6f}"),
            cell(row["delta"], "{:+.6f}"),
            cell(row["delta_pct"], "{:+.1f}%"),
            row["key"],
        )
        for row in rows
    ]

    widths = [max(len(headers[i]), *(len(line[i]) for line in body)) if body else len(headers[i]) for i in range(5)]

    lines = ["  ".join(header.ljust(widths[i]) for i, header in enumerate(headers))]
    lines.extend("  ".join(field.ljust(widths[i]) for i, field in enumerate(line)) for line in body)

    return "\n".join(lines)


# MAIN:


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Diff two or more profile JSONs of the same workload.")
    parser.add_argument("--input", nargs="+", required=True, help="Two or more profile JSON files from profile.py.")
    parser.add_argument("--sort", choices=METRICS, default="tottime", help="Compared metric. Default tottime.")
    parser.add_argument("--limit", type=int, default=30, help="Rows in the table. 0 keeps all. Default 30.")
    parser.add_argument("--all", action="store_true", help="Keep rows whose time did not change.")
    parser.add_argument("--output", help="Write the delta table as JSON to this path.")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Diff the profile JSONs and print (or persist) the delta table."""

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    args = parse_args(argv)

    try:
        payloads = [(Path(path), load_profile(Path(path))) for path in args.input]
    except ValueError as err:
        raise SystemExit(f"error: {err}") from err

    if len(payloads) < 2:
        raise SystemExit("error: --input needs two or more profile JSON files")

    try:
        workload = check_workloads_match(payloads)
    except ValueError as err:
        raise SystemExit(f"error: {err}") from err

    # One table compares two sides. With more than two inputs, compare the
    # oldest against the newest and say which runs were left out.
    payloads.sort(key=lambda item: str(item[1].get("created", "")))
    base_path, base = payloads[0]
    other_path, other = payloads[-1]
    if len(payloads) > 2:
        middle = ", ".join(str(path) for path, _ in payloads[1:-1])
        logger.warning("Comparing oldest against newest; not compared: %s", middle)

    rows = compare_rows(base, other, metric=args.sort, limit=args.limit, keep_identical=args.all)

    header = (
        f"Workload: {workload['deduper']}, size {workload['size']}, seed {workload['seed']}, "
        f"source {workload['source']}, phase {workload['phase']}\n"
        f"A: {base_path} (git {base['git'].get('commit')}, dirty={base['git'].get('dirty')})\n"
        f"B: {other_path} (git {other['git'].get('commit')}, dirty={other['git'].get('dirty')})"
    )
    logger.info("%s\n\n%s", header, render_table(rows, metric=args.sort))

    if args.output is not None:
        payload = {
            "script": "compare",
            "created": now_iso(),
            "workload": workload,
            "metric": args.sort,
            "git": {"a": base.get("git"), "b": other.get("git")},
            "files": {"a": str(base_path), "b": str(other_path)},
            "rows": rows,
        }
        written = write_results(Path(args.output), payload)
        logger.info("Delta table written to %s", written)


if __name__ == "__main__":
    main()
