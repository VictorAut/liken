"""Fail when the floors tox.toml tests and pyproject.toml declares differ.

Run as a precommit hook.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path


# Core dependencies (floors in [project].dependencies) and extras
# (floors in [project.optional-dependencies], possibly declared in more
# than one extra).
PYPROJECT_FLOOR_SOURCES: dict[str, list[tuple[str, str]]] = {
    "pandas": [("project", "dependencies")],
    "polars": [("project", "dependencies")],
    "pyarrow": [("project", "dependencies")],
    "pyspark": [("extras", "pyspark"), ("extras", "all")],
    "modin": [("extras", "modin"), ("extras", "all")],
    "ray": [("extras", "ray"), ("extras", "all")],
    "dask": [("extras", "dask"), ("extras", "all")],
}


def parse_declared_floors(pyproject: dict) -> dict[str, set[str]]:
    """Extract each backend's declared floor version from pyproject data.

    Args:
        pyproject: parsed contents of pyproject.toml.

    Returns:
        Mapping of backend name to the set of floor versions declared
        for it (more than one version means inconsistent declarations).
    """
    floors: dict[str, set[str]] = {}
    for backend, sources in PYPROJECT_FLOOR_SOURCES.items():
        floors[backend] = set()
        for kind, section in sources:
            if kind == "project":
                requirements = pyproject["project"].get("dependencies", [])
            else:
                requirements = pyproject["project"].get("optional-dependencies", {}).get(section, [])
            for requirement in requirements:
                name = requirement.split("[")[0].split(">=")[0].strip()
                if name != backend:
                    continue
                for fragment in requirement.split(">=")[1:]:
                    version = fragment.split(",")[0].split("<")[0].split("!")[0].strip()
                    if version:
                        floors[backend].add(version)
    return floors


def parse_tested_floors(tox: dict) -> dict[str, set[str]]:
    """Extract each backend's pinned floor version from tox env data.

    Args:
        tox: parsed contents of tox.toml.

    Returns:
        Mapping of backend name to the set of `==` pins for it found in
        the `<backend>-floor` env's install commands.
    """
    floors: dict[str, set[str]] = {}
    for backend in PYPROJECT_FLOOR_SOURCES:
        floors[backend] = set()
        env = tox.get("env", {}).get(f"{backend}-floor")
        if env is None:
            continue
        for command in env.get("commands_pre", []):
            if "--no-deps" in command:
                continue
            for requirement in command:
                if not requirement.startswith(f"{backend}"):
                    continue
                if "==" not in requirement:
                    continue
                version = requirement.split("==")[1].split(",")[0].strip()
                floors[backend].add(version)
    return floors


def main() -> int:
    """Check the repo's floor claims; return the hook exit code."""
    root = Path(__file__).parents[1]
    with (root / "pyproject.toml").open("rb") as f:
        pyproject = tomllib.load(f)
    with (root / "tox.toml").open("rb") as f:
        tox = tomllib.load(f)

    declared = parse_declared_floors(pyproject)
    tested = parse_tested_floors(tox)

    failures: list[str] = []
    for backend in sorted(PYPROJECT_FLOOR_SOURCES):
        declared_versions, tested_versions = declared[backend], tested[backend]
        if not declared_versions:
            failures.append(f"{backend}: no floor declared in pyproject.toml")
            continue
        if len(declared_versions) > 1:
            failures.append(f"{backend}: inconsistent declared floors in pyproject.toml: {sorted(declared_versions)}")
            continue
        if not tested_versions:
            failures.append(f"{backend}: no floor pin in tox.toml")
            continue
        if declared_versions != tested_versions:
            failures.append(
                f"{backend}: pyproject.toml declares >={next(iter(declared_versions))}, "
                f"tox.toml pins {sorted(tested_versions)}"
            )

    if failures:
        for failure in failures:
            print(f"matrix floor drift: {failure}", file=sys.stderr)
        print(
            "tox.toml floors and pyproject.toml floors are different. Make both state the same floor.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
