---
title: Installation and Quick Start
---

## Installation

Liken requires **Python 3.11+**.

Pick the workflow that matches your project's dependencies.

=== "uv"

    If you've set up a project with [uv](https://docs.astral.sh/uv/), add Liken to it with:

    ```bash
    uv add liken
    ```

    This will create a virtual environment (if one doesn't already exist), add `liken` to your `pyproject.toml`, and update `uv.lock`.

    To install Narwhals into an existing virtual environment without touching `pyproject.toml`, use:

    ```bash
    uv pip install liken
    ```

=== "pip"

    [Create and activate](https://docs.python.org/3/library/venv.html) a Python 3.11+ virtual environment and run:

    ```bash
    pip install liken
    ```

    `pip install` will not touch `pyproject.toml`. If you're working on a project, you'll need to record the dependency there yourself.

=== "poetry"

    From within a [Poetry](https://python-poetry.org/) project, run:

    ```bash
    poetry add liken
    ```

    This will add `liken` to your `pyproject.toml` and update `poetry.lock`.


### Extras

Liken supports `pandas`, `polars` and `pyarrow` in the default installation. Liken also exposes support for [multiple other DataFrame libraries](./index.md#supported-dataframe-libraries). For the authoritative list on optional dependencies, see `[project.optional-dependencies]` in [`pyproject.toml`](https://github.com/VictorAut/liken/blob/main/pyproject.toml).

Install them optionally by specifying one or more extras in square brackets, or by specifying `all` for all optional depencies:

=== "uv"

    ```bash
    uv add "liken[dask]"            # Installs dask extra
    uv add "liken[dask,pyspark]"    # Installs dask and pyspark extras
    uv add "liken[all]"             # Install all extras
    ```

=== "pip"

    ```bash
    pip install "liken[dask]"            # Installs dask extra
    pip install "liken[dask,pyspark]"    # Installs dask and pyspark extras
    pip install "liken[all]"             # Install all extras
    ```

=== "poetry"

    ```bash
    poetry add "liken[dask]"            # Installs dask extra
    poetry add "liken[dask,pyspark]"    # Installs dask and pyspark extras
    poetry add "liken[all]"             # Install all extras
    ```

## Use `liken` In Your Code

### Fuzzy Deduplication

```python exec="yes" source="above" session="quickstart" result="python"
import liken as lk
import pandas as pd

df = pd.DataFrame(
    {
        "id": [1, 2, 3],
        "address": [
            "london",
            "tokyo",
            "paris",
        ],
        "email": [
            "fizzpop@yahoo.com",
            "FizzPop@yaoo.com",
            "a@msn.fr",
        ],
    }
)

deduper = lk.dedupe(df).apply(lk.fuzzy(threshold=0.7))

print("With Liken fuzzy deduplication at threshold:")
print(deduper.drop_duplicates("email"))

print("Normal Pandas only exact match dedupication:")
print(df.drop_duplicates("email"))

```

### Canonicalization

```python exec="yes" source="above" session="quickstart" result="python"
import liken as lk
import pandas as pd

df = pd.DataFrame(
    {
        "id": [1, 2, 3],
        "address": [
            "london",
            "tokyo",
            "paris",
        ],
        "email": [
            "fizzpop@yahoo.com",
            "FizzPop@yaoo.com",
            "a@msn.fr",
        ],
    }
)

deduper = lk.dedupe(df).apply(lk.fuzzy(threshold=0.7))

print(deduper.canonicalize("email").collect())

```

Jump to the [tutorial](tutorials/first-steps.md) to dive deeper into how to build incrementally complex pipelines.
