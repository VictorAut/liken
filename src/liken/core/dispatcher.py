"""
This module provides wrappers to allow for a uniform interface across different
backends. The backends covered are:
    - Pandas
    - Polars
    - Spark DataFrames
    - Spark RDDs
    - Spark Rows

Whilst Pandas and Polars wrappers are similarly wrapped, note the following:
- Spark Rows inherits the majority of functionality related to getting
    columns, puting columns, fill na etc
- Conversely, Spark DataFrames take care of adding canonical IDs

Additional Points regarding Spark. Upon initialising the public API with a
Spark DataFrame, the wrapper will call the SparkDF class which will create
canonical IDs. However the output to this is RDDs which are then processed
by the executor into Spark Rows which are dispatched to worker nodes. Spark
Rows can be fully recovered to a Spark DataFrame using the same SparkDF class.

TODO:
    - CanonicalIdMixin should be defined first when inherited
    - A full interface can then be defined
"""

# from __future__ import annotations

from typing import Any

from liken.core.backend import Backend
from liken.core.registries import backends_registry
from liken.core.wrapper import DF


def get_backend(df: Any) -> Backend:
    for backend_cls in backends_registry.get_all().values():
        backend: Backend = backend_cls()  # instantiated

        try:
            if backend.is_match(df):
                return backend
        except Exception:
            continue
    raise ValueError(f"Unsupported dataframe type: {type(df)}")


# DISPATCHER:


def wrap(df: Any, id: str | None = None) -> DF:
    backend = get_backend(df)
    return backend.wrap(df, id=id)
