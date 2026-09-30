"""Dispatches a dataframe to its matching backend implementation."""

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
