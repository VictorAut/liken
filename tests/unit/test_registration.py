import builtins

import pandas as pd
import pytest


def test_backends_are_registered():
    from liken.core.registries import backends_registry

    assert set(backends_registry.get_all().keys()) == {
        "pandas",
        "polars",
        "modin",
        "dask",
        "ray",
        "pyspark",
        "pyarrow",
    }


@pytest.mark.parametrize(
    "backend_name",
    ["dask", "modin", "ray", "pyspark"],
)
def test_is_match_false_when_backend_import_fails(backend_name, monkeypatch):
    """A backend whose import fails for any reason is not a match.

    The lazy backends probe their library with a guarded import. A
    library that is installed but broken must read as absent, not crash
    dispatch (found on the modin floor: dask 2023.9.2 cannot import
    under pandas 1.5.3 and raised TypeError past the ImportError guard).
    """
    from liken.core.registries import backends_registry

    real_import = builtins.__import__

    def broken_import(name, *args, **kwargs):
        if name in {"dask.dataframe", "modin.pandas", "ray", "pyspark.sql"}:
            raise TypeError(f"broken backend library: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_import)

    backend_cls = backends_registry.get(backend_name)

    assert backend_cls().is_match(pd.DataFrame()) is False
