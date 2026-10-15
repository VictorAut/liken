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
    ["dask", "modin", "ray"],
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


################################
# custom.register validation  #
################################


def test_custom_register_rejects_non_callable():
    """U30: register(42) deve falhar com erro nomeado, não AttributeError."""
    import liken as lk
    from liken.exceptions import InvalidArgumentTypeError

    with pytest.raises(InvalidArgumentTypeError, match="`f` must be callable, got 'int'"):
        lk.custom.register(42)


def test_custom_register_rejects_uninspectable_signature():
    """U30: callable com assinatura não inspecionável é rejeitado no registro."""
    import liken as lk
    from liken.exceptions import InvalidArgumentTypeError

    class _Uninspectable:
        def __call__(self, *args, **kwargs):
            return None

        @property
        def __signature__(self):
            raise ValueError("no signature")

    with pytest.raises(InvalidArgumentTypeError, match="inspectable signature") as info:
        lk.custom.register(_Uninspectable())

    assert isinstance(info.value.__cause__, ValueError)


def test_pyspark_is_match_surfaces_non_import_errors(monkeypatch):
    """U49: o guard de `is_match` do pyspark cobre apenas o import da
    dependência opcional; outros erros devem subir até o dispatcher."""
    from liken.backends.pyspark.backend import PysparkBackend

    real_import = builtins.__import__

    def broken_import(name, *args, **kwargs):
        if name == "pyspark.sql":
            raise TypeError("broken backend library: pyspark")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_import)

    with pytest.raises(TypeError, match="broken backend library"):
        PysparkBackend().is_match(pd.DataFrame())
