def test_backends_are_registered():
    from liken.core.registries import backends_registry

    assert set(backends_registry.get_all().keys()) == {
        "pandas",
        "polars",
        "modin",
        "dask",
        "ray",
        "pyspark",
    }
