def test_all_extras(expect_backends):
    expect_backends(
        [
            # DEFAULT:
            "pandas",
            "polars",
            "pyarrow",
            # OPTIONAL:
            "modin",
            "dask",
            "ray",
            "pyspark",
        ]
    )
