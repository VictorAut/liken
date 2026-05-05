def test_all_extras(expect_backends):
    expect_backends(
        [
            "pandas",
            "polars",
            "pyspark",
        ]
    )
