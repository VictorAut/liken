import dask.dataframe as dd
import modin.pandas as mpd
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

import liken as lk


DATA = [
    [1, "london"],
    [2, "london"],
    [3, "paris"],
]
COLS = ["id", "address"]

DEFAULT_METRICS = ["exact", "0.5", "0.75", "0.9", "0.95", "0.99"]


def make_pandas_df():
    return pd.DataFrame(data=DATA, columns=COLS)


# unsupported backend


def test_explore_rejects_unsupported_backend():
    df = dd.from_pandas(make_pandas_df(), npartitions=1)

    with pytest.raises(
        ValueError, match="explore is only supported for the pandas, polars, modin and pyarrow backends"
    ):
        lk.dedupe(df).explore(["address"])


# duplicate rates


def test_explore_reports_duplicate_rates_pyarrow():
    df = pa.table({"id": [1, 2, 3], "address": ["london", "london", "paris"]})

    result = lk.dedupe(df).explore(["address"])

    assert isinstance(result, pa.Table)
    assert result.column("metric").to_pylist() == DEFAULT_METRICS
    # two of three rows share an address
    assert result.column("address").to_pylist()[0] == pytest.approx(1 / 3)


def test_explore_reports_duplicate_rates_pandas():
    df = make_pandas_df()

    result = lk.dedupe(df).explore(["address"])

    assert list(result.index) == DEFAULT_METRICS
    assert list(result.columns) == ["address"]
    # two of three rows share an address
    assert result["address"]["exact"] == pytest.approx(1 / 3)


def test_explore_reports_duplicate_rates_polars():
    df = pl.DataFrame(data=DATA, schema=COLS, orient="row")

    result = lk.dedupe(df).explore(["address"], thresholds=[0.9])

    assert result["metric"].to_list() == ["exact", "0.9"]
    assert result["address"][0] == pytest.approx(1 / 3)


# sampling


@pytest.mark.parametrize("make_df", [make_pandas_df, lambda: pl.DataFrame(data=DATA, schema=COLS, orient="row")])
def test_explore_sampled_rates_are_well_formed(make_df):
    df = make_df()

    result = lk.dedupe(df).explore(["address"], frac=0.5)

    rates = result["address"] if isinstance(result, pl.DataFrame) else result["address"].tolist()
    assert all(0 <= rate <= 1 for rate in rates)
    metrics = result["metric"].to_list() if isinstance(result, pl.DataFrame) else result.index.tolist()
    assert metrics == DEFAULT_METRICS


def test_explore_sampled_rates_pyarrow():
    df = pa.table({"id": [1, 2, 3], "address": ["london", "london", "paris"]})

    result = lk.dedupe(df).explore(["address"], frac=0.5)

    assert isinstance(result, pa.Table)
    assert result.column("metric").to_pylist() == DEFAULT_METRICS
    assert all(0 <= rate <= 1 for rate in result.column("address").to_pylist())


def test_explore_reports_pyarrow_dict_dedupers():
    df = pa.table({"id": [1, 2, 3], "email": ["a@x.com", "a@x.com", "b@x.com"]})

    result = lk.dedupe(df).explore({"email": lk.tfidf()})

    assert isinstance(result, pa.Table)
    # two of three rows share an email
    assert result.column("email").to_pylist()[0] == pytest.approx(1 / 3)


def test_explore_sampled_rates_modin():
    df = mpd.DataFrame(data=DATA, columns=COLS)

    result = lk.dedupe(df).explore(["address"], frac=0.5)

    assert all(0 <= rate <= 1 for rate in result["address"].tolist())
    assert list(result.index) == DEFAULT_METRICS


# FIXME: on an empty dataframe the column converts to a pyarrow null-typed
# array and the NA-placeholder coalesce in DF.get_array raises
# ArrowNotImplementedError. Affects explore, drop_duplicates and
# canonicalize on every backend, so there is no empty-frame explore test
# here. Fixing the coalesce is a separate change.
