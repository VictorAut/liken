import polars as pl

from liken.datasets import fake_1K
from liken.datasets import fake_10


def test_fake_10_is_10_rows_pandas():
    df = fake_10()

    assert len(df) == 10


def test_fake_1k_has_1000_rows_pandas():
    df = fake_1K()

    assert len(df) == 1000


def test_fake_1k_last_row_duplicates_penultimate_pandas():
    df = fake_1K()

    assert df.iloc[-1].equals(df.iloc[-2])


def test_fake_1k_has_1000_rows_polars():
    df = fake_1K(backend="polars")

    assert isinstance(df, pl.DataFrame)
    assert len(df) == 1000


def test_fake_1k_last_row_duplicates_penultimate_polars():
    df = fake_1K(backend="polars")

    assert df.row(-1, named=True) == df.row(-2, named=True)


# NOTE: no unsupported-backend test. Found during this task: the
# "Unsupported backend" ValueError in datasets._return_df is dead code —
# the backend registry raises catalogue.RegistryError, which does not
# subclass KeyError, so the except KeyError never catches it and the raw
# registry error propagates. Recorded as a bug finding; a source fix is
# out of scope for this change.
