import pytest

from dupegrouper.constants import CANONICAL_ID
from dupegrouper.dataframe import SparkDF, SparkRows


@pytest.mark.parametrize(
    "env_var_value",
    [
        "canonical_id",
        None,
        "random_id",
    ],
)
def test_canonical_id_env_var(env_var_value, lowlevel_dataframe, monkeypatch):
    df, wrapper, id = lowlevel_dataframe

    monkeypatch.delenv("CANONICAL_ID", raising=False)
    if env_var_value is not None:
        monkeypatch.setenv("CANONICAL_ID", env_var_value)

    df = wrapper(df, id)

    if isinstance(df, SparkDF):
        assert CANONICAL_ID not in df.columns
    elif isinstance(df, SparkRows):
        for row in df.unwrap():
            assert CANONICAL_ID in row.asDict().keys()
    else:
        assert CANONICAL_ID in df.columns

    monkeypatch.delenv("CANONICAL_ID", raising=False)
