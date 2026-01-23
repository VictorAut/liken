import pytest

from dupegrouper._constants import CANONICAL_ID
from dupegrouper._dataframe import wrap


@pytest.mark.parametrize(
    "env_var_value",
    [
        "canonical_id",
        None,
        "random_id",
    ],
)
def test_canonical_id_env_var(env_var_value, dataframe, monkeypatch):
    df, _ = dataframe

    monkeypatch.delenv("CANONICAL_ID", raising=False)
    if env_var_value is not None:
        monkeypatch.setenv("CANONICAL_ID", env_var_value)

    assert CANONICAL_ID not in df.columns

    wdf = wrap(df)
    df = wdf.unwrap()

    assert CANONICAL_ID in df.columns

    monkeypatch.delenv("CANONICAL_ID", raising=False)
