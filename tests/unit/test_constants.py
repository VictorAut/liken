import pytest
from ray.data import Dataset

from liken.constants import CANONICAL_ID
from liken.core.dispatcher import wrap


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

    _assert(df, _not=True)

    wdf = wrap(df)
    df = wdf.unwrap()

    _assert(df, _not=False)

    monkeypatch.delenv("CANONICAL_ID", raising=False)


def _assert(df, _not: bool):
    if isinstance(df, Dataset):
        if _not:
            assert CANONICAL_ID not in df.columns()
        else:
            assert CANONICAL_ID in df.columns()
    else:
        if _not:
            assert CANONICAL_ID not in df.columns
        else:
            assert CANONICAL_ID in df.columns
