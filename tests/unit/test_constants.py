import importlib
import os

import pytest

import dupegrouper.constants
import dupegrouper.dataframe
from dupegrouper.dataframe import SparkDF, SparkRows



def reload_imports():
    importlib.reload(dupegrouper.constants)
    importlib.reload(dupegrouper.dataframe)


@pytest.mark.parametrize(
    "env_var_value, expected_value",
    [
        # default
        ("canonical_id", "canonical_id"),
        # override to default
        (None, "canonical_id"),
        # arbitrary: different value
        ("random_id", "random_id"),
    ],
)
def test_canonical_id_env_var(env_var_value, expected_value, lowlevel_dataframe):
    df, wrapper, id = lowlevel_dataframe

    # remove the env var
    os.environ.pop("CANONICAL_ID", None)

    # and repopulate it if available
    if env_var_value:
        os.environ["CANONICAL_ID"] = env_var_value

    reload_imports()

    df = wrapper(df, id)

    if isinstance(df, SparkDF):
        assert expected_value not in df.columns
    elif isinstance(df, SparkRows):
        for row in df.unwrap():
            assert expected_value in row.asDict().keys()
    else:
        assert expected_value in df.columns