"""Unit tests for the `distributed` class flag on dataframe wrappers."""

import pytest

from liken.backends.dask.wrapper import DaskDF
from liken.backends.modin.wrapper import ModinDF
from liken.backends.pandas.wrapper import PandasDF
from liken.backends.polars.wrapper import PolarsDF
from liken.backends.pyarrow.wrapper import ArrowDF
from liken.backends.pyspark.wrapper import PysparkDF
from liken.backends.ray.wrapper import RayDF
from liken.core.wrapper import DF


def test_base_df_flag_is_false():
    assert DF.distributed is False


@pytest.mark.parametrize(
    "wrapper, expected",
    [
        (PandasDF, False),
        (PolarsDF, False),
        (ArrowDF, False),
        (PysparkDF, True),
        (DaskDF, True),
        (RayDF, True),
        (ModinDF, True),
    ],
    ids=["pandas", "polars", "pyarrow", "pyspark", "dask", "ray", "modin"],
)
def test_wrapper_distributed_flag(wrapper, expected):
    assert wrapper.distributed is expected
