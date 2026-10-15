import pytest

from liken.backends.pyspark.backend import PysparkBackend
from liken.exceptions import InvalidArgumentValueError


def test_create_df_requires_spark_session():
    with pytest.raises(InvalidArgumentValueError, match="spark_session arg must be provided for a spark dataframe"):
        PysparkBackend().create_df([("x",)], schema=None)
