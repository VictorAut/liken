import pytest

from liken.backends.pyspark.backend import PysparkBackend
from liken.exceptions import InvalidArgumentValueError


def test_create_df_requires_spark_session():
    with pytest.raises(InvalidArgumentValueError, match="Spark session required"):
        PysparkBackend().create_df([("x",)], schema=None)
