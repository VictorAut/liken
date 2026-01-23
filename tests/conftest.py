from unittest.mock import Mock, create_autospec

import pandas as pd
import polars as pl
import pytest
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import monotonically_increasing_id, row_number
from pyspark.sql.types import LongType, StringType
from pyspark.sql.window import Window

from dupegrouper._types import DataFrameLike
from dupegrouper.datasets.synthetic import fake_13
from dupegrouper.dedupe import BaseStrategy


@pytest.fixture(scope="session")
def blocking_key():
    return [
        "key_2",
        "key_2",
        "key_2",
        "key_2",
        "key_2",
        "key_2",
        "key_1",
        "key_1",
        "key_1",
        "key_1",
        "key_1",
        "key_1",
        "key_1",
    ]


@pytest.fixture(scope="session")
def canonical_id():
    return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]


@pytest.fixture(scope="session")
def spark():
    spark = (
        SparkSession.builder.master("local[1]")
        .appName("local-tests")
        .config("spark.executor.cores", "1")
        .config("spark.executor.instances", "1")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    yield spark
    spark.stop()


# raw data inputs


@pytest.fixture(scope="session")
def df_pandas():
    return fake_13("pandas")


@pytest.fixture(scope="session")
def df_polars():
    return fake_13("polars")


@pytest.fixture(scope="function")
def df_spark(spark):
    """Default is a single partition"""
    return fake_13("spark", spark_session=spark)


@pytest.fixture(scope="function")
def df_sparkrows(df_spark):
    return df_spark.collect()


@pytest.fixture(params=["pandas", "polars", "spark"])
def dataframe(request, df_pandas, df_polars, df_spark, spark):
    """return a tuple of positionally ordered input parameters of Duped

    This is useful for implementations that ARE part of the public API
    """
    if request.param == "pandas":
        return df_pandas, None
    if request.param == "polars":
        return df_polars, None
    if request.param == "spark":
        return df_spark, spark


# Mocks


@pytest.fixture
def strategy_mock():
    return Mock(spec=BaseStrategy)


@pytest.fixture
def mock_spark_session():
    return create_autospec(SparkSession)


# helpers


class Helpers:

    @staticmethod
    def get_column_as_list(df: DataFrameLike, col: str):
        if isinstance(df, pd.DataFrame) or isinstance(df, pl.DataFrame):
            return list(df[col])
        if isinstance(df, SparkDataFrame):
            return [value[col] for value in df.select(col).collect()]
        if isinstance(df, list):  # i.e. list[Row]
            return [value[col] for value in df]

    @staticmethod
    def add_column(df, column: list, label: str, dtype=None):
        if isinstance(df, pd.DataFrame):
            df = df.assign(**{label: column})
            return df
        if isinstance(df, pl.DataFrame):
            df = df.with_columns(pl.Series(name=label, values=column))
            return df
        if isinstance(df, SparkDataFrame):
            # add column to spark DataFrame
            if dtype is int:
                _type = LongType()
            if dtype is str:
                _type = StringType()
            labels_udf = F.udf(lambda indx: column[indx - 1], _type)
            df = df.withColumn("num_id", row_number().over(Window.orderBy(monotonically_increasing_id())))
            df = df.withColumn(label, labels_udf("num_id"))
            return df.drop("num_id")
        if isinstance(df, list):  # i.e. list[Row]
            # TODO
            pass


@pytest.fixture(scope="session", autouse=True)
def helpers():
    return Helpers
