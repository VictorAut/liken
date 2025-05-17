from unittest.mock import Mock, patch

import pandas as pd
import polars as pl
from pyspark.sql import SparkSession, DataFrame as SparkDataFrame
import pytest

from dupegrouper.base import DupeGrouper, DeduplicationStrategy
from dupegrouper.definitions import DataFrameLike
from dupegrouper.wrappers.dataframes import (
    WrappedPandasDataFrame,
    WrappedPolarsDataFrame,
    WrappedSparkDataFrame,
    WrappedSparkRows,
)


@pytest.fixture(scope="session")
def id():
    return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]


@pytest.fixture(scope="session")
def address():
    return [
        "123ab, OL5 9PL, UK",
        "99 Ambleside avenue park Road, ED3 3RT, Edinburgh, United Kingdom",
        "Calle Ancho, 12, 05688, Rioja, Navarra, Espana",
        "Calle Sueco, 56, 05688, Rioja, Navarra",
        "4 Brinkworth Way, GH9 5KL, Edinburgh, United Kingdom",
        "66b Porters street, OL5 9PL, Newark, United Kingdom",
        "C. Ancho 49, 05687, Navarra",
        "Ambleside avenue Park Road ED3, UK",
        "123ab, OL5 9PL, UK",
        "123ab, OL5 9PL, UK",
        "37 Lincolnshire lane, GH9 5DF, Edinburgh, UK",
        "37 GH9, UK",
        "totally random non existant address",
    ]


@pytest.fixture(scope="session")
def email():
    return [
        "bbab@example.com",
        "bb@example.com",
        "a@example.com",
        "hellothere@example.com",
        "b@example.com",
        "bab@example.com",
        "b@example.com",
        "hellthere@example.com",
        "hellathere@example.com",
        "irrelevant@hotmail.com",
        "yet.another.email@msn.com",
        "awesome_surfer_77@yahoo.com",
        "fictitious@never.co.uk",
    ]


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
def group_id():
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
    yield spark
    spark.stop()


# raw data i.e. no "GROUP ID"


@pytest.fixture(scope="session")
def df_pandas(id, address, email):
    return pd.DataFrame({"id": id, "address": address, "email": email})


@pytest.fixture(scope="session")
def df_polars(id, address, email):
    return pl.DataFrame({"id": id, "address": address, "email": email})


@pytest.fixture(scope="session")
def df_spark(spark, id, address, email, blocking_key):
    """default is a single partition"""
    return spark.createDataFrame(
        [[id[i], address[i], email[i], blocking_key[i]] for i in range(len(id))],
        schema=("id", "address", "email", "blocking_key"),
    ).repartition(1, "blocking_key")


@pytest.fixture(params=["pandas", "polars", "spark"], scope="session")
def dataframe(request, df_pandas, df_polars, df_spark, spark):
    """return a tuple of positionally ordered input parameters of DupeGrouper

    This is useful for implementations that ARE part of the public API
    """
    if request.param == "pandas":
        return df_pandas, None, None
    if request.param == "polars":
        return df_polars, None, None
    if request.param == "spark":
        return df_spark, spark, "id"


@pytest.fixture(params=["pandas", "polars", "spark_df", "spark_row"], scope="session")
def lowlevel_dataframe(request, df_pandas, df_polars, df_spark):
    """Most tests require the `dataframe` fixture, also defined above.

    However, this fixture offers all wrappers exhaustively, including the lower
    level wrapper for spark.

    This is useful for testing lower level implementations that are NOT part of
    the public API
    """
    if request.param == "pandas":
        return df_pandas, WrappedPandasDataFrame, None
    if request.param == "polars":
        return df_polars, WrappedPolarsDataFrame, None
    if request.param == "spark_df":
        return df_spark, WrappedSparkDataFrame, None
    if request.param == "spark_row":
        return df_spark.collect(), WrappedSparkRows, "id"  # i.e. list[Row]


# Mocks


@pytest.fixture
def dupegrouper_mock(dataframe):
    df, _, id = dataframe

    df_mock = Mock(spec=type(df))

    with patch("dupegrouper.base._wrap"):
        instance = DupeGrouper(df_mock, id)
        instance._df = Mock()
        yield instance


@pytest.fixture
def strategy_mock():
    return Mock(spec=DeduplicationStrategy)


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


@pytest.fixture(scope="session", autouse=True)
def helpers():
    return Helpers
