from unittest.mock import Mock
from unittest.mock import create_autospec

import dask.dataframe as dd
import modin.pandas as mpd
import pandas as pd
import polars as pl
import pytest
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.functions import row_number
from pyspark.sql.types import LongType
from pyspark.sql.types import StringType
from pyspark.sql.window import Window
from ray.data import Dataset as RayDataset

from liken.datasets import fake_10
from liken.liken import BaseDeduper


# ADDITIONAL DATA COLUMNS:


@pytest.fixture(scope="session")
def blocking_key():
    return [
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
    ]


@pytest.fixture(scope="session")
def canonical_id():
    return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


# SPARK:


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


# DATAFRAMES FOR INTEGRATION TESTS:


@pytest.fixture(scope="session")
def df_pandas():
    return fake_10("pandas")


@pytest.fixture(scope="session")
def df_polars():
    return fake_10("polars")


@pytest.fixture(scope="session")
def df_modin():
    return fake_10("modin")


@pytest.fixture(scope="session")
def df_ray():
    return fake_10("ray")


@pytest.fixture(scope="session")
def df_dask():
    return fake_10("dask")


@pytest.fixture(scope="function")
def df_spark(spark):
    """Default is a single partition"""
    return fake_10("spark", spark_session=spark)


@pytest.fixture(scope="function")
def df_sparkrows(df_spark):
    return df_spark.collect()


@pytest.fixture(params=["pandas", "polars", "modin", "ray", "dask", "spark"])
def dataframe(
    request,
    df_pandas,
    df_polars,
    df_modin,
    df_ray,
    df_dask,
    df_spark,
    spark,
):
    """return a tuple of positionally ordered input parameters of Dedupe

    This is useful for implementations that ARE part of the public API
    """
    if request.param == "pandas":
        return df_pandas, None
    if request.param == "polars":
        return df_polars, None
    if request.param == "modin":
        return df_modin, None
    if request.param == "ray":
        return df_ray, None
    if request.param == "dask":
        return df_dask, None
    if request.param == "spark":
        return df_spark, spark


# MOCKS:


@pytest.fixture
def deduplication_mock():
    return Mock(spec=BaseDeduper)


@pytest.fixture
def mock_spark_session():
    return create_autospec(SparkSession)


# HELPERS:


class Helpers:
    @staticmethod
    def get_column_as_list(df, col: str):
        if isinstance(df, pd.DataFrame):
            return [None if v is pd.NA else v for v in list(df[col])]
        if isinstance(df, pl.DataFrame):
            return list(df[col])
        if isinstance(df, mpd.DataFrame):
            return [None if v is pd.NA else v for v in list(df[col])]
        if isinstance(df, RayDataset):
            return [row[col] for row in df.take_all()]
        if isinstance(df, dd.DataFrame):
            df = df.compute()
            return df[col].tolist()
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

        if isinstance(df, mpd.DataFrame):
            df = df.assign(**{label: column})
            return df

        if isinstance(df, dd.DataFrame):

            def add_col(partition, partition_info=None):
                i = partition_info["number"]
                start = sum(df.map_partitions(len).compute()[:i])
                end = start + len(partition)

                partition = partition.copy()
                partition[label] = column[start:end]
                return partition

            meta = df._meta.copy()
            meta[label] = pd.Series(dtype=dtype if dtype else "object")

            return df.map_partitions(add_col, meta=meta)

        if isinstance(df, RayDataset):

            def add_col(batch):
                batch = batch.copy()
                batch[label] = column[: len(batch)]
                return batch

            return df.map_batches(add_col, batch_format="pandas")

        if isinstance(df, SparkDataFrame):
            if dtype is int:
                _type = LongType()
            if dtype is str:
                _type = StringType()
            labels_udf = F.udf(lambda indx: column[indx - 1], _type)
            df = df.withColumn(
                "num_id",
                row_number().over(Window.orderBy(monotonically_increasing_id())),
            )
            df = df.withColumn(label, labels_udf("num_id"))
            return df.drop("num_id")
        if isinstance(df, list):  # i.e. list[Row]
            pass


@pytest.fixture(scope="session", autouse=True)
def helpers():
    return Helpers
