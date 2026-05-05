from unittest.mock import Mock
from unittest.mock import create_autospec

import dask.dataframe as dd
import modin.pandas as mpd
import pandas as pd
import polars as pl
import pytest
import ray
from dask.distributed import Client
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.functions import row_number
from pyspark.sql.types import LongType
from pyspark.sql.types import StringType
from pyspark.sql.window import Window

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
def spark_session(request):
    backend = request.config.getoption("--backend")

    if backend != "pyspark":
        yield None
        return

    spark = (
        SparkSession.builder.master("local[1]")
        .appName("local-tests")
        .config("spark.executor.cores", "1")
        .config("spark.executor.instances", "1")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.driver.memory", "512m")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    yield spark
    spark.stop()


@pytest.fixture(scope="session", autouse=True)
def ray_session(request):
    backend = request.config.getoption("--backend")

    if backend != "ray":
        yield
        return

    ray.init(
        num_cpus=2,
    )
    yield
    ray.shutdown()


@pytest.fixture(scope="session", autouse=True)
def dask_client(request):
    backend = request.config.getoption("--backend")

    if backend != "dask":
        yield
        return

    client = Client(
        n_workers=2,
        threads_per_worker=1,
        processes=True,
    )

    yield client
    client.close()


def pytest_addoption(parser):
    parser.addoption(
        "--backend",
        action="store",
        default="pandas",
        help="Backend to run tests against",
    )


@pytest.fixture
def dataframe(
    request,
    spark_session,
):
    """return a tuple of positionally ordered input parameters of Dedupe

    This is useful for implementations that ARE part of the public API
    """

    backend = request.config.getoption("--backend")

    if backend == "pyspark":
        return fake_10("pyspark", spark_session=spark_session)
    try:
        return fake_10(backend)
    except Exception:
        raise ValueError(f"Unknown backend: {backend}")


# MOCKS:


@pytest.fixture
def deduplication_mock():
    return Mock(spec=BaseDeduper)


@pytest.fixture
def mock_spark_session():
    return create_autospec(SparkSession)


# HELPERS:


class Helpers:
    def __init__(self, request_object, spark_session=None):
        self.backend = request_object.config.getoption("--backend")
        self.spark_session = spark_session

    def get_column_as_list(self, df, col: str):

        if self.backend == "pandas":
            return [None if v is pd.NA else v for v in list(df[col])]

        if self.backend == "polars":
            return list(df[col])

        if self.backend == "modin":
            return [None if v is pd.NA else v for v in list(df[col])]

        if self.backend == "ray":
            return [row[col] for row in df.take_all()]

        if self.backend == "dask":
            if isinstance(df, dd.DataFrame):
                df = df.compute()
                return df[col].tolist()
            # i.e. pandas
            return [None if v is pd.NA else v for v in list(df[col])]

        if self.backend == "pyspark":
            return [value[col] for value in df.select(col).collect()]

    def add_column(self, df, column: list, label: str, dtype=None):

        if self.backend == "pandas":
            return df.assign(**{label: column})

        if self.backend == "polars":
            return df.with_columns(pl.Series(name=label, values=column))

        if self.backend == "modin":
            return df.assign(**{label: column})

        if self.backend == "dask":

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

        if self.backend == "ray":

            def add_col(batch):
                batch = batch.copy()
                batch[label] = column[: len(batch)]
                return batch

            return df.map_batches(add_col, batch_format="pandas")

        if self.backend == "pyspark":
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

    def create_df(self, data, schema):
        if self.backend == "pandas":
            df = pd.DataFrame(columns=schema, data=data)

        if self.backend == "polars":
            df = pl.DataFrame(schema=schema, data=data, orient="row")

        if self.backend == "modin":
            df = mpd.DataFrame(columns=schema, data=data)

        if self.backend == "dask":
            df = pd.DataFrame(columns=schema, data=data)
            df = dd.from_pandas(df)

        if self.backend == "ray":
            df = pd.DataFrame(columns=schema, data=data)

            df = ray.data.from_pandas(df)

        if self.backend == "pyspark":
            df = self.spark_session.createDataFrame(schema=schema, data=data)

        return df


@pytest.fixture(scope="session", autouse=True)
def helpers(request, spark_session):
    return Helpers(request, spark_session)
