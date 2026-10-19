import pytest

import liken as lk
from liken.backends.pyspark.backend import PysparkBackend
from liken.backends.pyspark.executor import PysparkExecutor
from liken.constants import CANONICAL_ID


pytest.importorskip("pyspark")


@pytest.fixture(scope="module")
def spark():
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.master("local[1]").config("spark.driver.memory", "512m").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    yield spark
    spark.stop()


@pytest.fixture
def df(spark):
    return spark.createDataFrame([(1, "london"), (2, "london"), (3, "paris")], schema=["id", "address"])


def test_executor_derives_session_from_frame(df):
    wdf = PysparkBackend().wrap(df, canonical_id=CANONICAL_ID)
    executor = PysparkExecutor()

    result = executor.execute(
        wdf,
        columns=None,
        dedupers={"address": [lk.exact()]},
        keep="first",
        drop_duplicates=True,
        drop_canonical_id=False,
        source_id=None,
        canonical_id=CANONICAL_ID,
    ).unwrap()

    rows = result.orderBy("id").collect()
    assert [row["id"] for row in rows] == [1, 3]
    assert len({row[CANONICAL_ID] for row in rows}) == 2
