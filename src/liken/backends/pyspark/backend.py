from typing import final

from liken.core.backend import Backend
from liken.core.registries import backends_registry


@final
@backends_registry.register("pyspark")
class SparkBackend(Backend):
    name = "pyspark"

    def is_match(self, df):
        try:
            import pyspark.sql as spark
            from pyspark.sql import Row
        except ImportError:
            return False

        if isinstance(df, spark.DataFrame):
            return True

        if isinstance(df, list) and df:
            return isinstance(df[0], Row)

        return False

    def create_df(self, data, schema, spark_session=None):
        if spark_session is None:
            raise ValueError("Spark session required")
        return spark_session.createDataFrame(data=data, schema=schema)

    def executor(self, spark_session=None):
        from liken.backends.pyspark.executor import SparkExecutor

        return SparkExecutor(spark_session=spark_session)

    def wrap(self, df, id=None):

        from liken.backends.pyspark.wrapper import SparkDF
        from liken.backends.pyspark.wrapper import SparkRows

        if isinstance(df, list):
            return SparkRows(df)

        return SparkDF(df, id)
