from typing import final

from liken.constants import CANONICAL_ID
from liken.constants import SPARK_SESSION_REQUIRED_MSG
from liken.core.backend import Backend
from liken.core.registries import backends_registry
from liken.exceptions import InvalidArgumentValueError


@final
@backends_registry.register("pyspark")
class PysparkBackend(Backend):
    name = "pyspark"
    supports_explore = False

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
            raise InvalidArgumentValueError(SPARK_SESSION_REQUIRED_MSG)
        return spark_session.createDataFrame(data=data, schema=schema)

    def executor(self, spark_session=None):

        from liken.backends.pyspark.executor import PysparkExecutor

        return PysparkExecutor(spark_session=spark_session)

    def wrap(self, df, source_id=None, canonical_id=CANONICAL_ID):

        from liken.backends.pyspark.wrapper import PysparkDF
        from liken.backends.pyspark.wrapper import PysparkRows

        if isinstance(df, list):
            return PysparkRows(df, canonical_id)

        return PysparkDF(df, source_id, canonical_id)
