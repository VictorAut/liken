from typing import final

from liken.constants import SPARK_SESSION_REQUIRED_FOR_CREATE_DF_MSG
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
        # frame factory: no frame exists yet to infer a session from
        if spark_session is None:
            raise InvalidArgumentValueError(SPARK_SESSION_REQUIRED_FOR_CREATE_DF_MSG)
        return spark_session.createDataFrame(data=data, schema=schema)

    def executor(self):

        from liken.backends.pyspark.executor import PysparkExecutor

        return PysparkExecutor()

    def wrap(self, df, source_id=None, *, canonical_id: str):

        from liken.backends.pyspark.wrapper import PysparkDF
        from liken.backends.pyspark.wrapper import PysparkRows

        if isinstance(df, list):
            return PysparkRows(df, canonical_id)

        return PysparkDF(df, source_id, canonical_id=canonical_id)
