from typing import Any, Literal

from pyspark.sql import SparkSession

from dupegrouper.strats_library import BaseStrategy


# TODO: typing here should be `Any`?
def validate_spark_args(spark_session: SparkSession | None = None, /) -> SparkSession:
    if not spark_session:
        raise ValueError("Invalid arg: spark_session must be provided for a spark dataframe")
    return spark_session


def validate_keep_arg(keep: Literal["first", "last"]) -> Literal["first", "last"]:
    if keep not in ("first", "last"):
        raise ValueError("Invalid arg: keep must be one of 'first' or 'last'")
    return keep


def validate_strat_arg(strat: Any):
    if not isinstance(strat, BaseStrategy):
        raise TypeError(f"Invalid arg: strat must be instance of BaseStrategy, got {type(strat).__name__}")
    return strat
