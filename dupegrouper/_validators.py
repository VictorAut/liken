from typing import Any, Final, Literal

from pyspark.sql import SparkSession

from dupegrouper._strats_library import BaseStrategy


INVALID: Final[str] = "Invalid arg: "
INVALID_SPARK: Final[str] = INVALID + "spark_session must be provided for a spark dataframe"
INVALID_KEEP: Final[str] = INVALID + "keep must be one of 'first' or 'last'"
INVALID_STRAT: Final[str] = INVALID + "strat must be instance of BaseStrategy, got {}"


# TODO: typing here should be `Any`?
def validate_spark_args(spark_session: SparkSession | None = None, /) -> SparkSession:
    if not spark_session:
        raise ValueError(INVALID_SPARK)
    return spark_session


def validate_keep_arg(keep: Literal["first", "last"]) -> Literal["first", "last"]:
    if keep not in ("first", "last"):
        raise ValueError(INVALID_KEEP)
    return keep


def validate_strat_arg(strat: Any):
    if not isinstance(strat, BaseStrategy):
        raise TypeError(INVALID_STRAT.format(type(strat).__name__))
    return strat
