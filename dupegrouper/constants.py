from __future__ import annotations
import os
import typing

from pyspark.sql.types import (
    StringType,
    IntegerType,
    LongType,
    DoubleType,
    FloatType,
    BooleanType,
    TimestampType,
    DateType,
)



# CONSTANTS:


# Default canonical_id label in the dataframe
CANONICAL_ID: typing.Final[str] = os.environ.get("CANONICAL_ID", "canonical_id")

# Pyspark sql conversion types
PYSPARK_TYPES = {
    "string": StringType(),
    "int": IntegerType(),
    "bigint": LongType(),
    "double": DoubleType(),
    "float": FloatType(),
    "boolean": BooleanType(),
    "timestamp": TimestampType(),
    "date": DateType(),
    # ...
}
