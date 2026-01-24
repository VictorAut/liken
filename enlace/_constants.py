from __future__ import annotations

import os
from typing import Final

from pyspark.sql.types import (
    BooleanType,
    DataType,
    DateType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    StringType,
    TimestampType,
)


# CONSTANTS:


# Default canonical_id label in the dataframe
CANONICAL_ID: Final[str] = os.environ.get("CANONICAL_ID", "canonical_id")

NA_PLACEHOLDER: Final[str] = "na"


# Pyspark sql conversion types
PYSPARK_TYPES: Final[dict[str, DataType]] = {
    "boolean": BooleanType(),
    "date": DateType(),
    "double": DoubleType(),
    "float": FloatType(),
    "int": IntegerType(),
    "bigint": LongType(),
    "string": StringType(),
    "timestamp": TimestampType(),
}
