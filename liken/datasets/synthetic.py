import typing

import pandas as pd
import polars as pl
import pyspark.sql as spark
from pyspark.sql import SparkSession


# fmt: off


COLUMNS = [
    "id", "address", "email", "account",
    "birth_country", "marital_status", "number_children", "property_type", 
    "property_height", "property_area_sq_ft", "property_sea_level_elevation_m", "property_num_rooms"
]

DATA10 = [
    [1, "123ab, OL5 9PL, UK", "bbab@example.com", "reddit", "spain", "married", 1, "rental", None, 545, 5, 3],
    [2, "99 Ambleside avenue park Road, ED3 3RT, Edinburgh, United Kingdom", "awesome_surfer_77@yahoo.com", "reddit", "spain", "married", 1, "rental", None, 452, 6, 3,],
    [3, "Calle Ancho, 12, 05688, Rioja, Navarra, Espana", "a@example.com", "facebook", "germany", "single", 2, "rental", 2.5, 623, 5, 3],
    [4, "Calle Sueco, 56, 05688, Rioja, Navarra", "hellothere@example.com", "pinterest", "japan", "married", 0, "owner", 4.0, 2077, 305, 6],
    [5, None, "b@example.com", "linkedin", "france", "married", 1, "rental", 2.7, 1045, 42, 4],
    [6, "C. Ancho 49, 05687, Navarra", "b@example.com", "reddit", "japan", "married", 1, "rental", 2.5, 1323, 132, 4],
    [7, "Ambleside avenue Park Road ED3, UK", "hellthere@example.com", "reddit", "germany", "married", 0, "owner", 2.5, 509, 200, 2],
    [8, "123ab, OL5 9PL, UK", "hellathere@example.com", "facebook", "japan", "single", 3, "owner", 2.5, 500, 300, 3],
    [9, None, "yet.another.email@msn.com", "flickr", "germany", "married", 1, "rental", 2.5, 345, 22, 3],
    [10, "66b Porters street, OL5 9PL, Newark, United Kingdom", "bab@example.com", "flickr", "malaysia", "single", 0, "owner", 2.5, 4000, 25, 8], 
]

# fmt: on


def fake_10(
    backend: typing.Literal["pandas", "polars", "spark"] = "pandas",
    spark_session: SparkSession | None = None,
) -> pd.DataFrame | pl.DataFrame | spark.DataFrame:
    """Synthetic 10 rows.

    Args:
        backend: One of "pandas", "polars" or "spark".
        spark_session: The pyspark spark session if requesting data using
            "spark" backend.

    Returns:
        A dataframe, in the defined backend.

    Raises:
        ValueError: if no spark session passed when requesting a spark dataframe.
    """
    if backend == "pandas":
        return pd.DataFrame(columns=COLUMNS, data=DATA10)
    if backend == "polars":
        return pl.DataFrame(schema=COLUMNS, data=DATA10, orient="row")
    if backend == "spark":
        if spark_session:
            return spark_session.createDataFrame(schema=COLUMNS, data=DATA10)
        raise ValueError("Spark Session not passed yet 'spark' backend requested")


def fake_1K(
    backend: typing.Literal["pandas", "polars", "spark"] = "pandas",
    spark_session: SparkSession | None = None,
) -> pd.DataFrame | pl.DataFrame | spark.DataFrame:
    """Synthetic 1K (one thousand) rows.

    Args:
        backend: One of "pandas", "polars" or "spark".
        spark_session: The pyspark spark session if requesting data using
            "spark" backend.

    Returns:
        A dataframe, in the defined backend.

    Raises:
        ValueError: if no spark session passed when requesting a spark dataframe.

    Warning: Not Implemented
        Due for implementation in a future version.
    """
    del backend, spark_session  # Unused
    pass


def fake_100K(
    backend: typing.Literal["pandas", "polars", "spark"] = "pandas",
    spark_session: SparkSession | None = None,
) -> pd.DataFrame | pl.DataFrame | spark.DataFrame:
    """Synthetic 100K (one hundred thousand) rows.

    Args:
        backend: One of "pandas", "polars" or "spark".
        spark_session: The pyspark spark session if requesting data using
            "spark" backend.

    Returns:
        A dataframe, in the defined backend.

    Raises:
        ValueError: if no spark session passed when requesting a spark dataframe.

    Warning: Not Implemented
        Due for implementation in a future version.
    """
    del backend, spark_session  # Unused
    pass


def fake_1M(
    backend: typing.Literal["pandas", "polars", "spark"] = "pandas",
    spark_session: SparkSession | None = None,
) -> pd.DataFrame | pl.DataFrame | spark.DataFrame:
    """Synthetic 1M (one million) rows.

    Args:
        backend: One of "pandas", "polars" or "spark".
        spark_session: The pyspark spark session if requesting data using
            "spark" backend.

    Returns:
        A dataframe, in the defined backend.

    Raises:
        ValueError: if no spark session passed when requesting a spark dataframe.

    Warning: Not Implemented
        Due for implementation in a future version.
    """
    del backend, spark_session  # Unused
    pass
