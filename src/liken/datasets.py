from __future__ import annotations

import random
from typing import TYPE_CHECKING
from typing import Any

from faker import Faker

from liken.core.registries import backends_registry
from liken.types import SupportedBackends
from liken.types import UserDataFrame


if TYPE_CHECKING:
    from pyspark.sql import SparkSession


Faker.seed(123)


# CONSTANTS:


# fmt: off


_SCHEMA10 = [
    "id", "address", "email", "account",
    "birth_country", "marital_status", "number_children", "property_type", 
    "property_height", "property_area_sq_ft", "property_sea_level_elevation_m", "property_num_rooms"
]

_DATA10 = [
    (1, "123ab, OL5 9PL, UK", "bbab@example.com", "reddit", "spain", "married", 1, "rental", None, 545, 5, 3),
    (2, "99 Ambleside avenue park Road, ED3 3RT, Edinburgh, United Kingdom", "awesome_surfer_77@yahoo.com", "reddit", "spain", "married", 1, "rental", None, 452, 6, 3),
    (3, "Calle Ancho, 12, 05688, Rioja, Navarra, Espana", "a@example.com", "facebook", "germany", "single", 2, "rental", 2.5, 623, 5, 3),
    (4, "Calle Sueco, 56, 05688, Rioja, Navarra", "hellothere@example.com", "pinterest", "japan", "married", 0, "owner", 4.0, 2077, 305, 6),
    (5, None, "b@example.com", "linkedin", "france", "married", 1, "rental", 2.7, 1045, 42, 4),
    (6, "C. Ancho 49, 05687, Navarra", "b@example.com", "reddit", "japan", "married", 1, "rental", 2.5, 1323, 132, 4),
    (7, "Ambleside avenue Park Road ED3, UK", "hellthere@example.com", "reddit", "germany", "married", 0, "owner", 2.5, 509, 200, 2),
    (8, "123ab, OL5 9PL, UK", "hellathere@example.com", "facebook", "japan", "single", 3, "owner", 2.5, 500, 300, 3),
    (9, None, "yet.another.email@msn.com", "flickr", "germany", "married", 1, "rental", 2.5, 345, 22, 3),
    (10, "66b Porters street, OL5 9PL, Newark, United Kingdom", "bab@example.com", "flickr", "malaysia", "single", 0, "owner", 2.5, 4000, 25, 8),
]


# fmt: on


_SCHEMA10_PLUS = [
    "id",
    "first_name",
    "last_name",
    "email",
    "phone",
    "address",
    "city",
    "country",
    "company",
    "job",
    "date_of_birth",
    "signup_date",
    "salary",
    "is_active",
]


# HELPERS:


fake = Faker()


def _return_df(
    schema: list[str],
    data: list[tuple[Any, ...]],
    backend: SupportedBackends = "pandas",
    spark_session: SparkSession | None = None,
) -> UserDataFrame:
    """Returns the dataframe based on the backend"""

    try:
        backend_cls = backends_registry.get(backend)
    except KeyError:
        raise ValueError(f"Unsupported backend: {backend}")

    backend_instance = backend_cls()

    return backend_instance.create_df(
        data=data,
        schema=schema,
        spark_session=spark_session,
    )


def maybe_null(value, p):
    return value if random.random() > p else None


def fake_row():
    return (
        fake.uuid4(),
        maybe_null(fake.first_name(), 0.02),
        maybe_null(fake.last_name(), 0.02),
        maybe_null(fake.email(), 0.02),
        maybe_null(fake.phone_number().replace("x", ", "), 0.02),
        maybe_null(fake.address().replace("\n", ", "), 0.02),
        maybe_null(fake.city(), 0.02),
        maybe_null(fake.country(), 0.02),
        maybe_null(fake.company(), 0.02),
        maybe_null(fake.job(), 0.02),
        maybe_null(fake.date_of_birth(minimum_age=18, maximum_age=80), 0.02),
        maybe_null(fake.date_time_this_decade(), 0.02),
        maybe_null(round(random.uniform(30000, 150000), 2), 0.02),
        random.choice([True, False]),
    )


# PUBLIC:


def fake_10(
    backend: SupportedBackends = "pandas",
    spark_session: SparkSession | None = None,
) -> UserDataFrame:
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
    return _return_df(
        schema=_SCHEMA10,
        data=_DATA10,
        backend=backend,
        spark_session=spark_session,
    )


def fake_1K(
    backend: SupportedBackends = "pandas",
    spark_session: SparkSession | None = None,
) -> UserDataFrame:
    """Synthetic 1K (one thousand) rows.

    Args:
        backend: One of "pandas", "polars" or "spark".
        spark_session: The pyspark spark session if requesting data using
            "spark" backend.

    Returns:
        A dataframe, in the defined backend.

    Raises:
        ValueError: if no spark session passed when requesting a spark dataframe.
    """
    data = [fake_row() for _ in range(999)]
    data.append(data[-1])  # duplicate last row for quick-glance

    return _return_df(
        schema=_SCHEMA10_PLUS,
        data=data,
        backend=backend,
        spark_session=spark_session,
    )


def fake_100K(
    backend: SupportedBackends = "pandas",
    spark_session: SparkSession | None = None,
) -> UserDataFrame:
    """Synthetic 100K (one hundred thousand) rows.

    Args:
        backend: One of "pandas", "polars" or "spark".
        spark_session: The pyspark spark session if requesting data using
            "spark" backend.

    Returns:
        A dataframe, in the defined backend.

    Raises:
        ValueError: if no spark session passed when requesting a spark dataframe.
    """
    data = [fake_row() for _ in range(99_999)]
    data.append(data[-1])  # duplicate last row for quick-glance

    return _return_df(
        schema=_SCHEMA10_PLUS,
        data=data,
        backend=backend,
        spark_session=spark_session,
    )


def fake_1M(
    backend: SupportedBackends = "pandas",
    spark_session: SparkSession | None = None,
) -> UserDataFrame:
    """Synthetic 1M (one million) rows.

    Args:
        backend: One of "pandas", "polars" or "spark".
        spark_session: The pyspark spark session if requesting data using
            "spark" backend.

    Returns:
        A dataframe, in the defined backend.

    Raises:
        ValueError: if no spark session passed when requesting a spark dataframe.
    """
    data = [fake_row() for _ in range(999_999)]
    data.append(data[-1])  # duplicate last row for quick-glance

    return _return_df(
        schema=_SCHEMA10_PLUS,
        data=data,
        backend=backend,
        spark_session=spark_session,
    )
