from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Protocol


if TYPE_CHECKING:
    from pyspark.sql import SparkSession

    from liken.core.executor import Executor
    from liken.core.wrapper import DF
    from liken.types import UserDataFrame


class Backend(Protocol):
    name: str

    def is_match(self, df: object) -> bool: ...
    def create_df(
        self,
        data: list[tuple[Any, ...]],
        schema: list[str],
        spark_session: SparkSession | None = None,
    ) -> UserDataFrame: ...
    def executor(self, **kwargs: Any) -> Executor: ...
    def wrap(self, df: UserDataFrame, id: str | None = None) -> DF: ...
