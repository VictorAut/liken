from typing import final

import polars as pl

from liken.backends.polars.executor import PolarsExecutor
from liken.backends.polars.wrapper import PolarsDF
from liken.core.backend import Backend
from liken.core.registries import backends_registry


@final
@backends_registry.register("polars")
class PolarsBackend(Backend):
    name = "polars"

    def is_match(self, df):

        return isinstance(df, pl.DataFrame)

    def create_df(self, data, schema, **kwargs):
        del kwargs  # Unused

        return pl.DataFrame(schema=schema, data=data, orient="row")

    def executor(self, **kwargs):
        del kwargs  # Unused
        return PolarsExecutor()

    def wrap(self, df, id=None):

        return PolarsDF(df, id)
