from typing import final

import pandas as pd

from liken.core.backend import Backend
from liken.core.registries import backends_registry


@final
@backends_registry.register("dask")
class DaskBackend(Backend):
    name = "dask"

    def is_match(self, df):
        try:
            import dask.dataframe as dd
        except ImportError:
            return False
        return isinstance(df, dd.DataFrame)

    def create_df(self, data, schema):
        import dask.dataframe as dd

        df = pd.DataFrame(columns=schema, data=data)
        return dd.from_pandas(df)

    def executor(self):
        from liken.backends.dask.executor import DaskExecutor

        return DaskExecutor()

    def wrap(self, df, id=None):
        from liken.backends.dask.wrapper import DaskDF

        return DaskDF(df, id)
