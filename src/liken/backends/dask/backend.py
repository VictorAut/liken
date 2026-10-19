from typing import final

import pandas as pd

from liken.core.backend import Backend
from liken.core.registries import backends_registry


@final
@backends_registry.register("dask")
class DaskBackend(Backend):
    name = "dask"
    supports_explore = False

    def is_match(self, df):
        try:
            import dask.dataframe as dd
        except Exception:
            return False
        return isinstance(df, dd.DataFrame)

    def create_df(self, data, schema, **kwargs):
        del kwargs  # Unused
        import dask.dataframe as dd

        df = pd.DataFrame(columns=schema, data=data)
        return dd.from_pandas(df)

    def executor(self):

        from liken.backends.dask.executor import DaskExecutor

        return DaskExecutor()

    def wrap(self, df, source_id=None, *, canonical_id: str):
        from liken.backends.dask.wrapper import DaskDF

        return DaskDF(df, source_id, canonical_id=canonical_id)
