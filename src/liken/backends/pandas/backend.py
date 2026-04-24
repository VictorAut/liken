from typing import final

import pandas as pd

from liken.backends.pandas.executor import PandasExecutor
from liken.backends.pandas.wrapper import PandasDF
from liken.core.backend import Backend
from liken.core.registries import backends_registry


@final
@backends_registry.register("pandas")
class PandasBackend(Backend):
    name = "pandas"

    def is_match(self, df):

        return isinstance(df, pd.DataFrame)

    def create_df(self, data, schema):
        return pd.DataFrame(columns=schema, data=data)

    def executor(self):

        return PandasExecutor()

    def wrap(self, df, id=None):
        return PandasDF(df, id)
