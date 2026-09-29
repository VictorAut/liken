from typing import final

import pyarrow as pa

from liken.backends.pyarrow.executor import PyarrowExecutor
from liken.backends.pyarrow.wrapper import ArrowDF
from liken.core.backend import Backend
from liken.core.registries import backends_registry


@final
@backends_registry.register("pyarrow")
class PyarrowBackend(Backend):
    name = "pyarrow"

    def is_match(self, df):

        return isinstance(df, pa.Table)

    def create_df(self, data, schema, **kwargs):
        del kwargs  # Unused
        if not data:
            return pa.table({name: pa.array([], type=pa.null()) for name in schema})
        return pa.Table.from_pylist([dict(zip(schema, row)) for row in data])

    def executor(self, **kwargs):
        del kwargs  # Unused
        return PyarrowExecutor()

    def wrap(self, df, id=None):
        return ArrowDF(df, id)
