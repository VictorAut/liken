from typing import final

import pandas as pd

from liken.core.backend import Backend
from liken.core.registries import backends_registry


@final
@backends_registry.register("ray")
class RayBackend(Backend):
    name = "ray"
    supports_explore = False

    def is_match(self, df):
        try:
            import ray
        except Exception:
            return False
        return isinstance(df, ray.data.Dataset)

    def create_df(self, data, schema, **kwargs):
        del kwargs  # Unused
        import ray

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        df = pd.DataFrame(columns=schema, data=data)

        return ray.data.from_pandas(df)

    def executor(self, **kwargs):
        del kwargs  # Unused

        from liken.backends.ray.executor import RayExecutor

        return RayExecutor()

    def wrap(self, df, source_id=None):
        from liken.backends.ray.wrapper import RayDF

        return RayDF(df, source_id)
