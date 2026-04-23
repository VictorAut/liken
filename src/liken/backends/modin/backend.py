from liken.core.backend import Backend
from liken.core.registries import backends_registry


@backends_registry.register("modin")
class ModinBackend(Backend):
    name = "modin"

    def is_match(self, df):
        try:
            import modin.pandas as mpd
        except ImportError:
            return False
        return isinstance(df, mpd.DataFrame)

    def create_df(self, data, schema, **_):
        import modin.pandas as mpd

        return mpd.DataFrame(columns=schema, data=data)

    def executor(self, **_):
        from liken.backends.modin.executor import ModinExecutor

        return ModinExecutor()

    def wrap(self, df, id=None):
        from liken.backends.modin.wrapper import ModinDF

        return ModinDF(df, id)
