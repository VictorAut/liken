import catalogue


NAMESPACE = "liken"

dedupers_registry = catalogue.create(NAMESPACE, "dedupers")
backends_registry = catalogue.create(NAMESPACE, "backends", entry_points=True)
