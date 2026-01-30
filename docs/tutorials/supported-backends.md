---
title: Supported Backends
---

## Pandas

All the examples we've seen throughout have imagines the use of a Pandas DataFrame. Use **Enlace's** `Dedupe` class as-is for Pandas.

## Polars

No usage difference for Polars:

```python
from enlace import Dedupe
import polars as pl

df = pl.read_csv(...)

dp = Dedupe(df)     # no change
```

## PySpark

**Enlace** supports PySpark DataFrames. This currently means PySpark `DataFrame` instances and *not* PySpark `RDD`s. Deduplication will be restricted to the scope of each partition if you are using a distributed dataset. You will have to pass a Spark Session object to `Dedupe`.

```python
from enlace import Dedupe
from pyspark.sql import SparkSession

SESSION = SparkSession(...)

df = ...

dp = Dedupe(df, spark_session=SESSION)
```

## Recap

!!! success "You learnt:"
    - **Enlace** supports multiple backends.
    - For PySpark backend, you must supply a `SparkSession` object.