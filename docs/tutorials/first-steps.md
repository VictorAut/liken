---
title: First Steps
---


## The Liken Workflow

1. Construct: `lk.dedupe(df)` wraps your DataFrame. Liken detects which backend to use from the DataFrame's type.
2. Apply: `.apply(...)` adds dedupers to a collection. You can apply one deduper, a dict of per-column predicates, or a pipeline.
3. Execute: `.drop_duplicates()` or `.canonicalize()` executes the applied dedupers, matches records, and returns a new DataFrame.

```python
import liken as lk

deduper = lk.dedupe(df)                 # construct
deduper = deduper.apply(lk.fuzzy())     # apply
df = deduper.drop_duplicates("address") # execute
```


## Instantiating

Pass your dataframe to the top-level `dedupe` function.


=== "Pandas"

    ```python
    import liken as lk
    import pandas as pd

    df = pd.read_csv("...")

    df = (
        lk.dedupe(df)
        # ...
    )
    ```

=== "Polars"

    ```python
    import liken as lk
    import polars as pl

    df = pl.read_csv(...)

    df = (
        lk.dedupe(df)
        # ...
    )
    ```

=== "Modin"

    ```python
    import liken as lk
    import modin.pandas as pd

    df = pd.read_csv("...")

    df = (
        lk.dedupe(df)
        # ...
    )
    ```

=== "Dask"

    ```python
    import liken as lk
    import dask.dataframe as dd

    df = dd.read_csv("...")

    df = (
        lk.dedupe(df)
        # ...
    )
    ```

=== "Ray"

    ```python
    import liken as lk
    import ray

    df = ray.data.read_csv("...")

    df = (
        lk.dedupe(df)
        # ...
    )
    ```

=== "PyArrow"

    ```python
    import liken as lk
    import pyarrow as pa
    import pyarrow.parquet as pq

    df = pq.read_table("...")

    df = (
        lk.dedupe(df)
        # ...
    )
    ```

=== "PySpark"

    ```python
    import liken as lk
    from pyspark.sql import SparkSession

    spark = SparkSession(**kwargs)

    df = spark.read.parquet("...")

    # the spark session is inferred from the dataframe
    df = (
        lk.dedupe(df)
        # ...
    )
    ```

## Duplicate Rate Profiling

Before choosing a threshold, measure the variance @ threshold for any column. `explore` reports, for each column you name, the share of records that would be removed under exact matching and under each fuzzy threshold:

```python
import liken as lk

lk.dedupe(df).explore(["address", "email"])
```

metric | address | email |
|--- | --- | --- |
|exact | 0.0  | 0.000000 |
|0.5   | 0.0  | 0.333333 |
|0.75  | 0.0  | 0.333333 |
|0.9   | 0.0  | 0.000000 |
|0.95  | 0.0  | 0.000000 |
|0.99  | 0.0  | 0.000000 |

/// caption
Output: Duplicate rates for the a two-column dataset. Neither address or email has exact matches. But at thresholds 0.5 and 0.75, 1/3 of the emails are duplicates, at higher thresholds, no emails are considered duplicates.
///

In the above:

- The `exact` row is the exact-matching duplicate rate. It is 0.0 for both columns: all three emails are distinct strings, as are the addresses.
- Each numbered row is one fuzzy threshold. The default sweep is 0.5, 0.75, 0.9, 0.95 and 0.99. At 0.5 and 0.75 the two email variants match, so one of three records is a redundant duplicate: a rate of 0.333. At 0.9 they no longer match.
- A rate of 0.333 on 3 records means `drop_duplicates` would remove exactly one record under that threshold.

On larger data, use a sample instead of scanning everything, for a faster exploration. `frac` takes a fraction where the default of 1.0 uses every row:

```python
lk.dedupe(df).explore(["email"], frac=0.1)
```

By default, each column is profiled with [`fuzzy`](../reference/liken.md#liken.fuzzy). To profile a column with a different threshold deduper, pass a dict instead of a list: keys are column labels, values are single-column threshold dedupers:

```python
lk.dedupe(df).explore({"email": lk.tfidf()})
```

!!! warning

    `explore` runs on the pandas, polars, modin and pyarrow backends. On dask, ray or pyspark, `explore` is not supported and raises a `ValueError`.
