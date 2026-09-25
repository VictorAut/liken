---
title: Backends
---

**Liken** runs on six DataFrame backends: pandas, polars, modin, dask, ray and pyspark. You do not choose a backend — **Liken** detects it from the DataFrame you pass to `lk.dedupe`: a pandas DataFrame gets the pandas backend, a `ray.data.Dataset` gets the ray backend. Passing an unsupported type raises `ValueError: Unsupported dataframe type`.

The deduplication API is the same on every backend. What differs is where the work runs and what you get back. This page lays out those differences; the [First Steps](../tutorials/first-steps.md#instantiating) tabs show the instantiation code for each.

## Installing

pandas and polars ship with **Liken** itself. The other backends are optional extras:

| Extra | Installs | For |
| ----- | ----- | ----- |
| *(default)* | pandas, polars | Single-machine work |
| `liken[modin]` | modin (with dask and ray kernels) | Drop-in pandas replacement |
| `liken[dask]` | dask | Partitioned DataFrames |
| `liken[ray]` | ray | Ray datasets |
| `liken[pyspark]` | pyspark | Spark clusters |
| `liken[all]` | all of the above | Everything |

=== "pip"

    ```bash
    pip install 'liken[modin]'
    ```

=== "uv"

    ```bash
    uv pip install 'liken[modin]'
    ```

## Instantiating

Each backend is instantiated by passing its DataFrame to `lk.dedupe` — see the [Instantiating](../tutorials/first-steps.md#instantiating) tabs for all six. One requirement stands out:

PySpark requires a `SparkSession`. Pass it explicitly, or **Liken** raises `ValueError: spark_session arg must be provided for a spark dataframe`:

```python
import liken as lk

df = lk.dedupe(spark_df, spark_session=spark).apply(lk.fuzzy()).drop_duplicates("address")
```

For every other backend the `spark_session` argument is ignored.

## Local vs Distributed

The first split is execution scope:

- **pandas, polars, modin** run over the whole DataFrame in one process. modin is a drop-in for pandas — its DataFrames are pandas-like, and **Liken** treats them locally.
- **dask, ray, pyspark** run over partitions — over batches, on ray. Deduplication executes per partition or batch, on the worker holding it.

The partitioned execution has a consequence you must plan around: **records are only matched within a partition**. Two identical rows in different partitions will not be deduplicated, and each partition's duplicates get ids from that partition's own numbering. Partition your data so that likely duplicates land together, or use `repartition` on the columns your rules match against — see [Use Partitioned Data](performance.md#use-partitioned-data).

## What You Get Back

`collect()` returns the same DataFrame type you passed in, with the work already staged by `drop_duplicates` or `canonicalize`:

| Backend | `collect()` returns |
| ----- | ----- |
| pandas | `pandas.DataFrame` |
| polars | `polars.DataFrame` |
| modin | `modin.pandas.DataFrame` |
| dask | `dask.dataframe.DataFrame` (lazy) |
| ray | `ray.data.Dataset` (lazy) |
| pyspark | `pyspark.sql.DataFrame` (lazy) |

`collect()` itself computes nothing — on the lazy backends the work executes when you trigger it, for example with `.compute()` (dask) or a Spark action.

Three methods pull data to the driver, so they trigger a collect on the distributed backends:

- `canonicalize(id=None)` — computing the auto-increment ids requires the whole frame's row positions.
- `canonicals()` — counts groups by materialising the canonical ids. On PySpark this needs Spark 4 or above, as documented on the method.
- `synthesize()` — builds the synthetic records from collected groups.

Passing an existing id column (`canonicalize(id="uid")`) avoids the auto-increment collect on dask and ray.

## Restrictions

One API difference to know before you write code against a distributed backend:

- `explore` runs on the pandas, polars and modin backends only. On dask, ray or pyspark it raises `ValueError`. Profile a sample locally first, then apply the chosen rules to the full distributed DataFrame.
- Custom dedupers receive a plain Python list built per partition on the worker (see [Data Size and Distributed Backends](../tutorials/custom-dedupers.md#data-size-and-distributed-backends)). Keep the function importable so it can be shipped to workers.

## Which One Do I Pick?

- **pandas** — in-memory frames. The reference backend: everything works, and results are the easiest to inspect.
- **polars** — in-memory frames on a modern, multithreaded execution engine.
- **modin** — pandas code that has outgrown one core. Scales pandas across cores while keeping the pandas API.
- **dask** — partitioned, pandas-like processing on one machine or a small cluster, when data does not fit in memory.
- **ray** — dataset-oriented pipelines with ray already in your stack.
- **pyspark** — data already lives in Spark. Deduplicate where the data is, rather than moving it.

All six accept the same dedupers, dict collections and pipelines. The choice is about where your data lives and how far it must scale, not about which rules you can write.
