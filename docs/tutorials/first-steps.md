---
title: First Steps
---

## Installation

```shell
pip install liken
```

## Introduction

**Liken** is a deduplication library for DataFrames. Code blocks shown in this tutorial assume that a DataFrame, labelled `df`, will be available runtime. No efforts are made to specify the nature of the data in `df`, rather the emphasis is on how to set up near deduplication correctly. There is synthetic data available in the [`liken.synthetic`](../reference/synthetic.md) module for easy access to dummy data.

## Instantiating

A DataFrame must be passed to the top-level `dedupe` function. **Liken** currently supports three backends: Pandas, Polars and PySpark.


=== "Pandas"

    ```python
    import liken as lk
    import pandas as pd

    df = pd.read_csv("...")

    df = (
        lk.dedupe(df)
        # .apply(...)
        # ...
        .collect()
    )
    ```

=== "Polars"

    ```python
    import liken as lk
    import polars as pl

    df = pl.read_csv(...)

    df = (
        lk.dedupe(df)
        # .apply(...)
        # ...
        .collect()
    )
    ```

=== "PySpark"

    ```python
    import liken as lk
    from pyspark.sql import SparkSession

    spark = SparkSession(**kwargs)

    df = spark.read.parquet("...")

    df = (
        lk.dedupe(df, spark_session=spark)
        # .apply(...)
        # ...
        .collect()
    )
    ```

## The Simplest Example

Drop *exact* duplicates on a single column:

```python

import liken as lk

df = dedupe(df).drop_duplicates("address").collect()
```

Drop *exact* duplicates multiple columns:

```python

import liken as lk

df = dedupe(df).drop_duplicates(columns=["address", "email"]).collect()
```

Things get complicated when your dataframe records aren't quite *exactly* the same. Look at the following instances of emails in this dummy dataset:


 id   |  address  |         email       
------|-----------|---------------------
  1   |  london   |  fizzpop@yahoo.com  
  2   |   tokyo   |  FizzPop@yahoo.com  
  3   |   paris   |       a@msn.fr      

/// caption
"fizzpop" and "FizzPop" just aren't the same...
///

As you can see, this dummy dataset containing 3 unique emails. Using `drop_duplicates` straight from pandas won't do anything here, as "fizzpop@yahoo.com" and "FizzPop@yahoo.com" are not the same strings. In this particular case, some light preprocessing could deal with the issue allowing you to proceed with some useful deduplication. However, that's not always possible — or at least not easily achievable.

## Near Deduplication

When things aren't *exactly* the same, you can still deduplicate data. **Liken** is built so that you can focus on defining *what* you want out of a near-deduplication process. The goal will be to be able to define neat and clear-cut ways to deduplicate data with the least amount of code possible. Before looking at how to use dedupers, let's look at what dedupers are available.

## Built-in Dedupers

**Liken** comes with many deduplication methods built-in:

| |               | Deduper                                              | Description                                                                 |
|-------------| ------------- | ----------------------------------------------------- | ---------------------------------------------------------------- |
| *Similarity* |*single-column*| [`exact`](../reference/liken.md/#liken.exact)       | You've already seen this in use *implicitely* in [The Simplest Example](../tutorials/first-steps.md#the-simplest-example)  |
| *Similarity* |*single-column*| [`fuzzy`](../reference/liken.md/#liken.fuzzy)       | Fuzzy string matching                                                                            |
| *Similarity* |*single-column*| [`tfidf`](../reference/liken.md/#liken.tfidf)       | String matching with Tf-Idf                                                                      |
| *Similarity* |*single-column*| [`lsh`](../reference/liken.md/#liken.lsh)           | String matching with Locality Sensitive Hashing (LSH)                                            |
| *Similarity* |*compound-column*| [`jaccard`](../reference/liken.md/#liken.jaccard) | Multi column similarity based on intersection of categorical data                                |
| *Similarity* |*compound-column*| [`cosine`](../reference/liken.md/#liken.cosine)   | Multi column similarity based on dot product of numerical data                                   |
| *Predicate* |*single-column*| [`isna`](../reference/liken.md/#liken.isna)                | Records where the column value is null/`None`                                        |
| *Predicate* |*single-column*| [`isin`](../reference/liken.md/#liken.isin)                | Records where the column value is in a list of members                               |
| *Predicate* |*single-column*| [`str_startswith`](../reference/liken.md/#liken.str_startswith)     | Records where the string starts with a pattern                                       |
| *Predicate* |*single-column*| [`str_endswith`](../reference/liken.md/#liken.str_endswith)       | Records where the string ends with a pattern                                         |
| *Predicate* |*single-column*| [`str_contains`](../reference/liken.md/#liken.str_contains)         | Records where the string contains a pattern. Accepts Regex.                          |
| *Predicate* |*single-column*| [`str_len`](../reference/liken.md/#liken.str_len)              | Records where the string length is bounded by a minimum and maximum length           |

*Single-column* dedupers apply to single columns and are implementation of near string matching. *Compound-column* dedupers are set operations where the values of the set are the values of the columns in a given record. *Similarity* dedupers have a `threshold` argument. *Predicate* dedupers choose an outcome based on a discrete outcome (e.g. is null / not null)
