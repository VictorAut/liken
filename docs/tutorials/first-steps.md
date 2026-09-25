---
title: First Steps
---

## Installation

See [Installation](../index.md#installation).

## Agent Skills

Using an AI coding agent? See [Agent Skills](../index.md#ai-agent-skills) to install the `liken-skills` bundle and have your agent follow these tutorials' APIs correctly.

??? tip "Using skills"
    You'll get much more efficient use from your agent by using `liken-skills`. **Liken** offers a lot of functionality, and the skills have been prepared to ensure that you match a solution to your use-case as ergonomically as possible.

## Introduction

Code blocks shown in this tutorial assume that a DataFrame, labelled `df`, will be available at runtime. No efforts are made to specify the nature of the data in `df`, the emphasis is on how to set up near deduplication correctly with **Liken**. There are datasets available for experimentation in the [`liken.datasets`](../reference/datasets.md) module for easy access to dummy data.


## Instantiating

A DataFrame must be passed to the top-level `dedupe` function.


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

=== "PySpark"

    ```python
    import liken as lk
    from pyspark.sql import SparkSession

    spark = SparkSession(**kwargs)

    df = spark.read.parquet("...")

    df = (
        lk.dedupe(df, spark_session=spark)
        # ...
    )
    ```

## The Dedupe Lifecycle

Every **Liken** workflow follows the same three beats:

1. **Construct.** `lk.dedupe(df)` wraps your DataFrame. **Liken** detects which backend to use from the DataFrame's type. Nothing has run yet.
2. **Stage.** `.apply(...)` adds dedupers to a collection. You can stage one deduper, a dict of per-column rules, or a pipeline. Nothing has run yet.
3. **Enact.** `.drop_duplicates()` or `.canonicalize()` runs the staged rules, matches records, and returns a new DataFrame. Your original DataFrame is left untouched.

```python
import liken as lk

deduper = lk.dedupe(df)                 # construct
deduper = deduper.apply(lk.fuzzy())     # stage
df = deduper.drop_duplicates("address") # enact
```

Every later tutorial builds on these three beats. Construct, stage, enact.

## The Simplest Example

For the simplest use cases, **Liken** provides familiar-feeling *exact* deduplication, without ceremony:

=== "Single Column"

    ```python

    import liken as lk

    df = lk.dedupe(df).drop_duplicates("address")
    ```

=== "Multiple Columns"

    ```python

    import liken as lk

    df = lk.dedupe(df).drop_duplicates(columns=["address", "email"])
    ```

With no deduper applied, `drop_duplicates` falls back to *exact matching* on the columns you pass: two rows are duplicates when their values are identical. Rows missing a value count as equal too — two rows with no address are duplicates of each other, and one is removed.

However, dataframe records may not be *exactly* repeated:

 id   |  address  |         email
------|-----------|---------------------
  1   |  london   |  fizzpop@yahoo.com
  2   |   tokyo   |  FizzPop@yahoo.com
  3   |   paris   |       a@msn.fr

/// caption
"fizzpop" and "FizzPop" aren't *exactly* the same, but *likely* are.
///

Using `drop_duplicates` straight from pandas won't do anything here, as "fizzpop@yahoo.com" and "FizzPop@yahoo.com" are not the same strings. Neither will the exact matching above: these two emails are distinct strings, so exact matching keeps all three rows. For records like these you need near deduplication.

## Near Deduplication

When things aren't *exactly* the same, you can still deduplicate data. **Liken** is built so that you can focus on defining *what* you want out of a near-deduplication process. The goal will be to be able to define neat and clear-cut ways to deduplicate data with the least amount of code possible. Before looking at how to use dedupers, let's look at what dedupers are available.

## Built-in Dedupers

**Liken** comes with many deduplication methods built-in:

| |               | Deduper                                              | Description                                                                 |
|-------------| ------------- | ----------------------------------------------------- | ---------------------------------------------------------------- |
| *Similarity* |*single-column*| [`exact`](../reference/liken/#liken.exact)       | You've already seen this in use *implicitly* in [The Simplest Example](./first-steps.md#the-simplest-example)  |
| *Similarity* |*single-column*| [`fuzzy`](../reference/liken/#liken.fuzzy)       | Fuzzy string matching                                                                            |
| *Similarity* |*single-column*| [`tfidf`](../reference/liken/#liken.tfidf)       | String token matching with Tf-Idf                                                                      |
| *Similarity* |*single-column*| [`lsh`](../reference/liken/#liken.lsh)           | String token matching with Locality Sensitive Hashing (LSH)                                            |
| *Similarity* |*compound-column*| [`jaccard`](../reference/liken/#liken.jaccard) | Multi column similarity based on intersection of categorical data                                |
| *Similarity* |*compound-column*| [`cosine`](../reference/liken/#liken.cosine)   | Multi column similarity based on dot product of numerical data                                   |
| *Predicate* |*single-column*| [`isna`](../reference/liken/#liken.isna)                | Records where the column value is null/`None`                                        |
| *Predicate* |*single-column*| [`isin`](../reference/liken/#liken.isin)                | Records where the column value is in a list of members                               |
| *Predicate* |*single-column*| [`str_startswith`](../reference/liken/#liken.str_startswith)     | Records where the string starts with a pattern                                       |
| *Predicate* |*single-column*| [`str_endswith`](../reference/liken/#liken.str_endswith)       | Records where the string ends with a pattern                                         |
| *Predicate* |*single-column*| [`str_contains`](../reference/liken/#liken.str_contains)         | Records where the string contains a pattern. Accepts Regex.                          |
| *Predicate* |*single-column*| [`str_len`](../reference/liken/#liken.str_len)              | Records where the string length is bounded by a minimum and maximum length           |

*Single-column* dedupers apply to single columns and are implementation of near string matching. *Compound-column* dedupers are set operations where the values of the set are the values of the columns in a given record. *Similarity* dedupers have a `threshold` argument. *Predicate* dedupers choose an outcome based on a discrete outcome (e.g. is null / not null).

To *use* dedupers, you have to *apply* them, which is covered in the next tutorial.

## Choosing a Threshold

Similarity dedupers compare values pairwise and keep the pairs whose score beats their `threshold`. All thresholds are on a 0–1 scale, and matching is strict: a pair must score *above* the threshold to match. Thresholds differ in what they measure:

- `fuzzy` uses [rapidfuzz](https://rapidfuzz.github.io/RapidFuzz/) string similarity. The default scorer, `simple_ratio`, compares whole strings character by character: `"london"` and `"londn"` score about 0.91. Matching is case-sensitive — `"LONDON"` and `"london"` score 0.0. Other scorers (`token_sort_ratio`, `token_set_ratio`, and more) change how strings are compared, not the scale.
- `tfidf` splits each value into character n-grams (`ngram`, default 3), weights them, and compares the resulting vectors by cosine similarity. It tolerates more drift than `fuzzy`, but its default `topn=2` keeps only the two best candidate matches per row — one of which is the row itself. In dense clusters of near-duplicates, raise `topn`.
- `lsh` builds MinHash signatures over character n-grams and matches approximately: the threshold estimates the Jaccard similarity of the n-gram sets. It scales well to very large data, but a match is probabilistic — pairs near the threshold can be missed.

`explore` shows where your data's duplicate rates actually fall, so you can pick a threshold from evidence rather than guesswork.

## Exploring Your Data

Before choosing a threshold, measure. `explore` reports, for each column you name, the share of records that would be removed under exact matching and under each fuzzy threshold:

```python
import liken as lk

lk.dedupe(df).explore(["address", "email"])
```

        address     email
metric
exact       0.0  0.000000
0.5         0.0  0.333333
0.75        0.0  0.333333
0.9         0.0  0.000000
0.95        0.0  0.000000
0.99        0.0  0.000000

/// caption
Duplicate rates for the dataset above. The two email variants register at 0.5 and 0.75, but not at 0.9.
///

Read it like this:

- The `exact` row is the exact-matching duplicate rate. It is 0.0 for both columns: all three emails are distinct strings, as are the addresses.
- Each numbered row is one fuzzy threshold. The default sweep is 0.5, 0.75, 0.9, 0.95 and 0.99. At 0.5 and 0.75 the two email variants match, so one of three records is a redundant duplicate: a rate of 0.333. At 0.9 they no longer match.
- A rate of 0.333 on 3 records means `drop_duplicates` would remove exactly one record under that rule.

On larger data, sample instead of scanning everything. `frac` takes a fraction in (0, 1]; the default 1.0 uses every row:

```python
lk.dedupe(df).explore(["email"], frac=0.1)
```

By default, each column is profiled with `fuzzy`. To profile a column with a different similarity deduper, pass a dict instead of a list: keys are column labels, values are single-column threshold dedupers:

```python
lk.dedupe(df).explore({"email": lk.tfidf()})
```

`explore` runs on the pandas, polars and modin backends. On dask, ray or pyspark it raises a `ValueError`.

## Missing Values

Real data has nulls. **Liken** does not drop them silently. For single-column rules, a missing value is replaced by the literal string `"na"` before matching, so nulls behave like that value:

| Deduper | What happens to nulls |
| --- | --- |
| `exact` | Nulls match nulls. |
| `fuzzy` | Nulls match nulls — `"na"` vs `"na"` scores a perfect match. |
| `tfidf` | Depends on `ngram`: not matched at the default `ngram=3`, matched at `ngram=1`. |
| `lsh` | Matched: empty signatures bucket together. |
| `isna` | Matches exactly the nulls. The only deduper that sees nulls as nulls. |
| Other predicates | The string `"na"` is tested like any other value. |

Two practical consequences:

- If `"na"` could collide with your data — a real value, or a predicate pattern such as `str_startswith("na")` — handle nulls explicitly with `isna` rules.
- Compound-column dedupers skip the substitution. `jaccard` ignores null values when building each record's set; `cosine` fills numeric NaNs with 0.
