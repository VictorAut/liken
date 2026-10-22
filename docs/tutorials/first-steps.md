---
title: First Steps
---


## Introduction

Code blocks shown in this tutorial assume that a DataFrame, labelled `df`, will be available at runtime.

Rarely are efforts are made to specify the nature of the data in `df`. Instead, the emphasis is on how to set up near deduplication correctly with Liken.

There are datasets available for experimentation in the [`liken.datasets`](../reference/datasets.md) module for easy access to dummy data.


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

## The Dedupe Lifecycle

Every Liken workflow follows the same formula:

1. **Construct.** `lk.dedupe(df)` wraps your DataFrame. Liken detects which backend to use from the DataFrame's type.
2. **Stage.** `.apply(...)` adds dedupers to a collection. You can stage one deduper, a dict of per-column rules, or a pipeline.
3. **Enact.** `.drop_duplicates()` or `.canonicalize()` runs the staged rules, matches records, and returns a new DataFrame.

```python
import liken as lk

deduper = lk.dedupe(df)                 # construct
deduper = deduper.apply(lk.fuzzy())     # stage
df = deduper.drop_duplicates("address") # enact
```

Every later tutorial builds on this formula.

## The Simplest Example

For the simplest use cases, Liken provides familiar-feeling *exact* deduplication, without ceremony:

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

When things aren't *exactly* the same, you can still deduplicate data. Liken is built so that you can focus on defining *what* you want out of a near-deduplication process. The goal will be to be able to define neat and clear-cut ways to deduplicate data with the least amount of code possible. Before looking at how to use dedupers, let's look at what dedupers are available.

## Built-in Dedupers

Liken comes with many deduplication methods built-in:

| |               | Deduper                                              | Description                                                                 | Complexity |
|-------------| ------------- | ----------------------------------------------------- | ---------------------------------------------------------------- | -------- |
| *Similarity* |*single-column*| [`exact`](../reference/liken/#liken.exact)       | You've already seen this in use *implicitly* in [The Simplest Example](./first-steps.md#the-simplest-example).  | *O(n)* |
| *Similarity* |*single-column*| [`fuzzy`](../reference/liken/#liken.fuzzy)       | Fuzzy string matching.                                                                            | *O(n²)* |
| *Similarity* |*single-column*| [`edit_distance`](../reference/liken/#liken.edit_distance) | Values within a maximum edit (Levenshtein) distance. | *O(n²)* |
| *Similarity* |*single-column*| [`within`](../reference/liken/#liken.within)     | Values within an absolute numeric tolerance. | *O(n log n)* |
| *Similarity* |*single-column*| [`tfidf`](../reference/liken/#liken.tfidf)       | String token matching with Tf-Idf. | *~O(n·c)* |
| *Similarity* |*single-column*| [`lsh`](../reference/liken/#liken.lsh)           | String token matching with Locality Sensitive Hashing (LSH). | *O(n)* |
| *Similarity* |*single-column*| [`sorted_neighbourhood`](../reference/liken/#liken.sorted_neighbourhood) | Fuzzy string matching inside a window of sorted-order neighbours. | *O(n log n)* |
| *Similarity* |*compound-column*| [`jaccard`](../reference/liken/#liken.jaccard) | Multi column similarity based on intersection of categorical data.                                | *O(n²)* |
| *Similarity* |*compound-column*| [`cosine`](../reference/liken/#liken.cosine)   | Multi column similarity based on dot product of numerical data.                                   | *O(n²)* |
| *Similarity* |*compound-column*| [`token_overlap`](../reference/liken/#liken.token_overlap) | Multi column set containment: a record whose values are a subset of another's matches.        | *O(n²)* |
| *Similarity* |*compound-column*| [`knn`](../reference/liken/#liken.knn) | Multi column k nearest neighbours on numerical data: each row matches its k best under a similarity threshold.        | *O(n log n)* |
| *Predicate* |*single-column*| [`isna`](../reference/liken/#liken.isna)                | Records where the column value is null/`None`.                                        | *O(n)* |
| *Predicate* |*single-column*| [`isin`](../reference/liken/#liken.isin)                | Records where the column value is in a list of members.                               | *O(n)* |
| *Predicate* |*single-column*| [`str_startswith`](../reference/liken/#liken.str_startswith)     | Records where the string starts with a pattern. | *O(n)* |
| *Predicate* |*single-column*| [`str_endswith`](../reference/liken/#liken.str_endswith)       | Records where the string ends with a pattern. | *O(n)* |
| *Predicate* |*single-column*| [`str_contains`](../reference/liken/#liken.str_contains)         | Records where the string contains a pattern. Accepts Regex. | *O(n)* |
| *Predicate* |*single-column*| [`str_len`](../reference/liken/#liken.str_len) | Records where the string length is bounded by a minimum and maximum length. | *O(n)* |
| *Predicate* |*single-column*| [`num_range`](../reference/liken/#liken.num_range) | Records where the numeric value falls inside the inclusive interval. | *O(n)* |
| *Predicate* |*single-column*| [`date_range`](../reference/liken/#liken.date_range) | Records whose dates, parsed under an explicit format, are at most `window` days apart. | *O(n)* |
| *Predicate* |*single-column*| [`sorted_tokens`](../reference/liken/#liken.sorted_tokens) | Records whose whitespace-separated words are the same, in any order. | *O(n)* |

*Single-column* dedupers apply to single columns and are implementation of near string matching. *Compound-column* dedupers are set operations where the values of the set are the values of the columns in a given record. *Similarity* dedupers have a `threshold` argument. *Predicate* dedupers choose an outcome based on a discrete outcome (e.g. is null / not null).

To *use* dedupers, you have to *apply* them, which is covered in the next tutorial.

### Choosing a Deduper

One route from your data problem to a built-in deduper. Start at the top and answer for the column you want to deduplicate:

- *Teal* dedupers are similarity-based dedupers
- *Orange* dedupers are rule-predicate dedupers

``` mermaid
flowchart TD

    classDef predicate fill:#fde68a,stroke:#b45309,color:#78350f
    classDef similarity fill:#99f6e4,stroke:#0d9488,color:#134e4a

    q{{"`**What are you deduplicating?**`"}}

    single(["`**One column**`"])
    sstr(["`**String**`"])
    num(["`**Numeric**`"])
    date(["`**Date**`"])
    comp(["`**Several Columns**`"])

    q --> single
    q --> comp

    single -- Rule predication on membership or null checks on any type --> anyrule
    anyrule["`**isin** <br/>**isna**`"]

    single --> sstr
    single --> date
    single --> num

    sstr -- Exact copies --> sexact
    sexact["`**exact**`"]
    sstr -- Same words, in any order --> stok
    stok["`**sorted_tokens**`"]
    sstr -- Rule predication on strings --> srule
    srule["`**str_startswith** <br/>**str_endswith** <br/>**str_contains** <br/>**str_len**`"]
    sstr --Typos or small edits --> sfuzzy
    sfuzzy["`**fuzzy** <br/>**edit_distance** <br/>**sorted_neighbourhood**`"]
    sstr -- Large columns, token similarity --> stfidf
    stfidf["`**lsh** <br/>**tfidf**`"]


    date -- Date in range --> ndate
    ndate["`**date_range**`"]

    num -- Within an absolute tolerance --> nwithin
    num -- Value in range --> nrange
    nwithin["`**within**`"]
    nrange["`**num_range**`"]

    comp -- Categorical sets --> cjac
    cjac["`**jaccard** <br/>**token_overlap**`"]
    comp -- Numeric vectors --> cvec
    cvec["`**cosine** <br/>**knn**`"]

    nrange ~~~ sstr

    class cjac,cvec,sed,nwithin,sexact,stok,sfuzzy,stfidf,ssn similarity
    class anyrule,nrange,ndate,srule predicate
```

test

``` mermaid
flowchart TD

    classDef predicate fill:#fde68a,stroke:#b45309,color:#78350f
    classDef similarity fill:#99f6e4,stroke:#0d9488,color:#134e4a

    q{{"`**What are you deduplicating?**`"}}

    rules(["`**Values predicate**`"])
    values(["`**Values differ**`"])
    single(["`**One column**`"])
    sstr(["`**String**`"])
    num(["`**Numeric**`"])
    comp(["`**Several Columns**`"])

    q --> rules
    q --> values
    values --> single
    values --> comp

    rules --> allrules
    allrules["`
        **isin** *(any type; single col)*
        **isna** *(any type; single col)*
        **str_startswith** *(str type; single col)*
        **str_endswith** *(str type; single col)*
        **str_contains** *(str type; single col)*
        **str_len** *(str type; single col)*
        **date_range** *(date type; single col)*
        **num_range** *(num type; single col)*
        `"]

    single --> sstr
    single --> num

    sstr -- Exact copies --> sexact
    sexact["`**exact**`"]
    sstr -- Same words, in any order --> stok
    stok["`**sorted_tokens**`"]
    sstr --Typos or small edits --> sfuzzy
    sfuzzy["`**fuzzy** <br/>**edit_distance** <br/>**sorted_neighbourhood**`"]
    sstr -- Large columns, token similarity --> stfidf
    stfidf["`**lsh** <br/>**tfidf**`"]


    num -- Within an absolute tolerance --> nwithin
    nwithin["`**within**`"]

    comp -- Categorical sets --> cjac
    cjac["`**jaccard** <br/>**token_overlap**`"]
    comp -- Numeric vectors --> cvec
    cvec["`**cosine** <br/>**knn**`"]

    class cjac,cvec,sed,nwithin,sexact,stok,sfuzzy,stfidf,ssn similarity
    class allrules predicate
```

Every rule-based deduper above — `isin`, `isna`, the `str_*` predicates, `num_range` and `date_range` — scans the column once, O(n). Reference pages: [`exact`](../reference/liken/#liken.exact), [`fuzzy`](../reference/liken/#liken.fuzzy), [`edit_distance`](../reference/liken/#liken.edit_distance), [`sorted_neighbourhood`](../reference/liken/#liken.sorted_neighbourhood), [`within`](../reference/liken/#liken.within), [`tfidf`](../reference/liken/#liken.tfidf), [`lsh`](../reference/liken/#liken.lsh), [`sorted_tokens`](../reference/liken/#liken.sorted_tokens), [`jaccard`](../reference/liken/#liken.jaccard), [`token_overlap`](../reference/liken/#liken.token_overlap), [`cosine`](../reference/liken/#liken.cosine), [`knn`](../reference/liken/#liken.knn), [`isin`](../reference/liken/#liken.isin), [`isna`](../reference/liken/#liken.isna), [`str_startswith`](../reference/liken/#liken.str_startswith), [`str_endswith`](../reference/liken/#liken.str_endswith), [`str_contains`](../reference/liken/#liken.str_contains), [`str_len`](../reference/liken/#liken.str_len), [`num_range`](../reference/liken/#liken.num_range), [`date_range`](../reference/liken/#liken.date_range). For wording a tree cannot route on, write a [custom deduper](./custom-dedupers.md). The taxonomy table under [Built-in Dedupers](#built-in-dedupers) describes each deduper in one line.

### Choosing a Threshold

Similarity dedupers compare values pairwise and keep the pairs whose score meets their `threshold`. All thresholds are on a 0–1 scale, and matching is inclusive: a pair scoring exactly *at* the threshold matches. At the extreme, `threshold=0` matches every pair, including pairs with no similarity at all. Thresholds differ in what they measure:

- `fuzzy` uses [rapidfuzz](https://rapidfuzz.github.io/RapidFuzz/) string similarity. The default scorer, `simple_ratio`, compares whole strings character by character: `"london"` and `"londn"` score about 0.91. Matching is case-sensitive: `"LONDON"` and `"london"` score 0.0. Other scorers (`token_sort_ratio`, `token_set_ratio`, and more) change how strings are compared, not the scale. With `topn=k`, `fuzzy` becomes approximate: each value keeps only its `k` best candidate matches (a pair matches when either side holds the other), which bounds the scores held per value, but a cluster of more than `k` near-identical records may not link.
- `tfidf` splits each value into character n-grams (`ngram`, default 3), weights them, and compares the resulting vectors by cosine similarity. It tolerates more drift than `fuzzy`, but its default `topn=2` keeps only the two best candidate matches per row, one of which is the row itself. In dense clusters of near-duplicates, raise `topn`.
- `lsh` builds MinHash signatures over character n-grams and matches approximately: the threshold estimates the Jaccard similarity of the n-gram sets. It scales well to very large data, but a match is probabilistic — pairs near the threshold can be missed.
- `sorted_neighbourhood` sorts the column by value and scores each row only against the next `window - 1` rows in sorted order, so scorer work falls from O(n²) to O(n·window). The match set is a subset of all-pairs `fuzzy` at the same threshold and scorer. Sorting is on raw values: similar values differing near the start ("OL5 9PL" vs "0L5 9PL") can fall outside each other's windows, and word-order variants are `sorted_tokens`' job. `fuzzy(topn=k)` is the different trade-off: k best partners anywhere in the column, still all-pairs scoring.

`explore` shows where your data's duplicate rates actually fall, so you can pick a threshold from evidence rather than guesswork.

## Exploring Your Data

Before choosing a threshold, measure. `explore` reports, for each column you name, the share of records that would be removed under exact matching and under each fuzzy threshold:

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
Duplicate rates for the [dataset](#the-simplest-example) above. Email variants registers at 0.5 and 0.75, but not at >0.9.
///

Which can read as:

- The `exact` row is the exact-matching duplicate rate. It is 0.0 for both columns: all three emails are distinct strings, as are the addresses.
- Each numbered row is one fuzzy threshold. The default sweep is 0.5, 0.75, 0.9, 0.95 and 0.99. At 0.5 and 0.75 the two email variants match, so one of three records is a redundant duplicate: a rate of 0.333. At 0.9 they no longer match.
- A rate of 0.333 on 3 records means `drop_duplicates` would remove exactly one record under that rule.

On larger data, use a sample instead of scanning everything, for a faster exploration. `frac` takes a fraction where the default of 1.0 uses every row:

```python
lk.dedupe(df).explore(["email"], frac=0.1)
```

By default, each column is profiled with `fuzzy`. To profile a column with a different similarity deduper, pass a dict instead of a list: keys are column labels, values are single-column threshold dedupers:

```python
lk.dedupe(df).explore({"email": lk.tfidf()})
```

`explore` runs on the pandas, polars, modin and pyarrow backends. On dask, ray or pyspark, `explore` is not supported and raises a `ValueError`.

## Missing Values

Real data has nulls. Liken does not drop null values and treats a missing value as `None` or any `NaN`. For the
similarity dedupers, missing values group with each other:

| Deduper | What happens to missing values |
| --- | --- |
| `exact` | Group with each other|
| `fuzzy` | Group with each other|
| `edit_distance` | Group with each other|
| `within` | Group with each other|
| `tfidf` | Group with each other at any given `ngram`; the vectoriser receives non-missing values only. |
| `lsh` | Same as `tfidf`. |
| `isna` | Matches exactly the missing values; `~isna` matches exactly the non-missing values. |
| `num_range` | Never matches: a missing value is outside any interval, positive or negated (`~num_range`). |
| `date_range` | Never matches: a missing value parses to nothing, positive or negated (`~date_range`). |
| `sorted_tokens` | Never matches: a missing value has no words to sort, positive or negated (`~sorted_tokens`). |
| Other predicates | A missing value never satisfies the predicate, on either polarity — except positive `isin`, where a missing value matches iff `None` is listed in `values`. |

Two practical consequences are worth noting:

- Missing handling is backend-independent: the pandas-family backends convert NaN to null on the way into Arrow, while polars and pyarrow keep float NaN. Both are one missing class.
- `exact` on compound columns collapses missing members: `(None, "x")` and `(NaN, "x")` share one key. `jaccard` and `token_overlap` exclude `None` values when building each record's set, so an all-null record matches nothing there; `cosine` fills numeric NaNs with 0, and `knn` does the same before finding neighbours — an entirely missing row becomes a zero vector and matches nothing.
- The empty string is a value, not a missing one. `str_len` never matches it, whatever the bounds; under `~str_len` it matches, like every non-matching value.
