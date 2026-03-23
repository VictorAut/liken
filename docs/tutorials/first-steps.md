---
title: First Steps
---

## Installation

```shell
pip install liken
```

## Introduction

**Liken** is a deduplication library for DataFrames, so in this tutorial we will assume that you have a basic awareness of at least one DataFrame library in Python. We'll steer clear of too much syntax outside of **Liken** — but our reference point will be [Pandas](https://pandas.pydata.org/) DataFrames.

Code blocks shown in this tutorial assume that a DataFrame, labelled `df`, will be available runtime. No efforts are made to specify the nature of the data in `df`, rather the emphasis is on how to set up near deduplication correctly.

## The Simplest Example

The Pandas DataFrame library provides deduplication facilities with it's DataFrame method [`.drop_duplicates`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop_duplicates.html). It's a widely used method and **Liken** borrows its syntax. If you've used this function before then you know that it does *exact* deduplication, which is to say it will keep only one instance of an *exactly* repeated record. 

As mentioned above, **Liken** recycles the `drop_duplicate` syntax:

```python

import liken as lk

df = dedupe(df).drop_duplicates("address").collect()
```

Above, we drop duplicates on an "address" column. We achieved this by virtue of the `Dedupe` class. We can also deduplicate multiple columns at once:

```python

import liken as lk

df = dedupe(df).drop_duplicates(columns=["address", "email"]).collect()
```

This example does what Pandas can do, at the cost of an additional line of code. However, things get sticky when your records aren't quite *exactly* the same. Look at the following instances of emails in this dummy dataset:


 id   |  address  |         email       
------|-----------|---------------------
  1   |  london   |  fizzpop@yahoo.com  
  2   |   tokyo   |  FizzPop@yahoo.com  
  3   |   paris   |       a@msn.fr      

/// caption
Unfortunately, "fizzpop" and "FizzPop" just aren't the same...
///

As you can see, this dummy dataset containing 3 unique emails. Using `drop_duplicates` straight from pandas won't do anything here, as "fizzpop@yahoo.com" and "FizzPop@yahoo.com" are not the same strings. In this particular case, some light preprocessing could deal with the issue allowing you to proceed with some useful deduplication. However, that's not always possible — or at least not easily achievable.

## Near Deduplication

When things aren't *exactly* the same, you can still deduplicate data. **Liken** is built so that you can focus on defining *what* you want out of a near-deduplication process. The goal will be to be able to define neat and clear-cut ways to deduplicate data with the least amount of code possible. Before looking at how to use dedupers, let's look at what dedupers are available.

## Built-in Dedupers

**Liken** comes with many deduplication methods built-in:

| |               | Strategy                                              | Description                                                                 |
|-------------| ------------- | ----------------------------------------------------- | ---------------------------------------------------------------- |
| *Similarity* |*single-column*| [`exact`](../../reference/liken.md/#liken.exact)       | You've already seen this in use *implicitely* in your [First Steps](../first-steps.md)  |
| *Similarity* |*single-column*| [`fuzzy`](../../reference/liken.md/#liken.fuzzy)       | Fuzzy string matching                                                                            |
| *Similarity* |*single-column*| [`tfidf`](../../reference/liken.md/#liken.tfidf)       | String matching with Tf-Idf                                                                      |
| *Similarity* |*single-column*| [`lsh`](../../reference/liken.md/#liken.lsh)           | String matching with Locality Sensitive Hashing (LSH)                                            |
| *Similarity* |*compound-column*| [`jaccard`](../../reference/liken.md/#liken.jaccard) | Multi column similarity based on intersection of categorical data                                |
| *Similarity* |*compound-column*| [`cosine`](../../reference/liken.md/#liken.cosine)   | Multi column similarity based on dot product of numerical data                                   |
| *Predicate* |*single-column*| [`isna`](../../reference/liken.md/#liken.isna)                | Records where the column value is null/`None`                                        |
| *Predicate* |*single-column*| [`isin`](../../reference/liken.md/#liken.isin)                | Records where the column value is in a list of members                               |
| *Predicate* |*single-column*| [`str_startswith`](../../reference/liken.md/#liken.str_startswith)     | Records where the string starts with a pattern                                       |
| *Predicate* |*single-column*| [`str_endswith`](../../reference/liken.md/#liken.str_endswith)       | Records where the string ends with a pattern                                         |
| *Predicate* |*single-column*| [`str_contains`](../../reference/liken.md/#liken.str_contains)         | Records where the string contains a pattern. Accepts Regex.                          |
| *Predicate* |*single-column*| [`str_len`](../../reference/liken.md/#liken.str_len)              | Records where the string length is bounded by a minimum and maximum length           |

*single-column* strategies apply to single columns and are implementation of near string matching. *compound-column* strategies are set operations where the values of the set are the values of the columns in a given record. *Similarity* dedupers have a `threshold` argument *predicate* dedupers choose an outcome based on a discrete outcome (e.g. is null / not null)

## Recap

!!! success "You learnt:"
    - **Liken** has an easy to use `Dedupe` class.
    - `Dedupe` let's you use a `drop_duplicates` method.
    - For near deduplication processes **Liken** refers to deduplication **strategies**.