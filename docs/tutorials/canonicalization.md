---
title: Canonicalization
---

??? info "TLDR?"

    Swap [`.drop_duplicates()`](../reference/liken.md#liken.Dedupe.drop_duplicates) with [`.canonicalize()`](../reference/liken.md#liken.Dedupe.canonicalize) **and** [`collect()`](../reference/liken.md#liken.Dedupe.collect) the results:

    ```python
    import liken as lk

    df = ...

    df = (
        lk.dedupe(df)
        .apply(...)
        .canonicalize(...)
        .collect()
    )
    ```

## Introduction

Up to now you've learnt how to use dedupers with `apply`, specifically within the context of *dropping duplicates*.

But, what if you want to retain your duplicate instances? And instead simply label them as such?

Liken supports **canonicalization**, where the deduplication process you are doing is not to drop data from your DataFrame, but rather to *link* it together. So, a deduper that defines a fuzzy string deduplication of an `address` column will *label* the duplicates as duplicates rather than *dropping* them. Records are instead *canonicalized*.

Linkage is provided by a **canonical** record, which in Liken is identified by the auto-generated `canonical_id` column.

!!! info "Entity Resolution"

    Retaining records as *known* duplicates instead of *dropping* duplicates is known as canonicalization in Liken. This is also known as **Entity Resolution**, in other literature. See the [Glossary](../about/glossary.md).


## Canonicalization

Let's look at a dummy dataset, `df`:


  uid   |  address  |          email
--------|-----------|-----------------------
  a001  |  london   |   fizzpop@yahoo.com
  a002  |   tokyo   |  fizzpop@yahoo.co.uk
  a003  |   paris   |        a@msn.fr

/// caption
Two very clearly similar emails exist.
///


We're going to aim to **link** the above email addresses. To do so, swap `.drop_duplicates` with `.canonicalize`, and collect the results:

```python
import liken as lk

df = (
    lk.dedupe(df)
    .apply(lk.fuzzy(threshold=0.85))
    .canonicalize(
        "email",
        keep="first",
    )
    .collect() # (1)!
)
```

1.  `canonicalize` does not return the dataframe, unlike `drop_duplicates`. It needs to be collected first!

The `keep` argument picks the *representative* of each duplicate cluster under the survivorship policy: `"first"` keeps the earliest record of the cluster in the dataframe's current row order, `"last"` the latest. The representative's identity is what the cluster's `canonical_id` will carry.

Now, `df` looks the same, with an extra `canonical_id` column:


  uid   |  address  |          email        |  canonical_id
--------|-----------|-----------------------|----------------
  a001  |  london   |   fizzpop@yahoo.com   |        0
  a002  |   tokyo   |  fizzpop@yahoo.co.uk  |        0
  a003  |   paris   |        a@msn.fr       |        2

/// caption
The two email addresses are linked to the canonical record "0".
///

`.canonicalize` creates a new `canonical_id` field. Any repeated `canonical_id` is a duplicate. In this instance that was an auto-incrementing numeric field. As such, the repeated `canonical_id` represents the index position in the DataFrame of the *canonical* record.

With `keep="last"` the cluster would be represented by its latest record instead, and both rows would carry `a002`'s position as their id.

## Canonical ID

By default, Liken created a new, autoincrementing ID field, "canonical_id". Control the generated label:

```python
.canonicalize(canonical_id="my_canonical_id")
```

You can override the auto-incrementing mechanism such that it copies from another ID field in the data. Control this behaviour by passing an explicit label to the `source_id` argument of `.canonicalize`. The values of the column you name become the `canonical_id` values: name `canonical_id` itself and it is left as is; name any other column and `canonical_id` is written from its values. For example:

```python
import liken as lk

df = (
    lk.dedupe(df)
    .apply(lk.fuzzy(threshold=0.85))
    .canonicalize(
        "email",
        keep="first",
        source_id="uid", # `source_id` arg included
    )
    .collect()
)
```

Now, checkout the variation in the output of `df`:


  uid   |  address  |          email        |  canonical_id
--------|-----------|-----------------------|----------------
  a001  |  london   |   fizzpop@yahoo.com   |      a001
  a002  |   tokyo   |  fizzpop@yahoo.co.uk  |      a001
  a003  |   paris   |        a@msn.fr       |      a003

/// caption
Canonical records are no longer identified by index position in the DataFrame, but instead based on a pre-existing (unique) identifier.
///


!!! info

    Checkout the guide on [Iterative Workloads](../in-practice/iterative-workloads.md) for more information on how to modify the output of the canonical ID in accordance to your potential problem scenarios.


## Inspecting Clusters

Once `canonicalize` has run, [`canonicals`](../reference/liken.md#liken.Dedupe.canonicals) answers the natural next question: *which clusters actually hold duplicates?* It returns a dict of `canonical_id` to record count, for every cluster with at least `n` records — the default `n` is 2:

```python
import liken as lk

result = (
    lk.dedupe(df)
    .apply(lk.fuzzy(threshold=0.85))
    .canonicalize(
        "email",
        keep="first",
        source_id="uid",
    )
)

result.canonicals(n=5) # default 2, but can be increased.
```

!!! warning
    `canonicals` collects the canonical ids to your machine, so on distributed backends (Ray, Dask, PySpark) it triggers a collect.
