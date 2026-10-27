---
title: "Iterative Workloads"
---

## Reminder

In the [Record Linkage tutorial](../tutorials/record-linkage.md) you found out that Liken creates a `canonical_id`. By default this `canonical_id` is an autoincrementing numeric identifier starting from zero.

In this chapter we explore the configuring needed to canonicalize a dataset iteratively. By iteratively we mean appending to the same dataset — for example a dataset of customers that is appended to with new customers in a given time interval.

!!! note
    Here we explore the implications for batch workloads, especially for datasets that tend to append data

## Canonical IDs

A new canonical ID every time we instantiate a `Dedupe` class isn't going to be practical for our use case. In fact, given our use case, we're likely to already have a canonical ID (literally a Liken `canonical_id`, or another). So we should use that instead and pass it in as a string identifier to the `source_id` argument of the `canonicalize` function. See [the tutorial](../tutorials/record-linkage.md) for a recap.

Note that `source_id` names the *source* column whose values become the canonical ids, while the `canonical_id` argument names the *output* column. If your source column is not already labelled `canonical_id`, pass a different `canonical_id` label — for example `canonicalize(source_id="uid", canonical_id="master_id")`. A `canonical_id` label that already exists in the frame is rejected with a `ValueError`.

## The Problem

Preparing an existing `canonical_id` column for an append is your job, not Liken's: new records need identifiers that continue the existing numbering before they are stacked onto the canonicalized dataset. Liken provides the preprocessors and dedupers; the identifier bookkeeping below is the part you carry out yourself. The suggested steps to take are:

1. Add a column, `canonical_id`, to the append dataset that continues the existing numbering: if the existing dataset holds `N` rows (ids `0` to `N-1`), the `n` appended rows get ids `N` to `N + n - 1`.
2. Append ("stack") your datasets.
3. Instantiate `Dedupe` and pass `source_id="canonical_id"` to the canonicalizer.

!!! warning
    This process is going to be a lot easier with numeric ids. It's possible to use string identifiers but it makes the process of incrementing on append datasets much harder to manage and reason about

## The Append Flow, Worked

Say your canonicalized customers currently hold ids `0` and `1`, and an append batch carries two new records with pre-assigned ids `2` and `3` — one of which is a case-variant of an existing customer:

 canonical_id  |     name
---------------|----------------
       0       |  Alice Wong
       1       |  Bob Lane
       2       |  alice wong
       3       |  Carol Fox

/// caption
Existing records (ids 0–1) stacked with the append batch (ids 2–3).
///

Canonicalize the stacked frame with `source_id="canonical_id"`:

```python
import liken as lk

pipeline = lk.pipeline().step(
    lk.col("name", preprocessors=[lk.preprocessors.lower()]).fuzzy(threshold=0.7)
)

df = lk.dedupe(stacked).apply(pipeline).canonicalize(source_id="canonical_id", keep="first")
```

 canonical_id  |     name
---------------|----------------
       0       |  Alice Wong
       1       |  Bob Lane
       0       |  alice wong
       3       |  Carol Fox

/// caption
The new duplicate inherited the existing id `0` — the cluster keeps its established identity.
///

Because `keep="first"` picks the earliest record of the cluster under the survivorship policy, the new record joins the existing id rather than the other way around. Later appends that duplicate other members of the cluster likewise merge onto id `0`. The dataset keeps one continuous identifier space across runs.

## Decision Tree

``` mermaid
flowchart TD

    df{{"`DataFrame already carries the output label?`"}}
    src1{{"`**source_id** defined in canonicalize()?`"}}
    src2{{"`DataFrame carries a source id column to use?`"}}
    srceqlabel{{"`**source_id** is the same as the output label?`"}}

    reject("`ValueError: choose another output label`")
    existing("`Use existing column as the canonical ids`")
    copycol("`Copy **source_id** into the output label`")
    autoincrement("`Create a new autoincrementing canonical id`")

    df-- yes -->src1
    df-- no -->src2

    src1-- no -->existing
    src1-- yes -->srceqlabel

    srceqlabel-- yes -->reject
    srceqlabel-- no -->existing

    src2-- no -->autoincrement
    src2-- yes -->copycol
```

In other words: the output label must be free. To re-canonicalize a frame that already carries its canonical ids, use the existing column via `source_id` — but only on a frame whose output label slot is free, e.g. by passing a fresh output label such as `canonicalize(source_id="canonical_id", canonical_id="master_id")`.
