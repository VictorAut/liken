---
title: "Iterative Workloads"
---

## Reminder

In the [Record Linkage tutorial](../tutorials/record-linkage.md) you found out that **Liken** creates a `canonical_id`. By default this `canonical_id` is an autoincrementing numeric identifier starting from zero.

In this chapter we explore the configuring needed to canonicalize a dataset iteratively. By iteratively we mean appending to the same dataset — for example a dataset of customers that is appended to with new customers in a given time interval.

!!! note
    Here we explore the implications for batch workloads, especially for datasets that tend to append data

## Canonical IDs

A new canonical ID every time we instantiate a `Dedupe` class isn't going to be practical for our use case. In fact, given our use case, we're likely to already have a canonical ID (literally an **Liken** `canonical_id`, or another). So we should use that instead and pass it in as a string identifier to the `id` argument of the `canonicalize` function. See [the tutorial](../tutorials/record-linkage.md) for a recap.

## The Problem

Preparing an existing `canonical_id` column for an append is your job, not **Liken's**: new records need identifiers that continue the existing numbering before they are stacked onto the canonicalized dataset. **Liken** provides the preprocessors and dedupers; the identifier bookkeeping below is the part you carry out yourself. The suggested steps to take are:

1. Add a column, `canonical_id`, to the append dataset that continues the existing numbering: if the existing dataset holds `N` rows (ids `0` to `N-1`), the `n` appended rows get ids `N` to `N + n - 1`.
2. Append ("stack") your datasets.
3. Instantiate `Dedupe` and pass `id="canonical_id"` to the canonicalizer.

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

Canonicalize the stacked frame with `id="canonical_id"`:

```python
import liken as lk

pipeline = lk.pipeline().step(
    lk.col("name", preprocessors=[lk.preprocessors.lower()]).fuzzy(threshold=0.7)
)

df = lk.dedupe(stacked).apply(pipeline).canonicalize(id="canonical_id", keep="first")
```

 canonical_id  |     name
---------------|----------------
       0       |  Alice Wong
       1       |  Bob Lane
       0       |  alice wong
       3       |  Carol Fox

/// caption
The new duplicate inherited the existing id `0` — the group keeps its established identity.
///

Because `keep="first"` picks the earliest record of the group, the new record joins the existing id rather than the other way around. Later appends that duplicate other members of the group likewise collapse onto id `0`. The dataset keeps one continuous identifier space across runs.

## Decision Tree

``` mermaid
flowchart TD

    df{{"`DataFrame already has a **canonical_id**?`"}}
    id1{{"`**id** defined in canonicalize()?`"}}
    id2{{"`DataFrame carries an **id** column to use?`"}}
    idiscanonical{{"`**id** is the same as **canonical_id**?`"}}

    autoincrement("`Create a new autoincrementing **canonical_id**`")
    copy("`Copy **id** to create **canonical_id**`")
    overwrite("`Copy **id** to overwrite **canonical_id**`")
    existing("`Use existing **canonical_id**`")

    df-- yes -->id1
    df-- no -->id2

    id1-- no -->existing
    id1-- yes -->idiscanonical

    idiscanonical-- yes -->existing
    idiscanonical-- no -->overwrite

    id2-- no -->autoincrement
    id2-- yes -->copy
```
