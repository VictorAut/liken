---
title: "How-to: Build Iterative Workloads"
---

## Reminder

In the [Record Linkage tutorial](../tutorials/record-linkage.md) you found out that **Enlace** creates a `canonical_id`. By defult this `canonical_id` is an autoincrementing numeric identifier starting from zero. 

In this chapter we explore the configuring needed to canonicalize a dataset iteratively. By iteratively we mean with the *same* dataset — for example a dataset of customers that is appended to with new customers in a given time interval. 

!!! note
    Here we explore the implications for batch workloads, especially for datasets that tend to append data

## Canonical IDs

A new canonical ID everytime we instantiate a `Dedupe` class isn't going to be practical for our use case. In fact, given our use case, we're likely to already have a canonical ID (literally an **Enlace** `canonical_id`, or another). So we should use that instead and pass it in as a string identifier to the `id` argument of the `canonicalize` function. See [the tutorial](../tutorials/record-linkage.md#a-note-on-canonical_ids) for a recap.

## The Problem

**Enlace** does not currently possess preprocessing capabilities. For iterative, batch workloads, you will have to do carry out preprocessing steps yourself. The suggested steps to take are:

1. Add a column, `canonical_id`, that is an auto-incrementing numeric identified starting from the length of the dataset you will be appending to (`N`) and add one, i.e. `N+1` -> `n+1` where `n` is the length of the append dataset.
2. Append ("stack") your datasets.
3. Instantiate `Dedupe` and pass `id="canonical_id"` to the canonicalizer.

!!! warning
    This process is going to be a lot easier with numeric ids. It's possible to use string identifiers but it makes the process of incrementing on append datasets much harder to manage and reason about
