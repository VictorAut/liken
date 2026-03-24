---
title: Applying Dedupers
---

## Applying Your First Deduper

In your [First Steps](../first-steps.md#the-simplest-example) you found out how to replicate exact deduplication with **liken** — in fact it was *the* [exact](../../reference/liken.md#liken.exact) deduper in use. It came bundled with `dedupe` when you called `drop_duplicates()` with no other deduper.

When it comes to 'using' dedupers, they are *applied* first, before calling `drop_duplicates()`:


```python {hl_lines="4"}
import liken as lk

df = lk.dedupe(df).apply(fuzzy()).drop_duplicates("address").collect()
```

Throughout **Liken** when you want to use a deduper, you always use `apply`. This is true regardless of the type of deduper (i.e. similarity based or predicate based).

## Dictionaries of Dedupers

We just saw how to deduplicate a dataframe having applied a *single* deduper. What if you want to deduplicate with a collection of dedupers?

The **Liken** solves this with dictionaries. `drop_duplicates()` no longer accepts a column label argument — columns will now be defined as the keys to the dictionary. 

```python
import liken as lk

collection = {
    "email": lk.exact(),
    "address": (
        lk.fuzzy(threshold=0.98),
        lk.tfidf(threshold=0.9, ngram=(1, 2), topn=1),
    ),
    (
        "marital_status",
        "has_car",
        "flat_or_house",
        "urban_or_rural",
    ): lk.jaccard(threshold=0.75),
}

df = lk.dedupe(df).apply(collection).drop_duplicates(keep="first").collect()
```

When using dictionaries, each key defines a deduper for a column of data dataframe. Dedupers can be passed as a tuple in which case each deduper will be used sequentially.
In the above case, the defined collection reads as "Deduplicate exact emails. Then, similar addresses using Fuzzy and then TF-iDF. Finally, any deduplicate records that have 3 out of 4 of those categories matching".

!!! note "`keep` arg"
    The `keep` argument accepts the literals "first" or "last" which defines which record will be kept from a duplicate set of records, based on their position in the dataframe.

## Pipelines of Dedupers

**Liken** exposes a pipeline builder function for you to build complex, composable pipelines. Above we saw that we dedupers can be defined as collections using a dictionary construct. Here, we first replicate that functionality but using the `liken.pipeline` module instead:

```python
import liken as lk

pipeline = (
    lk.pipeline()
    .step(lk.on("email").exact())
    .step(lk.on("address").fuzzy(threshold=0.98))
    .step(lk.on("address").tfidf(threshold=0.9, ngram=(1, 2), topn=1))
    .step(
        lk.on(
            (
                "marital_status",
                "has_car",
                "flat_or_house",
                "urban_or_rural",
            ).jaccard(threshold=0.8)
        ),
    )
)

df = lk.dedupe(df).apply(pipeline).drop_duplicates().collect()
```

A pipeline has the following features:

- Each `step` in a pipeline represents a deduplication step
- Column access is provided by `lk.on`.
- Dedupers are provided as method calls to `lk.on`.

### AND semantics

Consider an "address" column. Deduplicating with a similarity based dedupers, such as `fuzzy`, might get some good results. However, a typical "address" column might have several conditions that make a uniform application of a similarity deduper inadequate. For instance, the "address" column might contain nulls, which would typically be considered idnetical records. Alternatively, addresses might vary greatly in length, to the point that short addresses might be deduplicated too aggresively by a low `threshold` parameter in a deduper, thus yielding too many false positives. In such case you would be motivated to only apply a similarity deduper in conjunction with another deduper such that deduplication only happens when the formaer AND the later agree. 

AND semantics are supported in **Liken** when lists of dedupers are passed to a `step` in a pipeline. All conditions in a step must match for records to be linked by the pipeline:

```python
import liken as lk

pipeline = (
    lk.pipeline()
    .step(
        [
            lk.on("address").fuzzy(),
            lk.on("address").str_len(min=10),
        ] # AND: both conditions must hold
    )
    .step(lk.on("email").fuzzy(threshold=0.98))
)


df = lk.dedupe(df).apply(pipeline).canonicalize().collect()
```

In the above case, for the first step, both conditions must hold — similar addresses will only be deduplicated if the length of addresses has a minimum length of 10 characters.

Although it might appear that AND semantics are akin to filters, this is not true in **Liken's** internals. AND semantics are only supported *between dedupers*. Filter's are not supported because no provision is ever made to drop data. In that regard, what might appear as a filter, for example the `isna()` function, is a actually a *predicate* deduper whose action is to canonicalize all records that fit the null description.

!!! info
    AND semantics are supported between any **Liken** deduper but are especially effective when combining a similarity deduper with a predicate deduper. Additionally, **Liken** features an optimisation, "Rule Predication" which enforces the execution of the predicate deduper first given an AND semantic `step` — the subsequent dedupers will then only operate on the subset of identical records collected by the first predicate deduper. This optimisation works becase by their nature predicate dedupers operate close to *O(n)* whilst similarity dedupers generally operate at *O(^n^2)*.

### OR semantics

OR semantics behaviour is captured by distinct steps in a pipeline. 

```python
import liken as lk

pipeline = (
    lk.pipeline()
    .step(
        [
            lk.on("address").fuzzy(),
            lk.on("address").str_len(min=10),
        ]
    )
) # AND: both conditions must hold
```
```python
import liken as lk

pipeline = (
    lk.pipeline()
    .step(lk.on("address").fuzzy())
    .step(lk.on("address").str_len(min=10))
) # OR: either condition must hold
```

### NOT semantics

Predicate dedupers can be inverted to form a NOT semantics by using the `~` operator on the column accessor `lk.on`. If in our earlier example we wanted to deduplicate where the minimum length was *not* 20 characters then we would have the following:

```python
import liken as lk

pipeline = (
    lk.pipeline()
    .step(
        [
            lk.on("address").fuzzy(),
            ~lk.on("address").isna(), # NOT null
        ]
    )
)
```