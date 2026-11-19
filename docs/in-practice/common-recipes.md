---
title: "Common Recipes"
---


## Threshold vs Predicate

Liken's [built-in dedupers](./choosing-a-deduper.md#built-in-dedupers) are classed as being *Threshold* or *Predicate* dedupers.

Threshold dedupers are easy to understand — things are likened one-to-another when a similarity score exceeds the threshold. Predicate dedupers operate based on a binary choice: something either is or isn't something, where that something is what the predicate deduper selects for.

The above explanation might still sound tricky, especially regarding predicate dedupers. It's worth, therefore, listing some key facts about predicate dedupers.

1. Predicate dedupers are actually derived from the same base classes as threshold dedupers. Ultimately, this means you can use them just like you would a threshold deduper. But...
2. ...they feel like "filters".
3. Predicate Dedupers feel like "filters" because they can be powerfully composed using [AND semantics](../tutorials/applying-dedupers.md#and-semantics) with threshold dedupers in pipelines...
4. ...in that regard it's generally recommended to use predicate dedupers **only** when applying [pipelines](../tutorials/applying-dedupers.md#pipelines-of-dedupers) with `lk.pipeline()` — you can use them outside of pipelines, but the use cases are limited.
5. Treating them under the "filter" paradigm is especially useful when considering that Liken implements [predicate-evaluation optimizations](./performance.md#use-pipelines), which cut down on compute. Predicate evaluation states that when combining dedupers using [AND semantics](../tutorials/applying-dedupers.md#and-semantics), the predicate dedupers will be executed first regardless of the defined order — and subsequent threshold dedupers will operate on a subset of data.
6. Finally, predicate dedupers can always be subjected to a negation (with `~`), as defined in [NOT semantics](../tutorials/applying-dedupers.md#not-semantics).

## Powerful Pipelines

Consider deduplicating addresses. You can apply `lk.fuzzy()` with a well chosen `threshold` yet still get many misses, or worse, false positives.

To resolve this effectively, opt for tiered approaches that leverage the features of pipelines, and define several stages of deduplication:

```python
pipeline = (
    lk.pipeline()
    .step(
        [
            lk.col("address").exact(),
            lk.col("address").str_len(max_len=5),
            ~lk.col("address").isna(),
        ],
    )
    .step(
        [
            lk.col("address").fuzzy(threshold=0.95),
            lk.col("address").str_len(min_len=6, max_len=10),
        ],
    )
    .step(
        [
            lk.col("address").fuzzy(threshold=0.85),
            lk.col("address").str_len(min_len=11, max_len=20),
        ]
    )
    .step(
        [
            lk.col("address").fuzzy(threshold=0.75),
            lk.col("address").str_len(min_len=21),
        ]
    )
)
```

Now, only on longer `address` strings is more tolerance allowed.

### Notes on Pipeline Executes

Each step produces clusters of linked records. Steps combine as OR: records linked by any step end up in one cluster. But clusters are not recomputed from scratch at the end — each step rewrites the canonical ids built up to that point so far, so the order of steps can change the outcome.

 id   |    x     |    y
------|----------|---------
  0   |    a     |    m
  1   |    a     |    n
  2   |    z     |    n

/// caption
Three records: `x` links rows 0 and 1; `y` links rows 1 and 2.
///

With `x` first, step `x` merges rows 0 and 1; step `y` then merges rows 1 and 2, and row 2 inherits row 1's canonical id, all three records share one id. With `y` first, row 2's cluster forms before `x` runs — and `x` never touches it, so row 2 keeps its own id. Run both orders and compare:

```python
import liken as lk

xy = lk.pipeline().step(lk.col("x").exact()).step(lk.col("y").exact())
yx = lk.pipeline().step(lk.col("y").exact()).step(lk.col("x").exact())

lk.dedupe(df).apply(xy).canonicalize().collect()  # canonical_id: [0, 0, 0]
lk.dedupe(df).apply(yx).canonicalize().collect()  # canonical_id: [0, 0, 1]
```

!!! tip
    When dedupers overlap, order them from the broadest to the narrowest.

## Which Preprocessor When?

Preprocessors are great additions to pipelines for fine tuning of deduplication behaviour. Favour their usage over boilerplate preprocessing done before using Liken.

A few key points are worth noting:

1. [`strip`](../reference/preprocessors.md), [`remove_punctuation`](../reference/preprocessors.md), [`normalize_unicode`](../reference/preprocessors.md) should be made ample use of. There are few instances where not using them is meaningful.
2. [`lower`](../reference/preprocessors.md) and [`ascii_fold`](../reference/preprocessors.md) are more nuanced and should be used with more care.
3. [`alnum`](../reference/preprocessors.md) strips spaces — this can be powerful when used with `lk.fuzzy` but needs caution when used with tokenization based threshold dedupers (namely, `lk.tfidf` and `lk.lsh`).
4. [`collapse_whitespace`](../reference/preprocessors.md) collapses internal whitespace runs to single spaces. It complements `strip`, which only trims the edges. Whitespace at the start or end collapses to one space, it is not removed.
5. [`regex_replace`](../reference/preprocessors.md) a useful fail-safe option for all other cleaning processes.

## How Clusters Form

1. Applied dedupers pair up matching records.
2. Pairs merge *transitively*. If record `a` matches `b`, and `b` matches `c`, **all three land in one cluster** — even when `a` and `c` would not match each other directly.
3. One record per cluster becomes the representative, chosen by `keep` under the survivorship policy: the first or the last record of the cluster, in row order.

For canonicalization, additionally:

4. Every member of the cluster receives the representative's `canonical_id`.

## Transitivity

Look at the following dummy dataset.
- `"london"` and `"londn"` match at 0.91
- `"londn"` and `"lond"` match at 0.86
- `"london"` and `"lond"` match at 0.73

So, at a chose threshold of 0.85, only two out of three records directly match.

<!-- vale Vale.Spelling = NO -->

 id   |  address
------|-----------
  0   |  london
  1   |  londn
  2   |  lond

/// caption
No pair of the three is exactly equal, and `london`/`lond` do not match directly at a score of 0.85
///

<!-- vale Vale.Spelling = YES -->

All three records, however, receive the **same `canonical_id`**, with `"londn"` "bridging" the other two records. At threshold 0.95 no pair matches at all, and each record keeps its own id. Whether two records share an id is decided by the whole chain of matches, not by any single pair.

## Recipes

### Matching Company Names

Company columns carry legal suffixes (such as `Ltd`, `LLC`, `SE`, `GmbH`) that can create spurious fuzzy matching and even exact matching on casing:

```python
import liken as lk

pipeline = lk.pipeline().step(
    lk.col(
        "company",
        preprocessors=[
            lk.preprocessors.normalize_company(),
            lk.preprocessors.lower(),
        ],
    ).fuzzy(threshold=0.9)
)

df = lk.dedupe(df).apply(pipeline).drop_duplicates()
```

`normalize_company` strips the legal-form suffix and `lower` removes the casing, so `BASF SE` and `basf` merge, as do `Acme Ltd.` and `ACME Limited`.

### Cleaning Free Text Before Exact Matching

Descriptions and titles that differ only by filler words, like `the` and `a`, are the same text to you, but not to an exact matcher:

```python
import liken as lk

pipeline = lk.pipeline().step(
    lk.col(
        "description",
        preprocessors=[
            lk.preprocessors.remove_stopwords(),
            lk.preprocessors.collapse_whitespace(),
            lk.preprocessors.strip(),
        ],
    ).exact()
)

df = lk.dedupe(df).apply(pipeline).drop_duplicates()
```

`remove_stopwords` deletes the filler words, `collapse_whitespace` closes the gaps it leaves behind, and `strip` removes the outer spacing, so `the quick brown fox` and `quick  brown fox` become equal strings for the `exact` deduper.

!!! tip "Preprocessor order"
    These recipes chain preprocessors, which run left to right (top to bottom). Order matters: `collapse_whitespace` then `strip` after `remove_stopwords`, because stopword removal leaves stray spacing behind.

### Canonicalizing Person Names

Names arrive with titles, suffixes, punctuation and casing that hide the fact they are the same person:

```python
import liken as lk

pipeline = lk.pipeline().step(
    lk.col(
        "name",
        preprocessors=[
            lk.preprocessors.normalize_names(),
            lk.preprocessors.remove_punctuation(),
            lk.preprocessors.lower(),
        ],
    ).exact()
)

df = lk.dedupe(df).apply(pipeline).canonicalize()
```

`normalize_names` keeps only first, middle and last names, and the punctuation and casing preprocessors remove the rest, so `Dr. John A. Smith Jr.` and `john a smith` become one exact match.
