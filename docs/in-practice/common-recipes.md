---
title: "Common Recipes"
---

Variously common scenarios are explored here.

## Similarity vs Predicate

**Liken's** [built-in dedupers](../tutorials/first-steps.md#built-in-dedupers) are classed as being "Similarity" or "Predicate" dedupers.

Similarity dedupers are easy to understand — things are likened one-to-another when a similarity exceeds a threshold. Predicate dedupers operate based on a binary choice: something either is or isn't something, where that something is what the predicate deduper selects for.

The above explanation might still sound tricky, especially regarding predicate dedupers. It's worth, therefore, listing some key facts about predicate dedupers.

1. Predicate dedupers are actually derived from the same base classes as similarity dedupers. Ultimately, this means you can use them just like you would a similarity deduper. But...
2. ...they feel like "filters".
3. Predicate Dedupers feel like "filters" because they can be powerfully composed using [AND semantics](../tutorials/applying-dedupers.md#and-semantics) with similarity dedupers in pipelines...
4. ...in that regard it's generally recommended to use predicate dedupers **only** when defining pipelines with `lk.pipeline()` — you can use them outside of pipelines, but the use cases are limited.
5. Because they are fundamentally based on the same base classes as similarity dedupers, they are also accessed with the `lk.col()` expression.
6. The "filter" paradigm is especially useful when considering that **Liken** implements "rule predication" optimizations. Rule Predication states that when combining dedupers using [AND semantics](../tutorials/applying-dedupers.md#and-semantics), the predicate dedupers will be executed first regardless of the defined order — and subsequent similarity dedupers will operate on a subset of data.
7. Finally, predicate dedupers can always be subjected to a negation (with `~`), as defined in [NOT semantics](../tutorials/applying-dedupers.md#not-semantics)

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
            lk.col("address").str_len(min_len=5, max_len=10),
        ],
    )
    .step(
        [
            lk.col("address").fuzzy(threshold=0.85),
            lk.col("address").str_len(min_len=10, max_len=20),
        ]
    )
    .step(
        [
            lk.col("address").fuzzy(threshold=0.75),
            lk.col("address").str_len(min_len=20),
        ]
    )
)
```

Now, only on longer `address` strings is more tolerance allowed.

## Which Preprocessor When?

Preprocessors are great additions to pipelines for fine tuning of deduplication behaviour. Favour their usage over boilerplate preprocessing done before using **Liken**.

A few key points are worth noting:

1. [`strip`](../reference/preprocessors.md), [`remove_punctuation`](../reference/preprocessors.md), [`normalize_unicode`](../reference/preprocessors.md) should be made ample use of. There are few instances where not using them is meaningful.
2. [`lower`](../reference/preprocessors.md) and [`ascii_fold`](../reference/preprocessors.md) are more nuanced and should be used with more care.
3. [`alnum`](../reference/preprocessors.md) strips spaces — this can be powerful when used with `lk.fuzzy` but needs caution when used with tokenization based similarity dedupers (namely, `lk.tfidf` and `lk.lsh`).