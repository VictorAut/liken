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

Throughout **Liken** when you want to use a deduper, you `apply` it.

!!! note
    `Dedupe` only assumes the use of the `exact` strategy if you *don't* `apply` a strategy. This is the only assumption made with regards to which dedupers are used. Once you make use of `apply()` it's up to you to explicitely define what dedupers to use — only what is *in* the apply is executed.

## Dictionaries of Dedupers

We just saw how to deduplicate a dataframe having applied a *single* deduper. What if you want to deduplicate with more? 

The **Dict API** solves this with dictionaries. `drop_duplicates()` will no longer accept any arguments — columns will now be defined as the keys to the dictionary. 

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

The above might be understood as "Deduplicate only exact emails. Then, similar addresses. Finally, any records that have 3 out of 4 of those categories matching".

!!! note "`keep` arg"
    The `keep` argument accepts the literals "first" or "last" which defines which record will be kept from a duplicate set of records.

## Recap

If you're only ever looking to aim to *drop* duplicates then you can skip the next section on [**Record Linkage**](../../tutorials/record-linkage.md).

If the above is as complex a strategy as you'll ever use, then stop here.

!!! success "You learnt:"
    - Multiple strategies for multiple columns can be defined using a dictionaty, thus forming the **Dict API**.
    - The **Dict API** only allows a single call to `.apply`