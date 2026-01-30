---
title: Sequential API
---

## Applying Your First Strategy

In your first steps you found out how to use the `Dedupe` class with exact deduplication — in fact it was *the* [exact](../../reference/liken.md#liken.exact) strategy. It came bundled with `Dedupe` when you called `drop_duplicates()` with no other strategies.

When it comes to 'using' strategies we'll talk about *applying* them to `Dedupe` first, before finally calling `drop_duplicates()`.

Let's look at the example from your [First Steps](../first-steps.md#the-simplest-example), except this time we're going to assume you want to use the [fuzzy](../../reference/liken.md#liken.fuzzy) strategy instead:

```python {hl_lines="4"}
from liken import Dedupe, fuzzy

lk = Dedupe(df)
lk.apply(fuzzy())
df = lk.drop_duplicates("address")
```

That easy!

!!! note
    `Dedupe` only assumes the use of the `exact` strategy if you *don't* apply a strategy. This is the only assumption made with regards to which strategies are used. Once you make use of `apply()` it's up to you to explicitely define what strategies to use — only what is *in* the apply is executed.

## Applying More Than One Strategy

In the above example, you might want to dedupe the `address` column based on a more than just fuzzy string matching. Let's consider applying *multiple* strategies. If you're using multiple strategies you'll probably also want to control to what degree the strategy is going to deduplicate the column with the `threshold` argument:

```python {hl_lines="4 5"}

from liken import Dedupe, fuzzy, tfidf

lk = Dedupe(df)
lk.apply(fuzzy(threshold=0.9))
lk.apply(tfidf(threshold=0.7))
df = lk.drop_duplicates("address")
```

The repeated use of the `apply()` to define which strategies you'll use on a single column make up the **Sequential API**.

!!! tip "`threshold`"
    All the strategies defined in the previous tutotial on [Deduplication Strategies](../strategies.md#liken-ready-strategies) have a `threshold` which you can use to **tune** the overall 'strength' of deduplication.

## Recap

The Sequential API will get you quite far and for many use cases it is sufficient for the casual user of **Liken**. For example, it is an excellent way to add functionality to what would otherwise remain a Pandas data pipeline. However, you may have observed that building a 'complex' strategy with multiple stratgies applied to *different* columns would be quite verbose using the **Sequential API**. In the next tutorial we'll explore the use of something more flexible.

!!! success "You learnt:"
    - You can define what strategies to use with the `apply` method.
    - Repeated calls to `apply` on a *single* column constitute the **Sequential API**.
    - Overall deduplication 'strength' is controlled with the `threshold` parameter.