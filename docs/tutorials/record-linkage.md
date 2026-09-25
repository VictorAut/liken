---
title: Record Linkage
---

Up to now you've learnt how to use dedupers with `apply`, specifically within the context of *dropping duplicates*.

But, what if you want to retain your duplicate instances? And instead simply label them as such?

**Liken** supports **Record Linkage**, where the deduplication process you are doing is not to drop data from your DataFrame, but rather to *link* it together. So, a deduper that defines a fuzzy string deduplication of an `address` column will *label* the duplicates as duplicates rather than *dropping* them. Records are instead *canonicalized*.

Retaining records as *known* duplicates instead of *dropping* duplicates is known as **Record Linkage** in **Liken**. This is also known as **Entity Resolution**, in other literature. The link is provided by a **canonical** record, which in **Liken** is identified by the auto-generated `canonical_id` column.


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

The `keep` argument picks the *representative* of each duplicate group: `"first"` keeps the earliest record of the group in the dataframe's current row order, `"last"` the latest. The representative's identity is what the group's `canonical_id` will carry.

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

With `keep="last"` the group would be represented by its latest record instead, and both rows would carry `a002`'s position as their id.

You can control this behaviour by passing an explicit label to the `id` argument of `.canonicalize`. The values of the column you name become the `canonical_id` values: name `canonical_id` itself and it is left as is; name any other column and `canonical_id` is written from its values. For example:

```python
import liken as lk

df = (
    lk.dedupe(df)
    .apply(lk.fuzzy(threshold=0.85))
    .canonicalize(
        "email",
        keep="first",
        id="uid", # `id` arg included
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


## How Groups Form

Canonicalization is one model, end to end:

1. The applied dedupers pair up matching records — the same pairs that `drop_duplicates` would have collapsed.
2. Pairs merge *transitively*. If record `a` matches `b`, and `b` matches `c`, all three land in one group — even when `a` and `c` would not match each other directly.
3. One record per group becomes the representative, chosen by `keep`: the first or the last record of the group, in row order.
4. Every member of the group receives the representative's `canonical_id`.

Transitivity is worth seeing. At threshold 0.85, `"london"` and `"londn"` match (score 0.91), and `"londn"` and `"lond"` match (score 0.86) — but `"london"` and `"lond"` do not (score 0.73):

 id   |  address
------|-----------
  0   |  london
  1   |  londn
  2   |  lond

/// caption
No pair of the three is exactly equal, and `london`/`lond` do not match directly.
///

```python
import liken as lk

addresses = lk.pipeline().step(lk.col("address").fuzzy(threshold=0.85))

lk.dedupe(address_df).apply(addresses).canonicalize().collect()
```

All three records receive `canonical_id 0` — `"londn"` bridges the gap. At threshold 0.95 no pair matches at all, and each record keeps its own id. Whether two records share an id is decided by the whole chain of matches, not by any single pair.

## Inspecting Groups

Once `canonicalize` has run, `canonicals` answers the natural next question: *which groups actually hold duplicates?* It returns a dict of `canonical_id` to record count, for every group with at least `n` records — the default `n` is 2:

```python
import liken as lk

result = (
    lk.dedupe(df)
    .apply(lk.fuzzy(threshold=0.85))
    .canonicalize(
        "email",
        keep="first",
        id="uid",
    )
)

result.canonicals()  # {'a001': 2}
```

`a003` is absent from the result: its group holds a single record. Raise `n` to demand larger groups — `canonicals(3)` returns only groups of three or more records. `canonicals` collects the canonical ids to your machine, so on distributed backends it triggers a collect.


## Synthetic Records

A canonical record can be linked to several child records. Use `.synthesize` to create a new record per canonical group, coalescing the values of the group's child records:

```python
import liken as lk

result = (
    lk.dedupe(df)
    .apply(lk.fuzzy(threshold=0.85))
    .canonicalize(
        "email",
        keep="first",
        id="uid", # `id` arg included
    )
)

synthetic_records = result.synthesize()
```

`synthesize` returns a new DataFrame with exactly one record per canonical group. For each column, the synthetic value is the *first non-null* value among the group's records, in row order. The canonicalized DataFrame itself keeps all child records, unchanged — the synthetic records sit alongside them, in their own frame:

 canonical_id  |  uid   |  address  |        email
---------------|--------|-----------|---------------------
      a001     |  a001  |  london   |   fizzpop@yahoo.com
      a003     |  a003  |   paris   |       a@msn.fr

/// caption
One synthetic record per canonical group: `a001`'s group coalesces into a single record.
///

First non-null means the first record's null does not win — the first *non-null* value does:

  uid   |  address  |  email
--------|-----------|----------
  b001  |   None    |  x@y.io
  b002  |   oslo    |  x@y.io

```python
import liken as lk

result = lk.dedupe(null_df).apply(lk.exact()).canonicalize("email", id="uid")
result.synthesize()
```

 canonical_id  |  uid   |  address  |  email
---------------|--------|-----------|----------
      b001     |  b001  |   oslo    |  x@y.io

/// caption
`b001`'s address is null, so the synthetic record takes `b002`'s.
///

`synthesize` collects to your machine on distributed backends. Unlike `canonicals`, it does not refuse to run before a `canonicalize` — on an un-canonicalized chain it silently treats every record as its own group, which is rarely what you want.
