---
title: Record Linkage
---

Up to now you've learnt how to use dedupers with `apply`, specifically within the context of *dropping duplicates*.

But, what if you want to retain your duplicate instances? And instead simply label them as such?

**Liken** supports **Record Linkage**, which describes the idea that the deduplication process you are doing is not to drop data from your DataFrame, but rather to *link* it together. So, a deduper that defines a fuzzy string deduplication of an `address` column will *label* the duplicates as duplicates rather than *dropping* them.

Retaining records as *known* duplicates instead of *dropping* duplicates is known as **Record Linkage** in **Liken**. However, this is also known as **Entity Resolution**. The link is provided by a **canonical** record, which in **Liken** is identified by the auto-generated `canonical_id` column.


## Canonicalization

You've actually already implicitely encountered **Liken's** canonicalisation when dropping duplicates upon chosing [which `keep` argument to declare](../tutorials/applying-dedupers.md#dictionaries-of-dedupers).

Let's look at a dummy dataset, `df`:


  uid   |  address  |          email        
--------|-----------|-----------------------
  a001  |  london   |   fizzpop@yahoo.com   
  a002  |   tokyo   |  fizzpop@yahoo.co.uk  
  a003  |   paris   |        a@msn.fr

/// caption
Two very clearly similar emails exist.
///     


We're going to aim to **link** the above email addresses. To do so, swap `.drop_duplicates` with `.canonicalize`:

```python
import liken as lk

df = (
    lk.dedupe(df)
    .apply(lk.fuzzy(threshold=0.85))
    .canonicalize(
        "email",
        keep="first",
    )
    .collect()
)
```

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

You can control this behaviour by passing an explicit label to the `id` argument of `.canonicalize`. In that case, the `canonical_id` will become a copy of the defined `id`, or simply a reference to itself if it already exists. For example:

```python
import liken as lk

df = (
    lk.dedupe(df)
    .apply(fuzzy(threshold=0.85))
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


## Synthetic Records

A canonical record can be linked to several child records. Use `.synthesize` to create a new canonical record that coalesces the values of the fields of the various child records:

```python
import liken as lk

result = (
    lk.dedupe(df)
    .apply(fuzzy(threshold=0.85))
    .canonicalize(
        "email",
        keep="first",
        id="uid", # `id` arg included
    )
)

synthetic_records = result.synthesize()
```