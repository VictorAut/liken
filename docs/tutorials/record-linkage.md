---
title: Record Linkage
---

## What Is It

Up to now you've learnt about the two mechanism for applying strategies, the **Sequantial API** and the **Dict API**.

We've also done this within the context of *dropping duplicates*.

But, what if you want to retain your duplicate instances? And instead simply label them as such?

Enter **Record Linkage**, which conceptually describes the idea that the deduplication process you are doing is not to "clean" your dataset but rather to *link* it together. Now, a simple strategy that defines a fuzzy string deduplication of an `address` column will *label* the duplicates as duplicates rather than *dropping* them.

## Canonicalization

The idea behind **Record Linkage** as explained above is really quite simple. We do, however, need a mechanism to *decide* which of the records we have linked is the best one. That can be expressed as "canonicalization" by which we mean the process that chooses a "canonical" record.



## Enlace Implementation

You've actually already encountered **Enlace's** canonicalisation when dropping duplicates upon chosing [which `keep` argument to declare](../tutorials/applying-strategies/dict-api.md).

Let's look at a dummy dataset again:

```
>>> df
+--------+-----------+-----------------------+
|  uid   |  address  |          email        |
+--------+-----------+-----------------------+
|  a001  |  london   |   fizzpop@yahoo.com   |
|  a002  |   tokyo   |  fizzpop@yahoo.co.uk  |
|  a003  |   paris   |        a@msn.fr       |
+--------+-----------+-----------------------+
```

We're going to aim to link the above email addresses. To do so we're just going to swap `drop_duplicates()` with `canonicalize()`:

```python {hl_lines="5"}
from enlace import Dedupe, fuzzy

dp = Dedupe(df)
dp.apply(fuzzy(threshold=0.85))
df = dp.canonicalize("email", keep="first")
```

Now, checkout the outcome:

```
>>> df
+--------+-----------+-----------------------+----------------+
|  uid   |  address  |          email        |  canonical_id  |
+--------+-----------+-----------------------+----------------+
|  a001  |  london   |   fizzpop@yahoo.com   |        0       |
|  a002  |   tokyo   |  fizzpop@yahoo.co.uk  |        0       |
|  a003  |   paris   |        a@msn.fr       |        2       |
+--------+-----------+-----------------------+----------------+ 
```

!!! note
    The **Sequential API** and **Dict API** are equally valid as means to define strategies, regardless of whether the usecase requires canonicalization, or not.

## A Note On `canonical_id`s

Above you can see that a new field has been created. It's called `canonical_id` and any repeated `canonical_id` is, in fact, a duplicate. In that instance that was an auto-incrementing numeric field. As such, the repeated `canonical_id` represents the position in the DataFrame of the *canonical* record.

You can control this behaviour by passing an explicit label to the `id` argument of `canonicalize`. In that case, the `canonical_id` will become a copy of the defined `id`, or simply a reference to itself if it already exists. For example:

```python {hl_lines="5"}
from enlace import Dedupe, fuzzy

dp = Dedupe(df)
dp.apply(fuzzy(threshold=0.85))
df = dp.canonicalize("email", keep="first", id="uid") # included id arg
```

Now, checkout the variation:

```
>>> df
+--------+-----------+-----------------------+----------------+
|  uid   |  address  |          email        |  canonical_id  |
+--------+-----------+-----------------------+----------------+
|  a001  |  london   |   fizzpop@yahoo.com   |      a001      |
|  a002  |   tokyo   |  fizzpop@yahoo.co.uk  |      a001      |
|  a003  |   paris   |        a@msn.fr       |      a003      |
+--------+-----------+-----------------------+----------------+ 
```

This can be especially useful if instead of locating canonical records by index position in the DataFrame you want to do so based on a pre-existing identifier.
    

## Recap

Along with the [**Dict API**](../tutorials/applying-strategies/dict-api.md) understanding **Record Linkage** will cover the vast majority of users's needs. The next tutorial introduces the third and final [**Rules API**](../tutorials/advanced-strategies/rules-api.md) which exposes **Enlace's** most powerful functionality.

!!! success "You learnt:"
    - You only have to change `drop_duplicates()` for `canonicalize()` to achieve **Record Linkage** in **Enlace**.
    - Canonicalization creates a `canonical_id` field. Override the default auto-incrementing behaviour by defining an `id` arg.