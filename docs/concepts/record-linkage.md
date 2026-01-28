---
title: Record Linkage
---

## What is it

Up to now you've learnt about the two mechanism for applying strategies, the **Sequantial API** and the **Dict API**.

We've also done this within the context of *dropping duplicates*.

But, what if you want to retain your duplicate instances? And instead simply label them as such?

Enter "record linkage" which conceptually describes the idea that the deduplication process you are doing is not to "clean" your dataset but rather to *link* it together. Now, a simple strategy that defines a fuzzy string deduplication of an `address` column will *label* the duplicates as duplicates rather than *dropping* them.

## Canonicalization

The idea behind record linkage as explained above is really quite simple. We do, however, need a mechanism to *decide* which of the records we have linked is the best one. That can be expressed as "canonicalization" by which mean the process that chooses a "canonical" record. 

## Enlace implementation

You've actually already encountered **Enlace's** canonicalisation when dropping duplicates upon chosing [which `keep` argument to declare](../concepts/applying-strategies/dict-api.md).

Let's look at a dummy dataset again:

```
>>> df
+------+-----------+-----------------------+
| id   |  address  |          email        |
+------+-----------+-----------------------+
|  1   |  london   |   fizzpop@yahoo.com   |
|  2   |   tokyo   |  fizzpop@yahoo.co.uk  |
|  3   |   paris   |        a@msn.fr       |
+------+-----------+-----------------------+
```

We're going to aim to link the above email addresses. To do so we're just going to swap `drop_duplicates` with `canonicalize`:

```python {hl_lines="5"}
from enlace import Dedupe, fuzzy

dp = Dedupe(df)
dp.apply(fuzzy(threshold=0.85))
df = dp.canonicalize("email", keep="first")
```

Now, checkout the outcome:

```
>>> df
+------+-----------+-----------------------+----------------+
| id   |  address  |          email        |  canonical_id  |
+------+-----------+-----------------------+----------------+
|  1   |  london   |   fizzpop@yahoo.com   |        0       |
|  2   |   tokyo   |  fizzpop@yahoo.co.uk  |        0       |
|  3   |   paris   |        a@msn.fr       |        2       |
+------+-----------+-----------------------+----------------+ 
```

## A note on `canonical_id`s

Above you can see that a new field has been created. It's called `canonical_id` and any repeated id is, in fact, a duplicate.

## Asking the big questions