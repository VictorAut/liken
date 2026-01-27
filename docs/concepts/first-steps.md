---
title: First Steps
---

## Where Bears Roam

**Enlace** is fundamentally a deduplication library for DataFrames, so in this tutorial we will assume that you have a basic awareness of at least one DataFrame library in Python. We'll steer clear of too much syntax outside of **Enlace** — but our reference point will be [Pandas](https://pandas.pydata.org/) DataFrames.

??? question "Not a Pandas user?"
    Not to worry, **Enlace** supports [multiple backends](../concepts/supported-backends.md).

Additionally, code blocks shown in this tutorial will will assume that a DataFrame, generically labelled `df`, will be available. Perhaps because you did something like this:

```python
import pandas as pd

df = pd.read_csv("my_handy_dataset.csv")    # we won't be repeating this
```

!!! info "Talk about 'handy'..."
    **Enlace** provides synthesised data in the [`datasets` package](../concepts/using-synthetic-data.md). Now and then you may notice we're making reference to it's *handiest* dataset, [`fake_10`](../../reference/datasets/#enlace.datasets.fake_10).

## The Simplest Example

Back to Pandas. Pandas provides deduplication facilities with it's DataFrame method [`.drop_duplicates`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop_duplicates.html). It's well known and we'll be recycling the syntax in **Enlace**. If you've used this function before then you know that it does *exact* deduplication, which is to say it will keep only one instance of an *exactly* repeated record. It can, however, consider only certain columns. 

Let's replicate this with **Enlace's** `Dedupe` class, on our data's `address` column:

```python

from enlace import Dedupe

dp = Dedupe(df)
df = dp.drop_duplicates("address")
```

We can also deduplicate multiple columns at once:

```python

from enlace import Dedupe

dp = Dedupe(df)
df = dp.drop_duplicates(columns=["address", "email"])
```

Great! You've done what Pandas can do, at the cost of an additional line of code. However, things get sticky when your records aren't quite *exactly* the same. Look at the following instances of emails in this dummy dataset:

```
+------+-----------+---------------------+
| id   |  address  |         email       |
+------+-----------+---------------------+
|  1   |  london   |  fizzpop@yahoo.com  |
|  2   |   tokyo   |  FizzPop@yahoo.com  |
|  3   |   paris   |       a@msn.fr      |
+------+-----------+---------------------+
```

Unfortunately "fizzpop" and "FizzPop" just aren't the same...

## Near deduplication

When things aren't *exactly* the same, you can still deduplicate data — it's just going to be harder and it's going to fall to you to define how "near" two things are. **Enlace** is built so that you can focus on defining *what* you want out of a near deduplication process. The goal will be to be able to define neat and clear-cut ways to deduplicate data with the least amount of code possible. To that end, any near deduplication process from here on out will called a deduplication **strategy**, and in the next tutorial you'll find out how to use **Enlace's** native **strategies**.

## Recap

!!! success "You learnt:"
    - **Enlace** has an easy to use `Dedupe` class.
    - `Dedupe` let's you use a `drop_duplicates` method.
    - For near deduplication processes **Enlace** refers to deduplication **strategies**.