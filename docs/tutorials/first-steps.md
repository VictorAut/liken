---
title: First Steps
---

## Installation

The first step is to install **Liken**:

```shell
pip install liken
```

## Where Bears Roam

**Liken** is a deduplication library for DataFrames, so in this tutorial we will assume that you have a basic awareness of at least one DataFrame library in Python. We'll steer clear of too much syntax outside of **Liken** — but our reference point will be [Pandas](https://pandas.pydata.org/) DataFrames.

??? question "Not a Pandas user?"
    Not to worry, **Liken** supports [multiple backends](../tutorials/supported-backends.md).

Additionally, code blocks shown in this tutorial will assume that a DataFrame, generically labelled `df`, will be available. Perhaps because you did something like this:

```python
import pandas as pd

df = pd.read_csv("my_handy_dataset.csv") # This won't be repeated
```

## The Simplest Example

The Pandas DataFrame library provides deduplication facilities with it's DataFrame method [`.drop_duplicates`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop_duplicates.html). It's a well-known and widely used method and we'll be recycling the syntax in **Liken**. If you've used this function before then you know that it does *exact* deduplication, which is to say it will keep only one instance of an *exactly* repeated record. 

As mentioned above, **Liken** recycles the `drop_duplicate` syntax:

```python

from liken import Dedupe

lk = Dedupe(df)
df = lk.drop_duplicates("address")
```

Above, we drop duplicates on an "address" column. We achieved this by virtue of the `Dedupe` class. We can also deduplicate multiple columns at once:

```python

from liken import Dedupe

lk = Dedupe(df)
df = lk.drop_duplicates(columns=["address", "email"])
```

Great! You've done what Pandas can do, at the cost of an additional line of code. However, things get sticky when your records aren't quite *exactly* the same. Look at the following instances of emails in this dummy dataset:


 id   |  address  |         email       
------|-----------|---------------------
  1   |  london   |  fizzpop@yahoo.com  
  2   |   tokyo   |  FizzPop@yahoo.com  
  3   |   paris   |       a@msn.fr      

/// caption
Unfortunately, "fizzpop" and "FizzPop" just aren't the same...
///

As you can see, this dummy dataset containing 3 unique emails. Using `drop_duplicates` straight from pandas won't do anything here, as "fizzpop@yahoo.com" and "FizzPop@yahoo.com" are not the same strings. In this particular case, some light preprocessing could deal with the issue allowing you to proceed with some useful deduplication. However, that's not always possible — or at least not easily achievable.

## Near Deduplication

When things aren't *exactly* the same, you can still deduplicate data — it's just going to be harder and it's going to fall to you to define how "near" two things are. **Liken** is built so that you can focus on defining *what* you want out of a near deduplication process. The goal will be to be able to define neat and clear-cut ways to deduplicate data with the least amount of code possible. To that end, any near deduplication process from here-on-out will be called a deduplication **strategy**, and in the next tutorial you'll find out how to use **Liken's** native **strategies**.


## Recap

!!! success "You learnt:"
    - **Liken** has an easy to use `Dedupe` class.
    - `Dedupe` let's you use a `drop_duplicates` method.
    - For near deduplication processes **Liken** refers to deduplication **strategies**.