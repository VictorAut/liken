---
title: "How-to: Optimise for Performance"
---

Near Deduplication is a complex, compute-intensive task. Generally speaking, deduplication scales as *O(n^2^)*, as any given record has to be looked up against every other record. Below we discuss three techniques to make a deduplication scale more efficiently.

## Use the Rules API

The [Rules API](../tutorials/advanced-strategies/rules-api.md) implements the use of [discrete strategies](../tutorials/advanced-strategies/discrete-strategies.md). **Liken** implements predicate pushdown when using discrete stratagies. As their outputs are a binary choice, a `Rules` strategy that implements combinations can be used to optimise for size based on key conditions. Let's look at an example based on the following dummy data:

id| address                  | email
--|--------------------------|-------
1 | None                     | random.company@yahoo.com
2 | None                     | legit.holdings@msn.dk
3 | 43 queensbridge, n99 6lt | extreme.trees@plants.co.uk
4 | 65 lindberg way, 90345   | extreme.trees@plants.co.uk

/// caption
///

Now, an attempting to deduplicate the "email" column with an exact deduper could be implemented like such:

``` python
lk = Dedupe(df)
lk.apply(Rules(on("email", exact())))
lk.drop_duplicates()
```

In the background **Liken** will loop through all 4 email records and identify the last 2 as duplicates. But, you may be well placed to qualify your strategy with a discrete strategy. This might be that you only want to consider "email" instances as valid for deduplication if the "address" column is itself not null:

``` python
lk = Dedupe(df)
lk.apply(Rules(on("email", exact()) & on("address", ~isna())))
lk.drop_duplicates()
```

In this case, with predicate pushdown the exact deduper will only loop through the instances of records where the "address" column is not null, in this case only 2 records — this can result in a significant performance boost because the discrete strategy operates as *O(n)* returning a new *n' < n*. It's always useful to try to qualify a threshold strategy with a discrete strategy, especially if performance is an issue and the logic is well motivated from a logical point of view. Regardless of the above two cases, with our dummy data, the deduplicated data looks the same:

id| address                  | email
--|--------------------------|-------
1 | None                     | random.company@yahoo.com
2 | None                     | legit.holdings@msn.dk
3 | 43 queensbridge, n99 6lt | extreme.trees@plants.co.uk

/// caption
///

## Use the LSH strategy

The LSH strategy is faster than *O(n^2^)* (but slower than *O(n)*). In fact it is approximately *O(nk)* where *k << n*. LSH, however, requires extensive testing and must be tuned.

Use the [performance benchmarks](../explanation/performance.md) and computational scaling explanations to get an idea of where you can obtain a performance boost.

## Use Partitioned Data

**Liken** [supports the use of PySpark](../tutorials/supported-backends.md/#pyspark). Liken will execute the `Dedupe` class in every Spark worker node, where each worker node recieves a partition. You can achieve this by reading in an already partitioned dataset, or by re-partitioning a dataset.

!!! Note
    Re-partitioning in for deduplication workloads often makes use of a "Blocking Key". A blocking key is generated in the dataset and each partition is chosen based on the value of a blocking key. This is especially useful when we know that duplicates are never (or very unlikely) going to be found *across* blocking keys. As an example, the blocking key could be the first letter of a customer's name. This can then be used to divide (partition) a dataset into more manageable chunks that are already related by an inherently meaningful feature.