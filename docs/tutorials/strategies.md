---
title: Deduplication Strategies
---

## Defining Strategies

A **strategy** is **Liken's** term for a deduplication method. In the previous section we saw how we can use the *exact* strategy to get parity with Pandas's implementation.

Here we will see that **Liken** comes with ready-to-use strategies.

## **Liken**-Ready Strategies

It's worth taking stock of the fact that when we do any kind of near deduplication process we will have to limit ourselves to applying said near deduplication processes (**strategies**) to individual columns. Whilst this is generally true, there are some **stratgies** that are built for multiple columns by default.

|               | Strategy                                              | Description                                                                                      |
| ------------- | ----------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
|*single-column*| [`exact`](../reference/liken.md#liken.exact)       | You've already seen this in use *implicitely* in your [First Steps](../tutorials/first-steps.md)  |
|*single-column*| [`fuzzy`](../reference/liken.md/#liken.fuzzy)       | Fuzzy string matching                                                                            |
|*single-column*| [`tfidf`](../reference/liken.md/#liken.tfidf)       | String matching with Tf-Idf                                                                      |
|*single-column*| [`lsh`](../reference/liken.md/#liken.lsh)           | String matching with Locality Sensitive Hashing (LSH)                                            |
|*compound-column*| [`jaccard`](../reference/liken.md/#liken.jaccard) | Multi column similarity based on intersection of categorical data                                |
|*compound-column*| [`cosine`](../reference/liken.md/#liken.cosine)   | Multi column similarity based on dot product of numerical data                                   |


## Recap

!!! success "You learnt:"
    - **Liken** comes with ready to use **strategies**.
    - Strategies are split into those that have an effect on *single columns* and those that have an effect on *compound columns*.
