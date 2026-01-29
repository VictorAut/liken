---
title: Deduplication Strategies
---

## Defining Strategies

A **strategy** is **Enlace's** term for a deduplication method. In the previous section we saw how we can use the *exact* strategy to get parity with Pandas's implementation.

Here we will see that **Enlace** comes with ready-to-use strategies.

## **Enlace**-Ready Strategies

It's worth taking stock of the fact that when we do any kind of near deduplication process we will have to limit ourselves to applying said near deduplication processes (**strategies**) to individual columns. Whilst this is generally true, there are some **stratgies** that are built for multiple columns by default.

|               | Strategy                                              | Description                                                                                      |
| ------------- | ----------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
|*single-column*| [`exact`](../../reference/enlace/#enlace.exact)       | You've already seen this in use *implicitely* in your [First Steps](../tutorials/first-steps.md)  |
|*single-column*| [`fuzzy`](../../reference/enlace/#enlace.fuzzy)       | Fuzzy string matching                                                                            |
|*single-column*| [`tfidf`](../../reference/enlace/#enlace.tfidf)       | String matching with Tf-Idf                                                                      |
|*single-column*| [`lsh`](../../reference/enlace/#enlace.lsh)           | String matching with Locality Sensitive Hashing (LSH)                                            |
|*compound-column*| [`jaccard`](../../reference/enlace/#enlace.jaccard) | Multi column similarity based on intersection of categorical data                                |
|*compound-column*| [`cosine`](../../reference/enlace/#enlace.cosine)   | Multi column similarity based on dot product of numerical data                                   |


## Recap

!!! success "You learnt:"
    - **Enlace** comes with ready to use **strategies**.
    - Strategies are split into those that have an effect on *single columns* and those that have an effect on *compound columns*.
