---
title: Deduplication Strategies
---

## Defining Strategies

A **strategy** is **Liken's** term for a deduplication method. In the previous section we saw how we can use the *exact* strategy to get parity with Pandas's implementation. In fact, **Liken** comes with many more strategies to deduplicate data. 

## **Liken**-Ready Strategies

|               | Strategy                                              | Description                                                                                      |
| ------------- | ----------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
|*single-column*| [`exact`](../reference/liken.md#liken.exact)       | You've already seen this in use *implicitely* in your [First Steps](../tutorials/first-steps.md)  |
|*single-column*| [`fuzzy`](../reference/liken.md/#liken.fuzzy)       | Fuzzy string matching                                                                            |
|*single-column*| [`tfidf`](../reference/liken.md/#liken.tfidf)       | String matching with Tf-Idf                                                                      |
|*single-column*| [`lsh`](../reference/liken.md/#liken.lsh)           | String matching with Locality Sensitive Hashing (LSH)                                            |
|*compound-column*| [`jaccard`](../reference/liken.md/#liken.jaccard) | Multi column similarity based on intersection of categorical data                                |
|*compound-column*| [`cosine`](../reference/liken.md/#liken.cosine)   | Multi column similarity based on dot product of numerical data                                   |

/// caption
*single-column* strategies apply to single columns and are implementation of near string matching. *compound-column* strategies are set operations where the values of the set are the values of the columns in a given record.
///

## Recap

!!! success "You learnt:"
    - **Liken** comes with ready to use **strategies**.
    - Strategies are split into those that have an effect on *single columns* and those that have an effect on *compound columns*.
