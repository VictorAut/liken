---
title: Types of Strategies
---

## Continuous vs Discrete

Continuous strategies — or "similarity at a threshold" strategies are introduced early in **Liken** and are available in the top-level [`liken`](../reference/liken.md) package. Discrete ("binary") strategies are introduced later on as part of the more specialised Rules API in the [`liken.rules`](../reference/rules.md) sub-package.

Whilst they seem different, they are in fact all instances of the same base class in the internals of **Liken**, and it's why [customizing your own strategy](../tutorials/customizing-strategies.md) requires a single registration process *regardless* of whether you're looking to create a strategy that deduplicates at a similarity threshold or based on a discrete choice — on a single column or multiple columns.