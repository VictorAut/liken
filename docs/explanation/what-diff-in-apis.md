---
title: The APIs
---

## Sequential, Dict, Rules

Refer to the respective [Sequential API](../tutorials/applying-strategies/sequential-api.md), [Dict API](../tutorials/applying-strategies/dict-api.md) and [Rules API](../tutorials/advanced-strategies/rules-api.md) tutorials if you need to understand them.

### A quick comparison

| API         |  Apply few strats  | Apply many strats | Multiple `.apply` allowed | Multiple columns | Statements with `&` | Discrete strats |
| ----------- | ------------------ | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- | 
| Sequential  | :material-check:   | :material-close: | :material-check: | :material-close: | :material-close: | :material-close: |
| Dict        | :material-check:   | :material-check: | :material-close: | :material-check: | :material-close: | :material-close: |
| Rules       | :material-check:   | :material-check: | :material-close: | :material-check: | :material-check: | :material-check: |

### Do they Differ?

The Sequential API is actually a special case of the Dict API. Although multiple `apply` statements can be made with the Sequential API in reality what is happening is that a hidden dictionary is being populated under a "default" key.

The Rules API is entirely different. The `Rules` class is actually a subclass of a Python `tuple` with addtional validation rules that prevent any member being added to it that is not an `on` function. Actually, to be more specific, an instance of an `On` class which the `on` function wraps around.

The `On` class (not available via the **Enlace** package) is what implements the "and" combinations. It actually achieves this by self mutation whereby multiple combinations of `On` instances mutate the *first* one that precedes any chain of instances linked with `&`. 

Whilst effective, one of the consequences of this is that the contents of the `Rules` tuple *do* get mutated. — You may recall that technically a tuple is immutable so this is a real gray area in what is allowed / not allowed across other programming languages.

Check out the code if this interests you!