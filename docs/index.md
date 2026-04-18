<style>
.md-content .md-typeset h1 { display: none; }
</style>

<p align="center">
  <a href="https://victoraut.github.io/liken/">
    <img src="images/logo-name-dark.png#only-dark" alt="Liken">
    <img src="images/logo-name-light.png#only-light" alt="Liken">
  </a>
</p>

<p align="center">
<a href="https://pypi.python.org/pypi/liken"><img height="20" alt="PyPI Version" src="https://img.shields.io/pypi/v/liken"></a>
<img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/liken">
<img height="20" alt="PyPI Downloads" src="https://static.pepy.tech/badge/liken">
<img height="20" alt="Tests" src="https://img.shields.io/github/actions/workflow/status/VictorAut/liken/python-validation.yml?label=CI">
<img height="20" alt="Coverage" src="https://img.shields.io/codecov/c/github/VictorAut/liken">
<img height="20" alt="License" src="https://img.shields.io/github/license/VictorAut/liken">
</p>


***
**Source Code**: [https://github.com/VictorAut/liken](https://github.com/VictorAut/liken)
***

<div class="liken-definition">
  <em>Liken</em>:<br>
  <small><i>phrasal verb</i></small><br>
  <small>/ˈlaɪ.kən/</small><br>
  <strong>to say that something is similar to or has the same qualities as something else</strong>
</div>

**Liken** is a library providing enhanced deduplication tooling for DataFrames.

The key features are:

- Near deduplication
- Ready-to-use deduplication methods
- Record linkage and canonicalization
- Rules-based deduplication
- Pandas, Polars and PySpark support
- Customizable in pure Python

## Installation

<!-- termynal -->
```
$ pip install liken
---> 100%
```

## Use `liken` In Your Code

```python
import liken as lk

df = ... # e.g. read data

df = (
    lk.dedupe(df)
    .apply(lk.fuzzy())
    .drop_duplicates("name")
    .collect()
)
```


## License

**Liken** is licensed under the [Apache-2.0 License](https://www.apache.org/licenses/LICENSE-2.0.html). See the [LICENSE](https://github.com/VictorAut/liken/blob/main/LICENSE) file for more details.