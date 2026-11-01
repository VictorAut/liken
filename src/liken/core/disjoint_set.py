"""Disjoint-set (union-find) over the integers 0..n-1.

Internal structure backing pair merging in the dedupers. Union by size
with full path compression gives near-constant amortised cost per
operation.
"""

from __future__ import annotations


class DisjointSet:
    """Union-find over the integers 0..n-1.

    The set starts with every element in its own component. `union(i, j)`
    merges the components holding `i` and `j`; it is a no-op when they are
    already merged. `ds[i]` returns the root of `i`'s component, and is
    valid for every `i` in the range. Roots are stable under merges of
    other components: a component's root only changes when it absorbs or
    is absorbed by another.
    """

    def __init__(self, n: int) -> None:
        """Create `n` singleton components, one per integer in 0..n-1."""
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        self._parent = list(range(n))
        self._size = [1] * n

    def __len__(self) -> int:
        return len(self._parent)

    def __getitem__(self, i: int) -> int:
        """Return the root of element `i`'s component."""
        root = i
        while self._parent[root] != root:
            root = self._parent[root]
        # Full path compression: re-point every node on the path at the root.
        while self._parent[i] != root:
            self._parent[i], i = root, self._parent[i]
        return root

    def union(self, i: int, j: int) -> None:
        """Merge the components holding `i` and `j`.

        No-op when the two already share a component. The root of the
        larger component wins the merge; on a size tie the first
        argument's root wins.
        """
        root_i = self[i]
        root_j = self[j]
        if root_i == root_j:
            return
        if self._size[root_i] < self._size[root_j]:
            root_i, root_j = root_j, root_i
        # root_i is the larger set's root: it becomes the new root.
        self._parent[root_j] = root_i
        self._size[root_i] += self._size[root_j]
