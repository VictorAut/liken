"""Unit tests for the shared block-wise pair generator helper."""

import pytest

from liken.dedupers._blockwise import DEFAULT_BLOCK_MEMORY
from liken.dedupers._blockwise import block_rows_for
from liken.dedupers._blockwise import resolve_workers
from liken.dedupers._blockwise import similar_pairs_blockwise


##########################
# block size resolution #
#########################


@pytest.mark.parametrize(
    "n, itemsize, cap, expected",
    [
        (1000, 4, 64 * 1024 * 1024, 16777),  # 64 MiB / (1000 * 4) floors to 16777
        (10, 4, 64 * 1024 * 1024, 1_677_721),  # larger than n: fine, loop clamps
        (1000, 4, 4000, 1),  # cap / 4000 floors to 1
        (1000, 4, 4001, 1),  # just over one row's band still floors to 1
        (1000, 4, 8000, 2),
    ],
    ids=["sixty-four-mib", "cap-exceeds-n", "cap-floors-to-one", "just-over-one-band", "two-rows"],
)
def test_block_rows_for_floors_to_whole_rows(n, itemsize, cap, expected):
    assert block_rows_for(n, itemsize, cap) == expected


@pytest.mark.parametrize(
    "n, itemsize, cap",
    [(0, 4, 1024), (1000, 0, 1024), (1000, 4, 0), (1000, 4, -5)],
    ids=["zero-rows", "zero-itemsize", "zero-cap", "negative-cap"],
)
def test_block_rows_for_clamps_degenerate_values_to_one_row(n, itemsize, cap):
    """A cap below one row's score band clamps to a one-row block, never raises."""
    assert block_rows_for(n, itemsize, cap) == 1


#######################
# workers resolution #
######################


def test_resolve_workers_explicit_integer_wins():
    assert resolve_workers(4, distributed=False) == 4
    assert resolve_workers(4, distributed=True) == 4


def test_resolve_workers_negative_one_explicit_wins():
    assert resolve_workers(-1, distributed=False) == -1
    assert resolve_workers(-1, distributed=True) == -1


def test_resolve_workers_none_local_uses_all_cores():
    import os

    assert resolve_workers(None, distributed=False) == os.cpu_count()


def test_resolve_workers_none_distributed_is_single_threaded():
    assert resolve_workers(None, distributed=True) == 1


def test_resolve_workers_passes_validated_values_through():
    """Validation belongs to the constructing deduper; resolution is a lookup."""
    assert resolve_workers(0, distributed=False) == 0
    assert resolve_workers(7, distributed=True) == 7


##############################
# pair generation correctness#
#############################


def _stub_scores(s1: str, s2: str, score_cutoff: float | None = None) -> float:
    """Deterministic scorer: sum of lengths. Pure Python, order visible."""
    return float(len(s1) + len(s2))


def _ref_pairs(scored, bound, *, low=True):
    """Brute-force reference in yield order: upper triangle, row-major."""
    pairs = []
    for pos in range(len(scored) - 1):
        for offset in range(1, len(scored) - pos):
            score = _stub_scores(scored[pos], scored[pos + offset])
            if (low and score <= bound) or (not low and score >= bound):
                pairs.append((pos, pos + offset))
    return pairs


def test_similar_pairs_blockwise_pair_order_matches_reference_low_bound():
    """The block shape yields the same pairs in the same order as a brute scan."""
    scored = ["a", "bb", "ccc", "dddd", "e", "ff", "ggg", "hhhh"]
    bound = 6  # len sums <= 6 match

    got = list(
        similar_pairs_blockwise(
            scored,
            compare=_stub_scores,
            low=True,
            bound=float(bound),
            block_rows=2,
            workers=1,
        )
    )

    assert got == _ref_pairs(scored, bound, low=True)


def test_similar_pairs_blockwise_pair_order_matches_reference_high_bound():
    scored = ["a", "bb", "ccc", "dddd", "e", "ff", "ggg", "hhhh"]
    bound = 6

    got = list(
        similar_pairs_blockwise(
            scored,
            compare=_stub_scores,
            low=False,
            bound=float(bound),
            block_rows=3,
            workers=1,
        )
    )

    assert got == _ref_pairs(scored, bound, low=False)


def test_similar_pairs_blockwise_single_row_batched_call():
    """One strip covering all rows degenerates into the batched shape."""
    scored = ["a", "bb", "ccc"]

    got = list(
        similar_pairs_blockwise(
            scored,
            compare=_stub_scores,
            low=True,
            bound=5.0,
            block_rows=10,
            workers=1,
        )
    )

    assert got == _ref_pairs(scored, 5, low=True)


def test_similar_pairs_blockwise_fewer_than_two_values_yields_nothing():
    assert (
        list(
            similar_pairs_blockwise(
                ["a"],
                compare=_stub_scores,
                low=True,
                bound=10.0,
                block_rows=5,
                workers=1,
            )
        )
        == []
    )
    assert (
        list(
            similar_pairs_blockwise(
                [],
                compare=_stub_scores,
                low=True,
                bound=10.0,
                block_rows=5,
                workers=1,
            )
        )
        == []
    )


def test_similar_pairs_blockwise_preserves_original_indices():
    """Pairs come back in the index space of the caller's present list."""
    scored = ["a", "bb", "ccc", "dddd"]

    # every pair matches (sum of lengths >= 1)
    got = list(
        similar_pairs_blockwise(
            scored,
            compare=_stub_scores,
            low=False,
            bound=0.0,
            block_rows=2,
            workers=1,
        )
    )

    assert got == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def test_similar_pairs_blockwise_matches_rapidfuzz_cdist_scores():
    """The helper's scores match a direct rapidfuzz cdist call per block."""
    from rapidfuzz import fuzz
    from rapidfuzz import process

    scored = ["london", "londn", "londonn", "paris", "pariss"]
    threshold = 80.0

    expected = []
    for pos in range(len(scored) - 1):
        scores = process.cdist([scored[pos]], scored[pos + 1 :], scorer=fuzz.ratio)[0]
        for offset, score in enumerate(scores):
            if score >= threshold:
                expected.append((pos, pos + 1 + offset))

    got = list(
        similar_pairs_blockwise(
            scored,
            compare=fuzz.ratio,
            low=False,
            bound=threshold,
            block_rows=2,
            workers=1,
        )
    )

    assert got == expected


def test_default_block_memory_is_sixty_four_mib():
    assert DEFAULT_BLOCK_MEMORY == 64 * 1024 * 1024
