import pytest

from utils import get_inverse


@pytest.parametrize(
    "permutation, inverse",
    [([2, 7, 4, 9, 8, 3, 5, 0, 6, 1], [7, 9, 0, 5, 2, 6, 8, 1, 4, 3])],
)
def test_permutation_inverse(permutation, inverse):
    assert get_inverse(permutation) == inverse
