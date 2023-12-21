import pytest
from santa2023.identities import replace_moves

@pytest.mark.parametrize(
    "permutation, moves1, moves2, expected_permutation",
    [
        (['f1', 'f2', 'f3'], ['r1', 'r2'], ['d1', 'd2'], None),
        (['f1', 'f2', 'f3'], [], ['d1', 'd2'], None),
        (['f1', 'f2', 'f3'], ['f1', 'f2'], [], ['f3']),
        (['f1', 'f2', 'f3'], ['f1', 'f2', 'f3'], [], []),
        (['f1', 'f2', 'f3'], ['f1'], ['d1'], ['d1', 'f2', 'f3']),
        (['f1', 'f2', 'f3'], ['f2'], ['d2'], ['f1', 'd2', 'f3']),
        (['f1', 'f2', 'f3'], ['f3'], ['d1', 'd2'], ['f1', 'f2', 'd1', 'd2']),
        (['f1', 'f2', 'f3'], ['f1', 'f2'], ['d1', 'd2'], ['d1', 'd2', 'f3']),
        (['r1', '-f1', 'f1', '-f1'], ('f1', '-f1'), (), ['r1', '-f1']),
    ]
)
def test_replace_moves(permutation, moves1, moves2, expected_permutation):
    assert replace_moves(permutation, moves1, moves2) == expected_permutation