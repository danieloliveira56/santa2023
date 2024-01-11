import pytest

from santa2023.identities import replace_moves
from santa2023.puzzle import Permutation, read_puzzle_info
from santa2023.utils import CSV_BASE_PATH, get_inverse, sorted_solution

all_puzzle_info = read_puzzle_info(CSV_BASE_PATH / "puzzle_info.csv")
for puzzle_info in all_puzzle_info.values():
    move_ids = list(puzzle_info.keys())
    for id in move_ids:
        puzzle_info[f"-{id}"] = get_inverse(puzzle_info[id])


@pytest.mark.parametrize(
    "permutation, moves1, moves2, expected_permutation",
    [
        (["f1", "f2", "f3"], ["r1", "r2"], ["d1", "d2"], None),
        (["f1", "f2", "f3"], [], ["d1", "d2"], None),
        (["f1", "f2", "f3"], ["f1", "f2"], [], ["f3"]),
        (["f1", "f2", "f3"], ["f1", "f2", "f3"], [], []),
        (["f1", "f2", "f3"], ["f1"], ["d1"], ["d1", "f2", "f3"]),
        (["f1", "f2", "f3"], ["f2"], ["d2"], ["f1", "d2", "f3"]),
        (["f1", "f2", "f3"], ["f3"], ["d1", "d2"], ["f1", "f2", "d1", "d2"]),
        (["f1", "f2", "f3"], ["f1", "f2"], ["d1", "d2"], ["d1", "d2", "f3"]),
        (["r1", "-f1", "f1", "-f1"], ("f1", "-f1"), (), ["r1", "-f1"]),
    ],
)
def test_replace_moves(permutation, moves1, moves2, expected_permutation):
    assert replace_moves(permutation, moves1, moves2) == expected_permutation


@pytest.mark.parametrize(
    "puzzle_type, permutation1, permutation2",
    [
        ("cube_2/2/2", ("-d0", "-d1", "-d0", "-d0"), ("-d1", "d0")),
        ("cube_2/2/2", ("-d0", "-d1", "d0", "-r1"), ("-d1", "-r1")),
        ("cube_2/2/2", ("-d1", "-d0", "-d1", "d0"), ("d1", "d1")),
        ("cube_2/2/2", ("d0", "-f0", "-f1", "-r0"), ("-f1", "-f0")),
        (
            "cube_3/3/3",
            ("-r0", "-r0", "-r1", "r2", "-r1", "-r0", "r2"),
            ("-r0", "-r0", "-r0", "-r1", "r2", "-r1", "r2"),
        ),
        (
            "cube_3/3/3",
            ("-r0", "-r0", "-r1", "r2", "-r1", "-r0", "r2"),
            ("-r0", "-r0", "-r0", "-r1", "-r1", "r2", "r2"),
        ),
        (
            "cube_3/3/3",
            ("-r0", "-r0", "-r1", "r2", "-r1", "-r0", "r2"),
            ("r0", "r1", "r1", "r2", "r2"),
        ),
        ("cube_4/4/4", ("r3", "-r1", "r3", "r3"), ("-r3", "-r1")),
        (
            "cube_19/19/19",
            (
                "r5",
                "r11",
                "r3",
                "-r13",
                "-r15",
                "r4",
                "-r18",
                "r11",
                "r3",
                "-r13",
                "-r15",
                "r4",
                "-r18",
                "-r18",
                "-r1",
            ),
            (
                "-r1",
                "r3",
                "r3",
                "r4",
                "r4",
                "r5",
                "r11",
                "r11",
                "-r13",
                "-r13",
                "-r15",
                "-r15",
                "-r18",
                "-r18",
                "-r18",
            ),
        ),
        (
            "cube_19/19/19",
            (
                "r5",
                "r11",
                "r3",
                "-r13",
                "-r15",
                "r4",
                "-r18",
                "r11",
                "r3",
                "-r13",
                "-r15",
                "r4",
                "-r18",
                "-r18",
                "-r1",
            ),
            (
                "-r1",
                "r3",
                "r3",
                "r4",
                "r4",
                "r5",
                "r11",
                "r11",
                "-r13",
                "-r13",
                "-r15",
                "-r15",
                "r18",
            ),
        ),
        (
            "cube_4/4/4",
            (
                "f0",
                "f0",
                "f2",
                "f2",
                "f3",
                "f3",
                "r3",
                "r3",
            ),
            (
                "f1",
                "f1",
                "r0",
                "r0",
                "f0",
                "f0",
                "f1",
                "f1",
                "f2",
                "f2",
                "f3",
                "f3",
            ),
        ),
        (
            "cube_4/4/4",
            (
                "f0",
                "f0",
                "f2",
                "f2",
                "f3",
                "f3",
                "r3",
                "r3",
                "f2",
                "f2",
                "f3",
                "f3",
            ),
            (
                "f1",
                "f1",
                "r0",
                "r0",
                "f0",
                "f0",
                "f1",
                "f1",
            ),
        ),
        (
            "cube_2/2/2",
            "d0.d0.f0",
            "d1.d1.-f1.d0.d0.d1.d1",
        ),
        (
            "cube_2/2/2",
            "d0.d0.-f0",
            "d1.d1.f1.d0.d0.d1.d1",
        ),
        (
            "cube_2/2/2",
            "d0.d0.f1",
            "d1.d1.-f0.d0.d0.d1.d1",
        ),
        (
            "cube_2/2/2",
            "d0.d0.r0",
            "d1.d1.-r1.d0.d0.d1.d1",
        ),
        (
            "cube_2/2/2",
            "d0.d0.-r0",
            "d1.d1.r1.d0.d0.d1.d1",
        ),
        (
            "cube_2/2/2",
            "d0.d0.r1",
            "d1.d1.-r0.d0.d0.d1.d1",
        ),
        (
            "cube_2/2/2",
            "d0.d0.d0",
            "d1.d1.d0.d0.d0.d1.d1",
        ),
        (
            "cube_2/2/2",
            "d0.d0.d1",
            "d1.d1.d1.d0.d0.d1.d1",
        ),
        (
            "cube_2/2/2",
            "f0.f0.d0",
            "f1.f1.-d1.f0.f0.f1.f1",
        ),
        (
            "cube_2/2/2",
            "r0.r0.d0",
            "r1.r1.-d1.r0.r0.r1.r1",
        ),
        (
            "cube_2/2/2",
            "r0.r0.f0",
            "r1.r1.-f1.r0.r0.r1.r1",
        ),
        (
            "cube_2/2/2",
            "f0.f0.r0",
            "f1.f1.-r1.f0.f0.f1.f1",
        ),
        (
            "cube_2/2/2",
            "f0.r1.f1.d0.d0.-f0.-r0.f0.d0",
            "f0.r1.f1.d1.d1.f1.r1.-f1.-d0.d1.d1",
        ),
        (
            "cube_4/4/4",
            "f0.f0.f2.f2.f3.f3.r3.r3.f2.f2.f3.f3.d0",
            "f1.f1.r0.r0.f2.f2.f3.f3.-d3.f0.f0.f1.f1.f2.f2.f3.f3",
        ),
        (
            "cube_4/4/4",
            "d0.d0.d2.d2.d3.d3.r3.r3.d2.d2.d3.d3.f0.f1",
            "d1.d1.r0.r0.d2.d2.d3.d3.-f3.-f2.d0.d0.d1.d1.d2.d2.d3.d3",
        ),
        (
            "cube_4/4/4",
            "r0.r0.r2.r2.r3.r3.f3.f3.r2.r2.r3.r3.d0",
            "r1.r1.f0.f0.r2.r2.r3.r3.-d3.r0.r0.r1.r1.r2.r2.r3.r3",
        ),
        (
            "cube_5/5/5",
            "f3.f4.r0.-f3.-f4",
            "-f0.-f1.-f2.-d4.f0.f1.f2",
        ),
        (
            "cube_5/5/5",
            "f0.f1.f2.f3.f4.r0",
            "-d4.f0.f1.f2.f3.f4",
        ),
        (
            "cube_3/3/3",
            "f0.f1.f2.d0",
            "r0.f0.f1.f2",
        ),
        (
            "cube_3/3/3",
            "f0.f1.f2.-d0",
            "-r0.f0.f1.f2",
        ),
        (
            "cube_3/3/3",
            "f0.f1.f2.r0",
            "-d2.f0.f1.f2",
        ),
        (
            "cube_3/3/3",
            "-f0.-f1.-f2.d0",
            "-r2.-f0.-f1.-f2",
        ),
        (
            "cube_3/3/3",
            "-f0.f2.-r2.d0.-r1.r2.-d2.-r0.f0.-d1.f2.-r0.-d1.-r1.-f0.-d2.f0.d2.-f0.-r0.f0.f0.r0.d1.-f0.-f2",
            "-f0.f2.-r2.d0.-r1.r2.-d2.-r0.f0.-d1.f2.-r0.-d1.-r1.-f0.-d2.f0.d2.-f0.-r0.f0.f0.r0.d1.-f0.-f2",
        ),
    ],
)
def test_equivalency(puzzle_type, permutation1, permutation2):
    if isinstance(permutation1, str):
        permutation1 = permutation1.split(".")
    if isinstance(permutation2, str):
        permutation2 = permutation2.split(".")
    puzzle_info = all_puzzle_info[puzzle_type]
    n = len(list(puzzle_info.values())[0])
    result1 = list(range(n))
    for move_id in permutation1:
        result1 = [result1[i] for i in puzzle_info[move_id]]
    result2 = list(range(n))
    for move_id in permutation2:
        result2 = [result2[i] for i in puzzle_info[move_id]]

    assert result1 == result2


# @pytest.mark.parametrize(
#         "b",
#     ["", "-"]
# )
# @pytest.mark.parametrize(
#      "a",
#         [
#             "f0",
#             "f1",
#             "f2",
#             "f3",
#             "f4",
#             "r0",
#             "r1",
#             "r2",
#             "r3",
#             "r4",
#             "d0",
#             "d1",
#             "d2",
#             "d3",
#             "d4",
#         ],
# )
# def test_equivalency2(a, b):
#     puzzle_info = all_puzzle_info["cube_5/5/5"]
#     n = len(list(puzzle_info.values())[0])
#
#     permutation1 = ["f3", "f4", "r0", "-f3", "-f4"]
#
#     result1 = list(range(n))
#     for move_id in permutation1:
#         result1 = [result1[i] for i in puzzle_info[move_id]]
#
#     permutation2 = ["-f0", "-f1", "-f2", b+a, "f0", "f1", "f2"]
#     result2 = list(range(n))
#     for move_id in permutation2:
#         result2 = [result2[i] for i in puzzle_info[move_id]]
#     assert result1 == result2


def test_permutation_class():
    p1 = Permutation([0, 1, 2, 3, 4], "r0")
    p2 = Permutation([1, 2, 3, 4, 0], "r1")

    p3 = p1 * p2

    assert p1.move_ids == ["r0"]
    assert (~p1).move_ids == ["-r0"]
    assert p2.move_ids == ["r1"]
    assert (~p2).move_ids == ["-r1"]
    assert p3.move_ids == ["r0", "r1"]
    assert (~p3).move_ids == ["-r1", "-r0"]
    assert (~~p3).move_ids == ["r0", "r1"]

    assert p1.mapping == (0, 1, 2, 3, 4)
    assert p2.mapping == (1, 2, 3, 4, 0)
    assert p3.mapping == (1, 2, 3, 4, 0)

    p4 = p3 * p2
    assert p4.mapping == (2, 3, 4, 0, 1)

    puzzle_info = all_puzzle_info["cube_2/2/2"]
    f0 = Permutation(puzzle_info["f0"], "f0")
    f0_ = Permutation(puzzle_info["-f0"], "-f0")
    f1 = Permutation(puzzle_info["f1"], "f1")
    f1_ = Permutation(puzzle_info["-f1"], "-f1")
    r0 = Permutation(puzzle_info["r0"], "r0")
    r0_ = Permutation(puzzle_info["-r0"], "-r0")
    r1 = Permutation(puzzle_info["r1"], "r1")
    r1_ = Permutation(puzzle_info["-r1"], "-r1")
    d0 = Permutation(puzzle_info["d0"], "d0")
    d0_ = Permutation(puzzle_info["-d0"], "-d0")
    d1 = Permutation(puzzle_info["d1"], "d1")
    d1_ = Permutation(puzzle_info["-d1"], "-d1")

    assert (f0 * f0_).mapping == tuple(range(24))
    assert (f0 * f0) == (f0_ * f0_)

    assert (d0_ * d1_ * d0_ * d0_) == (d1_ * d0)

    for x, x_ in ((f0, f0_), (f1, f1_), (r0, r0_), (r1, r1_), (d0, d0_), (d1, d1_)):
        assert ~x == x_
        assert ~x_ == x
        assert ~~x == x
        assert ~~x_ == x_
        assert ~(x * x) == (~x) * (~x)
        assert ~(x * x_) == x * (~x)


@pytest.mark.parametrize(
    "solution, expected_solution",
    [
        ("r1.r2.-r0", "-r0.r1.r2"),
        (
            "f0.f0.r2.r2.f0.d2.d2.f0.d2.-r0.d2.f0.-r2.-f2.-d2.f2.f2.-d0.r1.r1.f2.f2.d0.f2.f2.-r0.r1.r2.-d1.-d2.d0",
            "f0.f0.r2.r2.f0.d2.d2.f0.d2.-r0.d2.f0.-r2.-f2.-d2.f2.f2.-d0.r1.r1.f2.f2.d0.f2.f2.-r0.r1.r2.d0.-d1.-d2",
        ),
    ],
)
def test_sotution_sorting(solution, expected_solution):
    if isinstance(solution, str):
        solution = solution.split(".")
    if isinstance(expected_solution, str):
        expected_solution = expected_solution.split(".")

    result = sorted_solution(
        solution,
        sorting_key=lambda x: (-0.1 if x.startswith("-") else 0.1)
        + int(x.replace("-", "")[1:]),
    )

    assert result == expected_solution


# @pytest.mark.parametrize(
#     "fdr, fdr_original, fdr_result",
#     [
#         (fdr, original, result)
#         for fdr, rules in quarter_turns.items()
#         for original, result in rules.items()
#     ],
# )
# def test_rotation_rules(fdr, fdr_original, fdr_result):
#     puzzle_info = all_puzzle_info["cube_3/3/3"]
#     n = 3
#
#     permutation1 = [f"{fdr}{i}" for i in range(n)] + [fdr_original]
#     permutation2 = [fdr_result] + [f"{fdr}{i}" for i in range(n)]
#     result1 = list(range(54))
#     result2 = list(range(54))
#
#     for move_id in permutation1:
#         result1 = [result1[i] for i in puzzle_info[move_id]]
#     for move_id in permutation2:
#         result2 = [result2[i] for i in puzzle_info[move_id]]
#
#     assert result1 == result2
