import os
from pathlib import Path
from typing import List

PUZZLE_TYPES = [
    "cube_2/2/2",
    "cube_3/3/3",
    "cube_4/4/4",
    "cube_5/5/5",
    "cube_6/6/6",
    "cube_7/7/7",
    "cube_8/8/8",
    "cube_9/9/9",
    "cube_10/10/10",
    "cube_19/19/19",
    "cube_33/33/33",
    "wreath_6/6",
    "wreath_7/7",
    "wreath_12/12",
    "wreath_21/21",
    "wreath_33/33",
    "wreath_100/100",
    "globe_1/8",
    "globe_1/16",
    "globe_2/6",
    "globe_3/4",
    "globe_6/4",
    "globe_6/8",
    "globe_6/10",
    "globe_3/33",
    "globe_33/3",
    "globe_8/25",
    "all",
]

CSV_BASE_PATH = Path(__file__).parent.parent.parent / "data"


def print_solution(solution, debug=False):
    print(".".join(solution))
    if debug:
        print(".".join([f"{i:{len(m)}d}" for i, m in enumerate(solution)]))


def get_inverse(permutation):
    return [
        i
        for i, _ in sorted(
            [(i, v) for i, v in enumerate(permutation)], key=lambda x: x[1]
        )
    ]


def read_solution(filename):
    with open(filename, "r") as f:
        f.readline()
        lines = f.readlines()
    # lines = sorted(lines, key)
    return {
        int(line.split(",")[0].strip()): line.split(",")[1].strip().split(".")
        for line in lines
    }


def export_solution(puzzles, solution):
    solution_score = calculate_score(solution)
    filename = f"submission_{calculate_score(solution):0_}"
    i = 0
    suffix = ""
    while os.path.exists(f"{filename}{suffix}.csv"):
        i += 1
        suffix = f"({i})"
    filename = f"{filename}{suffix}.csv"

    print(f"Saving solution of score {solution_score} as '{filename}'")
    with open(filename, "w") as f:
        f.write("id,moves\n")
        if isinstance(solution, dict):
            for p in puzzles:
                f.write(f"{p._id},{'.'.join(solution[p._id])}\n")
        else:
            for p, permutation in zip(puzzles, solution):
                f.write(f"{p._id},{'.'.join(permutation)}\n")


def calculate_score(solution):
    if isinstance(solution, dict):
        return sum(len(p) for p in solution.values())
    else:
        return sum(len(p) for p in solution)


def remove_identity(permutation):
    for i in range(len(permutation) - 1, 0):
        if (
            permutation[i] == f"-{permutation[i + 1]}"
            or permutation[i + 1] == f"-{permutation[i]}"
        ):
            permutation.pop(i)
            permutation.pop(i + 1)
    return permutation


def default_sorting_key(x):
    return x.replace("-", "")


def cube_sorting_key(x):
    k = int(x.replace("-", "")[1:])
    if x.startswith("-"):
        k -= 0.1
    return k


def sorted_solution(puzzle, solution: List[str]):
    """
    Sorts commutative moves in the solution of cube and wreath puzzles
    :param puzzle:
    :param solution: Sequence of moves for the puzzle
    :return:
    """
    if puzzle.type.startswith("globe"):
        return [move_id.replace("-f", "f") for move_id in solution[puzzle._id]]

    sorting_key = default_sorting_key
    if puzzle.type.startswith("cube"):
        sorting_key = cube_sorting_key

    current_group = solution[:1]
    new_solution = []
    for move in solution[1:]:
        if move.replace("-", "")[0] == current_group[-1].replace("-", "")[0]:
            current_group.append(move)
        else:
            new_solution += sorted(current_group, key=sorting_key)
            current_group = [move]
    new_solution += sorted(current_group, key=sorting_key)
    return new_solution


def clean_cube_solution(puzzle, solution):
    solution = sorted_solution(
        puzzle,
        solution,
    )
    str_solution = "." + ".".join(solution) + "."
    old_str_solution = ""
    cube_size = int(puzzle.type.split("/")[1])

    while old_str_solution != str_solution:
        old_str_solution = str_solution
        for fdr in "fdr":
            for i in range(cube_size):
                str_solution = str_solution.replace(
                    f".{fdr}{i}.{fdr}{i}.{fdr}{i}.", f".-{fdr}{i}."
                )
                str_solution = str_solution.replace(
                    f".-{fdr}{i}.-{fdr}{i}.-{fdr}{i}.", f".{fdr}{i}."
                )
                str_solution = str_solution.replace(f".-{fdr}{i}.{fdr}{i}.", ".")
                str_solution = str_solution.replace(f".{fdr}{i}.-{fdr}{i}.", ".")
                str_solution = str_solution.replace(
                    f".-{fdr}{i}.-{fdr}{i}.", f".{fdr}{i}.{fdr}{i}."
                )
                str_solution = str_solution.replace(
                    f".{fdr}{i}.{fdr}{i}.{fdr}{i}.{fdr}{i}.", "."
                )
        solution = sorted_solution(
            puzzle,
            str_solution[1:-1].split("."),
        )
        str_solution = "." + ".".join(solution) + "."

    return str_solution[1:-1].split(".")
