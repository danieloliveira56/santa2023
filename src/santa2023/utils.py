import json
import os
from pathlib import Path
from typing import List

PUZZLE_TYPES = [
    "all",
    "cube",
    "wreath",
    "globe",
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
]

CACHE_BASE_PATH = Path(__file__).parent.parent.parent / ".cache"


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
            for id, moves in solution.items():
                f.write(f"{id},{'.'.join(moves)}\n")
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


def move_letter(move):
    return move.replace("-", "")[0]


def move_idx(move):
    return int(move.replace("-", "")[1:])


def cache_translation(fun):
    cache_translation.cache_ = {}

    def inner(puzzle_type, allowed_moves):
        if puzzle_type in cache_translation.cache_:
            return cache_translation.cache_[puzzle_type]

        if not CACHE_BASE_PATH.exists():
            CACHE_BASE_PATH.mkdir()
        cache_path = (
            CACHE_BASE_PATH
            / f"{puzzle_type.replace('/', '_')}_rotation_translation.json"
        )
        if cache_path.exists():
            with open(cache_path, "r") as f:
                cache_translation.cache_[puzzle_type] = json.load(f)
        else:
            translation = fun(puzzle_type, allowed_moves)
            with open(cache_path, "w") as f:
                json.dump(translation, f)
            cache_translation.cache_[puzzle_type] = translation
        return cache_translation.cache_[puzzle_type]

    return inner


def sorted_solution(puzzle, solution: List[str]):
    """
    Sorts commutative moves in a solution
    :param puzzle:
    :param solution: Sequence of moves for the puzzle
    :return:
    """
    sorting_key = cube_sorting_key
    if puzzle.type.startswith("wreath"):
        sorting_key = default_sorting_key

    if puzzle.type.startswith("globe"):
        solution = [move.replace("-f", "f") for move in solution]
        longitude_size = int(puzzle.type.split("_")[1].split("/")[1])
        current_group = solution[:1]
        new_solution = []
        for move in solution[1:]:
            if move_letter(move) == move_letter(current_group[-1]) == "r":
                current_group.append(move)
            elif (
                move_letter(move) == move_letter(current_group[-1]) == "f"
                and move_idx(move) % longitude_size
                == move_idx(current_group[-1]) % longitude_size
            ):
                current_group.append(move)
            else:
                new_solution += sorted(current_group, key=sorting_key)
                current_group = [move]

        new_solution += sorted(current_group, key=sorting_key)
        return new_solution

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


def debug_list(l, row_width=50, start=None, end=None):
    if start is None:
        start = 0
    if end is None:
        end = len(l)
    for i in range(start, end, row_width):
        print(".".join(l[i : min(len(l), i + row_width)]))
        print(" ".join([f"{j:{len(str(l[j]))}d}" for j in range(i, min(len(l), i + row_width))]))

    # print(".".join(l[start:end]))


def clean_solution(puzzle, solution):
    if puzzle.type.startswith("cube"):
        return clean_cube_solution(puzzle, solution)
    elif puzzle.type.startswith("globe"):
        return clean_globe_solution(puzzle, solution)
    else:
        return sorted_solution(puzzle, solution)


def print_globe(globe_type):
    print(globe_type)
    lat_size = int(globe_type.split("_")[1].split("/")[0])
    lon_size = int(globe_type.split("_")[1].split("/")[1])
    print(f"Lat: {lat_size}, Lon: {lon_size}")
    w = 3
    for i in range(lat_size + 1):
        for j in range(2 * lon_size):
            print(f"{j + 2 * i * lon_size:{w}d}", end=" ")
        print()


def clean_cube_solution(puzzle, solution):
    """
    Removes trivially redundant moves from a cube solution
    """

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


def clean_globe_solution(puzzle, solution):
    """
    Removes trivially redundant moves from a globe solution
    """
    latitude_size = int(puzzle.type.split("_")[1].split("/")[0])
    longitude_size = int(puzzle.type.split("_")[1].split("/")[1])

    str_solution = "." + ".".join(solution) + "."
    old_str_solution = ""

    while old_str_solution != str_solution:
        old_str_solution = str_solution
        for i in range(latitude_size + 1):
            str_solution = str_solution.replace(f".r{i}.-r{i}.", ".")
            str_solution = str_solution.replace(f".-r{i}.r{i}.", ".")
            str_solution = str_solution.replace(
                "." + ".".join([f"r{i}"] * 2 * (longitude_size)) + ".", "."
            )
            str_solution = str_solution.replace(
                "." + ".".join([f"-r{i}"] * 2 * (longitude_size)) + ".", "."
            )
        for i in range(2 * longitude_size):
            str_solution = str_solution.replace(f".-f{i}.", f".f{i}.")
            str_solution = str_solution.replace(f".f{i}.f{i}.", ".")

        solution = sorted_solution(
            puzzle,
            str_solution[1:-1].split("."),
        )
        str_solution = "." + ".".join(solution) + "."

    return str_solution[1:-1].split(".")
