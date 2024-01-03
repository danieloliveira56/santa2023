import os

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
