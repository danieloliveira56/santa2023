import json

from main import Puzzle


def get_inverse(permutation):
    return [
        i
        for i, _ in sorted(
            [(i, v) for i, v in enumerate(permutation)], key=lambda x: x[1]
        )
    ]


def read_puzzles(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
    return [Puzzle(*line.strip().split(",")) for line in lines[1:]]


def read_puzzle_info(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
    type_moves = [line.strip().split(",", maxsplit=1) for line in lines[1:]]
    return {
        type: json.loads(moves.strip('"').replace("'", '"'))
        for type, moves in type_moves
    }


def read_solution(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
    return [line.split(",")[1].strip().split(".") for line in lines[1:]]


def remove_identity(permutation):
    return permutation
    for i in range(len(permutation) - 1, 0):
        if (
            permutation[i] == f"-{permutation[i + 1]}"
            or permutation[i + 1] == f"-{permutation[i]}"
        ):
            permutation.pop(i)
            permutation.pop(i + 1)
    return permutation
