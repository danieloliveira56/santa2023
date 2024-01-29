import json
from pathlib import Path

from santa2023.puzzle import Puzzle

CSV_BASE_PATH = Path(__file__).parent.parent.parent / "data"


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


PUZZLE_INFO = read_puzzle_info(CSV_BASE_PATH / "puzzle_info.csv")
PUZZLES = read_puzzles(CSV_BASE_PATH / "puzzles.csv")
for p in PUZZLES:
    p.initialize_move_list(PUZZLE_INFO[p.type])
