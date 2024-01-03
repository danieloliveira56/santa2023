import json
from random import choices

from santa2023.utils import get_inverse


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


class Permutation:
    def __init__(self, mapping, move_ids=[]):
        if isinstance(move_ids, str):
            self._move_ids = [move_ids]
        else:
            self._move_ids = move_ids
        self._mapping = tuple(mapping)

    @property
    def move_ids(self):
        return self._move_ids

    def split(self, puzzle_info):
        return [Permutation(puzzle_info[move_id], move_id) for move_id in self.move_ids]

    @property
    def mapping(self):
        return self._mapping

    def __mul__(self, other):
        return Permutation(
            [self.mapping[i] for i in other.mapping],
            self.move_ids + other.move_ids,
        )

    def __invert__(self):
        inv_moves = []
        for id in reversed(self.move_ids):
            if id.startswith("-"):
                inv_moves.append(id[1:])
            else:
                inv_moves.append("-" + id)
        return Permutation(
            get_inverse(self.mapping),
            inv_moves,
        )

    def __len__(self):
        return len(self.move_ids)

    def __lt__(self, other):
        if len(self) < len(other):
            return True
        # elif len(self) == len(other):
        #     return self._move_ids > other._move_ids
        else:
            return False

    def __le__(self, other):
        if len(self) < len(other):
            return True
        elif len(self) == len(other):
            return self._move_ids >= other._move_ids
        else:
            return False

    def __repr__(self):
        return f"{'.'.join(self.move_ids)}"

    def __eq__(self, other):
        return self.mapping == other.mapping


class Puzzle:
    def __init__(self, id, puzzle_type, solution, initial, num_wildcards):
        self._id = int(id)
        self._type = puzzle_type
        self._initial = initial.split(";")
        self._current = initial.split(";")
        self._solution = solution.split(";")
        self._num_wildcards = int(num_wildcards)
        self._permutations = []

    def initialize_move_list(self, allowed_moves):
        self._allowed_moves = {}
        for id, permutation in allowed_moves.items():
            self._allowed_moves[id] = permutation.copy()
            self._allowed_moves[f"-{id}"] = get_inverse(permutation)

    @property
    def allowed_move_ids(self):
        return list(self._allowed_moves.keys())

    def random_solution(self, size):
        return list(choices(self.allowed_move_ids, k=size))

    def permutate(self, move_id):
        self._permutations.append(move_id)
        permutation = self._allowed_moves[move_id]
        self._current = [self._current[i] for i in permutation]
        return self

    def full_permutation(self, permutation_list):
        for move_id in permutation_list:
            self.permutate(move_id)
        return self

    def clone(self):
        cloned_puzzle = Puzzle(self._id, self._type, "", "", self._num_wildcards)
        cloned_puzzle._solution = self._solution.copy()
        cloned_puzzle._current = self._current.copy()
        cloned_puzzle._initial = self._initial.copy()
        cloned_puzzle._allowed_moves = self._allowed_moves
        assert self._current == self._initial
        return cloned_puzzle

    @property
    def score(self):
        return (
            2 * len(self._current) * max(0, self.count_mismatches - self._num_wildcards)
            + len(self)
            + (0 if self.is_solved else 2)
        )

    @property
    def count_mismatches(self):
        return sum([c1 != c2 for c1, c2 in zip(self._current, self._solution)])

    @property
    def permutations(self):
        return self._permutations.copy()

    @property
    def type(self):
        return self._type

    @property
    def is_solved(self):
        return self.count_mismatches <= self._num_wildcards
        # return self._current == self._solution

    @property
    def submission(self):
        return f"{self._id},{'.'.join(self._permutations)}"

    @property
    def current_pattern_hash(self):
        return tuple(self._current).__hash__()

    def __len__(self):
        return len(self._permutations)

    def __getitem__(self, item):
        return self._permutations[item]

    def __repr__(self):
        return (
            "----------\n"
            f"{self._id}: "
            f"{self._type} "
            f"[{self._num_wildcards}]\n"
            f"{''.join(self._initial)}\n"
            f"{''.join(self._current)}[{self.score}]\n"
            f"{''.join(self._solution)}\n"
            f"{self.submission}\n"
            "----------"
        )
