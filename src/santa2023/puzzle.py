import json
from random import choices

from santa2023.utils import get_inverse
import Levenshtein

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

    @property
    def size(self):
        return len(self._mapping)

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


def cube_taboo_list(size):
    taboo = {}
    for frd in "fdr":
        for i in range(size):
            # f1.-f1 not allowed
            taboo[f"{frd}{i}"] = [f"-{frd}{i}"]
            # -f1.f1 and -f1.-f1 not allowed
            taboo[f"-{frd}{i}"] = [f"{frd}{i}", f"-{frd}{i}"]

            # f1.f1.f1 not allowed
            taboo[(f"{frd}{i}", f"{frd}{i}")] = [f"{frd}{i}"]
            # -f1.-f1.-f1 not allowed
            taboo[(f"-{frd}{i}", f"-{frd}{i}")] = [f"-{frd}{i}"]

            for j in range(i):
                # f2.f1 not allowed
                taboo[f"{frd}{i}"].append(f"{frd}{j}")
                # f2.-f1 not allowed
                taboo[f"{frd}{i}"].append(f"-{frd}{j}")
                # -f2.f1 not allowed
                taboo[f"-{frd}{i}"].append(f"{frd}{j}")
                # -f2.-f1 not allowed
                taboo[f"-{frd}{i}"].append(f"-{frd}{j}")

    return taboo


def wreath_taboo_list(size):
    return {"r": ["-r"], "-r": ["r"], "l": ["-l"], "-l": ["l"]}


def globe_taboo_list(latitude_size, longitude_size):
    taboo = {}
    for i in range(latitude_size + 1):
        taboo[f"r{i}"] = [f"-r{i}"]
        taboo[f"-r{i}"] = [f"r{i}"]
    for i in range(2 * longitude_size):
        taboo[f"f{i}"] = [f"-f{i}", f"f{i}"]
        taboo[f"-f{i}"] = [f"-f{i}", f"f{i}"]
    return taboo


class Puzzle:
    def __init__(self, id, puzzle_type, solution, initial, num_wildcards):
        self._id = int(id)
        self._type = puzzle_type
        self._initial = initial.split(";")
        self._current = initial.split(";")
        self._solution = solution.split(";")
        self._num_wildcards = int(num_wildcards)
        self._permutations = []
        self._taboo = None

    def initialize_move_list(self, allowed_moves):
        self._allowed_moves = {}
        for id, permutation in allowed_moves.items():
            self._allowed_moves[id] = permutation.copy()
            self._allowed_moves[f"-{id}"] = get_inverse(permutation)

    @property
    def taboo_list(self):
        if self._taboo is None or len(self._taboo) == 0:
            if self.type.startswith("cube"):
                s = int(self.type.split("_")[1].split("/")[0])
                self._taboo = cube_taboo_list(s)
            elif self.type.startswith("wreath"):
                s = int(self.type.split("_")[1].split("/")[0])
                self._taboo = wreath_taboo_list(s)
            else:
                s1 = int(self.type.split("_")[1].split("/")[0])
                s2 = int(self.type.split("_")[1].split("/")[1])
                self._taboo = globe_taboo_list(s1, s2)

        return self._taboo

    @property
    def allowed_moves(self):
        return self._allowed_moves

    @property
    def allowed_move_ids(self):
        return list(self._allowed_moves.keys())

    @property
    def current_allowed_move_ids(self):
        if len(self) == 0:
            return self.allowed_move_ids

        forbidden_moves = self.taboo_list[self[-1]]
        if len(self) > 1:
            forbidden_moves += self.taboo_list.get((self[-2], self[-1]), [])

        return [
            move_id
            for move_id in self.allowed_move_ids
            if move_id not in forbidden_moves
        ]

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
        # assert self._current == self._initial
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
    def levenshtein_distance(self):
        return Levenshtein.distance("".join(self._current), "".join(self._solution))

    @property
    def levenshtein_ratio(self):
        return Levenshtein.ratio("".join(self._current), "".join(self._solution))

    # "quickmedian",
    # "median",
    # "median_improve",
    # "setmedian",
    # "setratio",
    # "seqratio",
    # "distance",
    # "ratio",
    # "hamming",
    # "jaro",
    # "jaro_winkler",
    # "editops",
    # "opcodes",
    # "matching_blocks",
    # "apply_edit",
    # "subtract_edit",
    # "inverse",

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

    @property
    def size(self):
        return len(self._current)

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


class WreathPuzzle(Puzzle):
    def __init__(self, id, puzzle_type, solution, initial, num_wildcards):
        super().__init__(id, puzzle_type, solution, initial, num_wildcards)
        self._size = int(self.type.split("_")[1].split("/")[0])
        self._allowed_moves = {
            "r": [i for i in range(int(self.type.split("_")[1].split("/")[0]))],
            "-r": [i for i in range(int(self.type.split("_")[1].split("/")[0]))],
            "l": [i for i in range(int(self.type.split("_")[1].split("/")[0]))],
            "-l": [i for i in range(int(self.type.split("_")[1].split("/")[0]))],
        }

    def random_solution(self, size):
        return list(choices(self.allowed_move_ids, k=size))

    def permutate(self, move_id):
        self._permutations.append(move_id)
        if move_id == "r":
            self._current = [self._current[-1]] + self._current[:-1]
        elif move_id == "-r":
            self._current = self._current[1:] + [self._current[0]]
        elif move_id == "l":
            self._current = self._current[1:] + [self._current[0]]
        elif move_id == "-l":
            self._current = [self._current[-1]] + self._current[:-1]
        return self

    @property
    def current_allowed_move_ids(self):
        if len(self) == 0:
            return self.allowed_move_ids

        forbidden_moves = self.taboo_list[self[-1]]
        if len(self) > 1:
            forbidden_moves += self.taboo_list.get((self[-2], self[-1]), [])

        return [
            move_id
            for move_id in self.allowed_move_ids
            if move_id not in forbidden_moves
        ]

    def __str__(self):
        return (
            f"{'.'.join(self._solution)}\n"
            f"{'.'.join(self._current)}\n"
            # f"{'.'.join(self._current[:self._size])}\n"
            # f"{'  ' * (self._size)}{self._current[self._size]}\n"
            # f"{'  ' * (self._size+1) + '.'.join(self._current[self._size+1:])}\n"
        )
