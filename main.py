import itertools
from random import choice, choices, randrange, sample

from genetic import crossover, mutate
from utils import (
    get_inverse,
    read_puzzle_info,
    read_puzzles,
    read_solution,
    remove_identity,
)


class Puzzle:
    def __init__(self, id, puzzle_type, solution, initial, num_wildcards):
        self._id = id
        self._type = puzzle_type
        self._initial = initial.split(";")
        self._current = initial.split(";")
        self._solution = solution.split(";")
        self._num_wildcards = int(num_wildcards)
        self._permutations = []

    def set_allowed_moves(self, allowed_moves):
        self._allowed_moves = allowed_moves

    @property
    def allowed_move_ids(self):
        return list(self._allowed_moves.keys())

    def random_solution(self, size):
        permutations = choices(self.allowed_move_ids, k=size)
        return ["-" * randrange(2) + p for p in permutations]

    def permutate(self, move_id, inverse=False):
        permutation = self._allowed_moves[move_id]
        if inverse:
            permutation = get_inverse(permutation)
            self._permutations.append(f"-{move_id}")
        else:
            self._permutations.append(move_id)

        self._current = [self._current[i] for i in permutation]
        return self

    def full_permutation(self, permutation_list):
        for move_id in permutation_list:
            self.permutate(move_id.strip("-"), move_id.startswith("-"))
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
            2
            * len(self._current)
            * sum([c1 != c2 for c1, c2 in zip(self._current, self._solution)])
            + len(self)
            + (0 if self.is_solved else 2)
        )

    @property
    def permutations(self):
        return self._permutations.copy()

    @property
    def type(self):
        return self._type

    @property
    def is_solved(self):
        return self._current == self._solution

    @property
    def submission(self):
        return f"{self._id},{'.'.join(self._permutations)}"

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


def get_identities(puzzle_info, depth):
    move_ids = list(puzzle_info.keys())
    for id in move_ids:
        puzzle_info[f"-{id}"] = get_inverse(puzzle_info[id])
    move_ids = list(puzzle_info.keys())
    puzzle_size = len(list(puzzle_info.values())[0])
    permutation_map = {tuple(range(puzzle_size)): [()]}
    for n in range(1, depth + 1):
        if n == 3:
            identities = [
                ".".join(p)
                for p in permutation_map[tuple(range(puzzle_size))]
                if len(p)
            ]
        for permutation in itertools.permutations(move_ids, r=n):
            has_identity = False
            if n > 2:
                permutation_str = ".".join(permutation)
                for p in identities:
                    if p in permutation_str:
                        has_identity = True
                        break
            if has_identity:
                continue
            result = puzzle_info[permutation[0]]
            for move_id in permutation[1:]:
                result = [result[i] for i in puzzle_info[move_id]]
            id = tuple(result)
            if id in permutation_map:
                permutation_map[id].append(permutation)
            else:
                permutation_map[id] = [permutation]
    return permutation_map


if __name__ == "__main__":
    num_iterations = 1000
    size_population = 100
    lucky_survivors = 10
    num_crossovers = 20
    num_mutations = 200

    puzzle_info = read_puzzle_info("puzzle_info.csv")
    puzzles = read_puzzles("puzzles.csv")
    sample_submission = read_solution("sample_submission.csv")

    for p in puzzles:
        p.set_allowed_moves(puzzle_info[p.type])

    permutation_map = get_identities(puzzle_info["cube_2/2/2"], 5)
    replacements = []
    for k, v in permutation_map.items():
        lengths = [len(p) for p in v]
        min_p = min(lengths)
        max_p = max(lengths)
        if min_p < max_p:
            replacement = [p for p in v if len(p) == min_p][0]
            replacements += [(p, replacement) for p in v if len(p) > min_p]
    # Sort replacements by larger sequences
    replacements = sorted(replacements, key=lambda x: -len(x[0]))

    for puzzle_idx, p in enumerate(puzzles):
        if p.type != "cube_2/2/2":
            continue
        cur_sol = p.clone().full_permutation(sample_submission[puzzle_idx])
        p_str = cur_sol.submission
        has_repl = True
        while has_repl:
            has_repl = False
            cur_len = len(p_str)
            for p1, p2 in replacements:
                find_str = ".".join(p1)
                replace_str = ".".join(p2)
                p_str = p_str.replace(f".{find_str}", f".{replace_str}")
                p_str = p_str.replace("..", ".")
                p_str = p_str.replace(f",{find_str}", f",{replace_str}")
            if len(p_str) < cur_len:
                has_repl = True
        new_sol = p.clone().full_permutation(p_str.split(",")[1].split("."))
        cur_score = cur_sol.score
        new_score = new_sol.score
        print(f"***{new_sol.submission}")

    solution_score = {
        "original": 0,
        "crossover": 0,
        "replace-mutation": 0,
        "delete-mutation": 0,
        "insert-mutation": 0,
    }

    for puzzle_idx, p in enumerate(puzzles):
        initial_solution = sample_submission[puzzle_idx]
        current_score = len(initial_solution)
        initial_permutations = [
            p.random_solution(len(initial_solution)) for _ in range(size_population)
        ]
        initial_permutations.append(initial_solution)

        pool = [
            (p.clone().full_permutation(permutation), "original")
            for permutation in initial_permutations
        ]

        for i in range(num_iterations):
            for j in range(num_crossovers):
                new_p = crossover(*sample(pool, 2))
                pool.append(
                    (p.clone().full_permutation(remove_identity(new_p)), "crossover")
                )
            for j in range(num_mutations):
                new_p, mutation_type = mutate(
                    choice(pool)[0].permutations, p.allowed_move_ids
                )
                pool.append(
                    (p.clone().full_permutation(remove_identity(new_p)), mutation_type)
                )

            pool = sorted(pool, key=lambda x: x[0].score)
            pool = pool[: (size_population - lucky_survivors)] + sample(
                pool[(size_population - lucky_survivors) :], k=lucky_survivors
            )
            new_score = pool[0][0].score
            if new_score < current_score:
                solution_score[pool[0][1]] += current_score - new_score
                current_score = new_score

            print(
                f"Searching {puzzle_idx}/{len(puzzles)}, "
                f"End of iteration {i+1}/{num_iterations}, "
                f"Pool size: {len(pool)} "
                f"Score: {len(initial_solution)}->{new_score}"
            )
        if pool[0][0].is_solved:
            print(f"***{pool[0][0].submission}")
        else:
            print("No solution found")
        print(solution_score)
