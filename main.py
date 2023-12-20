import itertools
from random import choice, sample

from genetic import crossover, mutate
from utils import (get_inverse, read_puzzle_info, read_puzzles, read_solution,
                   remove_identity)


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
