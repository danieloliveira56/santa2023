import itertools
import numpy as np
import scipy

from puzzle import read_puzzle_info, read_puzzles
from utils import get_inverse, read_solution

def get_identities(puzzle_info, depth):
    move_ids = list(puzzle_info.keys())
    for id in move_ids:
        puzzle_info[f"-{id}"] = get_inverse(puzzle_info[id])

    move_ids = list(puzzle_info.keys())
    puzzle_size = len(list(puzzle_info.values())[0])
    permutation_map = {tuple(range(puzzle_size)): [()]}
    for n in range(1, depth + 1):
        print(f"Mapping depth={n}/{depth}")
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
            result = np.arange(puzzle_size)
            for move_id in permutation:
                # result *= permutation_matrices[move_id]
                result = [result[i] for i in puzzle_info[move_id]]
            id = tuple(result)
            if id in permutation_map:
                permutation_map[id].append(permutation)
            else:
                permutation_map[id] = [permutation]
    return permutation_map


def identities(args):
    initial_solution = read_solution(filename=args.initial_solution_file)
    puzzles = read_puzzles("puzzles.csv")
    puzzle_info = read_puzzle_info("puzzle_info.csv")
    for p in puzzles:
        p.set_allowed_moves(puzzle_info[p.type])

    permutation_map = get_identities(puzzle_info[args.puzzle_type], args.depth)
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
    replacements = [(".".join(p1), ".".join(p2) if len(p2) else "") for p1, p2 in replacements]
    print(replacements)

    for puzzle_idx, p in enumerate(puzzles):
        if p.type != args.puzzle_type:
            continue
        cur_sol = p.clone().full_permutation(initial_solution[puzzle_idx])
        p_str = cur_sol.submission
        has_repl = True
        while has_repl:
            has_repl = False
            for p1, p2 in replacements:
                if f".{p1}" in p_str or f",{p1}" in p_str:
                    has_repl = True
                    p_str = p_str.replace(f".{p1}", f".{p2}")
                    p_str = p_str.replace(f",{p1}", f",{p2}")
                    p_str = p_str.replace("..", ".")
                    break

        new_sol = p.clone().full_permutation(p_str.split(",")[1].split("."))
        cur_score = cur_sol.score
        new_score = new_sol.score
        print(f"***({cur_score}->{new_score})***{new_sol.submission}")
