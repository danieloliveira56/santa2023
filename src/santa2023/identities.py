import math
import itertools
import numpy as np
import scipy
from typing import Iterable

from santa2023.puzzle import read_puzzle_info, read_puzzles
from santa2023.utils import get_inverse, read_solution, export_solution


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
    print(permutation_map)
    return permutation_map


def replace_moves(permutation, moves1, moves2):
    if len(moves1) == 0:
        return
    for i in range(len(permutation)-len(moves1)+1):
        if tuple(permutation[i:i+len(moves1)]) == tuple(moves1):
            new_permutation = permutation[:i] + list(moves2)
            if i+len(moves1) < len(permutation):
                new_permutation += permutation[i+len(moves1):]
            return new_permutation


def identities(args):
    solution = read_solution(filename=args.initial_solution_file)
    puzzles = read_puzzles("puzzles.csv")
    puzzle_info = read_puzzle_info("puzzle_info.csv")
    for p in puzzles:
        p.initialize_move_list(puzzle_info[p.type])

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


    new_solution = []
    try:
        for i, (permutation, puzzle) in enumerate(zip(solution, puzzles)):
            if puzzle.type != args.puzzle_type:
                new_solution.append(permutation)
                continue
            new_permutation = permutation.copy()
            has_repl = True
            while has_repl:
                has_repl = False
                for p1, p2 in replacements:
                    replaced_permutation = replace_moves(new_permutation, p1, p2)
                    if replaced_permutation:
                        new_permutation = replaced_permutation
                        has_repl = True
                        break
            new_solution.append(new_permutation)
            print(f"{puzzle._id}: {len(permutation)}->{len(new_permutation)}")

    except KeyboardInterrupt:
        new_solution = new_solution + solution[len(new_solution):]
        pass
    except Exception as e:
        raise e
    export_solution(puzzles, new_solution)


def argmax(x: Iterable, key):
    max_value = -math.inf
    max_idx = None
    max_element = None
    for i, element in enumerate(x):
        value = key(element)
        if value > max_value:
            max_idx = i
            max_element = element
            max_value = value

    return max_idx, max_element


def simple_wildcards(args):
    solution = read_solution(filename=args.initial_solution_file)
    puzzles = read_puzzles("puzzles.csv")
    puzzle_info = read_puzzle_info("puzzle_info.csv")
    for p in puzzles:
        p.initialize_move_list(puzzle_info[p.type])
    new_solution = []

    for permutation, puzzle in zip(solution, puzzles):
        if puzzle._num_wildcards == 0:
            new_solution.append(permutation)
            continue
        p = puzzle.clone()
        print(puzzle._id, puzzle._num_wildcards, len(permutation))
        for move_idx, move_id in enumerate(permutation):
            p.permutate(move_id)
            if move_idx < len(permutation)-1 and p.is_solved:
                print(f"Found wildcard {len(permutation)}->{move_idx+1}")
                break
        new_solution.append(p.permutations)
    export_solution(puzzles, new_solution)


def shortcut(args):
    solution = read_solution(filename=args.initial_solution_file)
    puzzles = read_puzzles("puzzles.csv")
    puzzle_info = read_puzzle_info("puzzle_info.csv")
    for p in puzzles:
        p.initialize_move_list(puzzle_info[p.type])
    new_solution = []
    for permutation, puzzle in zip(solution, puzzles):
        print(f"Searching shortcuts for {puzzle._id}", end="\r")
        new_permutation = permutation.copy()
        has_shortcut = True
        while has_shortcut:
            has_shortcut = False
            p = puzzle.clone()
            pattern_map = {p.current_pattern_hash: [0]}
            for move_idx, move_id in enumerate(new_permutation):
                p.permutate(move_id)
                id = p.current_pattern_hash
                pattern_map[id] = pattern_map.get(id, []) + [move_idx]
            candidate_shortcuts = [
                positions for positions in pattern_map.values() if len(positions) > 1
            ]
            if len(candidate_shortcuts) == 0:
                continue
            # candidate_shortcuts = sorted(candidate_shortcuts, key=lambda x: min(x)-max(x))
            _, longest_shortcut = argmax(
                candidate_shortcuts, key=lambda x: max(x) - min(x)
            )
            idx1 = min(longest_shortcut)
            idx2 = max(longest_shortcut)
            has_shortcut = True
            new_permutation = new_permutation[: idx1 + 1] + new_permutation[idx2 + 1 :]
            print(f"Searching shortcuts for {puzzle._id}: {len(permutation)}->{len(new_permutation)}")

        new_solution.append(new_permutation)

    export_solution(puzzles, new_solution)
