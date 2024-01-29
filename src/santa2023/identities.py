import math
import os
import time
from pathlib import Path
from typing import Iterable
import networkx as nx
import sympy.combinatorics

from santa2023.data import PUZZLE_INFO, PUZZLES
from santa2023.puzzle import Permutation, WreathPuzzle
from santa2023.utils import (PUZZLE_TYPES, calculate_score, export_solution,
                             get_inverse, print_globe, read_solution, clean_solution, debug_list)


def get_identities(puzzle_info, depth):
    func_start = time.time()
    move_ids = list(puzzle_info.keys())
    for id in move_ids:
        puzzle_info[f"-{id}"] = get_inverse(puzzle_info[id])

    puzzle_size = len(list(puzzle_info.values())[0])
    identity = tuple(range(puzzle_size))
    permutation_map = {identity: [()]}
    levels = {
        1: {
            tuple([move_id]): tuple(move_mapping)
            for move_id, move_mapping in puzzle_info.items()
        }
    }
    levels[2] = {
        tuple(list(permutation) + [move_id]): tuple(
            [permutation_mapping[i] for i in move_mapping]
        )
        for move_id, move_mapping in puzzle_info.items()
        for permutation, permutation_mapping in levels[1].items()
    }
    for n in range(3, depth + 1):
        start = time.time()
        print(f"Mapping depth={n}/{depth}", end="\r")
        levels[n] = {
            tuple(list(permutation) + [move_id]): tuple(
                [permutation_mapping[i] for i in move_mapping]
            )
            for move_id, move_mapping in puzzle_info.items()
            for permutation, permutation_mapping in levels[n - 1].items()
            if (move_id != f"-{permutation[-1]}" and permutation[-1] != f"-{move_id}")
            and permutation_mapping != identity
        }
        print(f"Mapped depth={n}/{depth} in {time.time()-start:.2f}s")

    for level in levels.values():
        for permutation, permutation_mapping in level.items():
            if permutation_mapping in permutation_map:
                permutation_map[permutation_mapping].append(permutation)
            else:
                permutation_map[permutation_mapping] = [permutation]

    print(permutation)
    print(f"\nPermutation Map Size: {len(permutation_map):0,}")
    print(f"Total time: {time.time()-func_start:.2f}s")
    exit()
    return permutation_map


def get_fast_identities(puzzle_info, depth, max_time=None):
    func_start = time.time()
    move_ids = list(puzzle_info.keys())
    for id in move_ids:
        puzzle_info[f"-{id}"] = get_inverse(puzzle_info[id])

    puzzle_size = len(list(puzzle_info.values())[0])
    identity = tuple(range(puzzle_size))

    base_permutations = sorted(
        [Permutation(mapping, move_id) for move_id, mapping in puzzle_info.items()]
    )
    permutation_map = {identity: Permutation(identity)}
    added_permutations = [permutation_map[identity]]
    for level in range(1, depth + 1):
        if max_time and time.time() - func_start > int(max_time):
            print(f"Aborting at level {level}, exceeded {max_time}s")
            break
        start = time.time()
        print(f"Mapping depth={level}/{depth}", end="\r")
        print(f"len(added_permutations): {len(added_permutations)}")
        print(f"len(base_permutations): {len(base_permutations)}")
        new_permutations = [
            p1 * p2 for p1 in added_permutations for p2 in base_permutations
        ] + [
            p1 * p2 * ~p1 * ~p2 for p1 in added_permutations for p2 in base_permutations
        ]
        print(
            f"Mapping depth={level}/{depth}: {len(new_permutations):0,} new permutations computed in {time.time() - start:.2f}s"
        )
        start = time.time()
        added_permutations = []
        for i, p1 in enumerate(new_permutations):
            print(f"Mapping permutation {i+1:0,}/{len(new_permutations):0,}", end="\r")
            id = p1.mapping
            if id not in permutation_map:
                permutation_map[id] = p1
                added_permutations.append(p1)

        print(
            f"                   {len(added_permutations):0,} mapped in {time.time() - start:.2f}s"
        )

    print(f"\nPermutation Map Size: {len(permutation_map):0,}")
    print(f"Total time: {time.time()-func_start:.2f}s")

    return permutation_map


def replace_moves(permutation, moves1, moves2):
    if len(moves1) == 0:
        return
    for i in range(len(permutation) - len(moves1) + 1):
        if tuple(permutation[i : i + len(moves1)]) == tuple(moves1):
            new_permutation = permutation[:i] + list(moves2)
            if i + len(moves1) < len(permutation):
                new_permutation += permutation[i + len(moves1) :]
            return new_permutation


def identities(args):
    if args.fast:
        print("Running Fast Identities")
        fast_identities(args)
    else:
        print("Running Slow Identities")
        slow_identities(args)


def slow_identities(args):
    solution = read_solution(filename=args.initial_solution_file)

    permutation_map = get_identities(PUZZLE_INFO[args.puzzle_type], args.depth)
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
        for i, (permutation, puzzle) in enumerate(zip(solution, PUZZLES)):
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
                        print(p1, p2)
                        print(new_permutation)
                        print(replaced_permutation)
                        new_permutation = replaced_permutation.copy()
                        assert (
                            puzzle.clone().full_permutation(new_permutation).is_solved
                        )
                        has_repl = True
                        break
            new_solution.append(new_permutation)
            print(f"{puzzle._id}: {len(permutation)}->{len(new_permutation)}")

    except KeyboardInterrupt:
        new_solution = new_solution + solution[len(new_solution) :]
        pass
    except Exception as e:
        raise e
    export_solution(PUZZLES, new_solution)


def wreath(args):
    puzzle = PUZZLES[334]
    wreath = WreathPuzzle(
        puzzle._id,
        puzzle.type,
        ";".join(puzzle._initial),
        ";".join(puzzle._solution),
        puzzle._num_wildcards,
    )
    print(wreath)

    sol = "l.-r.-l.l.l.l.l.l.l.-r.-l.l.l.l.l.l.l.l.l.l.l.-r.-r.-r.-r.-l.l.l.l.l.l.-r.-l.-r.-l.-l.l.l.l.l.l.l.l.l.-r.r.-l.l.l.l.l.l.l.-r.-r.-r.r.-r.r.-r.r.-r.r.-r.r.-r.r.-r.r.-r.r.-r.r.-r.r.-r.r.-r.r.-r.r.-r.r.-r.r.-r.-r.r.-r.r.r.-r.r.-r.r.-r.-r.r.-r.r.-r.r.r.-r.-r.r.r.-r.r.-l.l.-r.r.-l.l.-r.-r.r.-r.-l.r.r.-r.-r.-l.l.r.l.-l.-r.r.-r.r.l.-r.-l.l.-r.r.-r.-r.-r.-r.-l.l.-l.-r.r.-l.l.l.r.-l.-l.-l.-l.-l.-r.r.l.-r.-r.-r.-r.r.-l.-r.-r.-r.-r.-r.-r.-r.-r.-l.-l.-r.r.-l.-l.-l.l.l.l.-r.r.-l.-l.-l.l.l.l.l.-l.-l.-l.-l.l.l.l.l.-l.-l.-l.-l.l.l.l.l.-l.-l.-l.-l.l.l.l.r.-r.-l.-l.-l.l.l.l.r.-r.-l.-l.-l.l.l.l.l.-l.-l.-l.-l.l.l.l.r.-r.-l.-l.-l.l.l.l.l.-l.-l.-l.-l.l.l.l.-r.r.-l.-l.-l.l.l.l.l.-l.-l.-l.-l.l.l.l.l.-l.-l.-l.-l.l.l.l.-r.r.-l.-l.-l.l.l.l.-r.r.-l.-l.-l.l.l.l.-r.r.-l.-l.-l.l.l.l.r.-r.-l.-l.-l.l.l.l.-r.r.-l.-l.-l.l.l.l.r.-r.-l.-l.-l.l.l.l.r.-r.-l.-l.-l.l.l.l.-r.r.-l.-l.-l.l.l.l.l.-l.-l.-l.-l.l.l.l.-r.r.-l.-l.-l.l.l.l.-r.r.-l.-l.-l.l.l.l.-r.r.-l.-l.-l.l.l.l.-r.r.-l.-l.-l.l.l.l.l.-l.-l.-l.-l.l.l.l.r.-r.-l.-l.-l.l.l.l.r.-r.-l.-l.-l.l.l.l.-r.r.-l.-l.-l.-l.l.l.l.l.r.-r.-l.-l.-l.l.l.l.-r.r.-l.-l.-l.-l.l.l.l.l.r.-r.l.r.-r.-l.r.-r.-l.l.-l.-l.-l.l.l.l.-r.r.-l.-l.-l.-l.l.l.l.l.l.-l.-r.r.-l.-l.-l.l.l.l.-r.r.-l.-r.l.r.-l.-r.-r.-l.-l.r.l.-l.-r.l.-l.l.r.-r.-r.l.r.-l.r.-r.l.r.-l.l.-r.r.-r.-r.r.-l.-l.l.l.-l.r.-r.l.r.l.-l.-l.-l.-l.-l.-l.-l.-l.-l.-l.r.r.r.r.r.r.r.r.r.-l.l.l.l.l.l.-r.-r.-r.r.-r.r.-r.r.-r.r.-r.r.-r.r.-r.r.-r.r.-r.r.-r.r.-r.r.-r.r.-r.r.-r.r.-r.r.-r.r.-r.r.-r.r.-r.r.-r.r.-r.l.r.-r.r.-r.r.-r.r.-r.r.r.r.-r.-r.-r.r.-r.-l.r.-r.r.-r.r.-r.r.-r.r.l.-r.-r.r.-l.r.-r.l.-l.r.-r.l.-l.r.r.r.l.-r.-r.-r.r.r.-r.r.-r.-l.-r.r.l.-r.r.r.-r.r.r.-l.-l.-l.-l.-l.l.-r.l.r.-l.-r.-r.r.-l.-r.l.l.-r.-l.l.l.l.r.l.r.l.l.l.l.l.l.-r.-l.-l.-l.-l.-l.-l.-l.l.l.l.-r.-l.-l.-l.l.l.l.r.-l.-l.-l.l.l.l.-r.-l.-l.-l.l.l.l.r.-l.-l.-l.l.l.l.-r.-l.-l.-l.l.l.l.r.-l.-l.-l.l.l.l.-r.-l.-l.-l.l.l.l.r.-l.-l.-l.l.l.l.-r.-l.-l.-l.l.l.l.r.-l.-l.-l.l.l.l.-r.-l.-l.-l.l.l.l.r.-l.-l.-l.l.l.l.-r.-l.-l.-l.l.l.l.r.-l.-l.-l.l.l.l.-r.-l.-l.-l.l.l.l.r.-l.-l.-l.l.l.l.-r.-l.-l.-l.l.l.l.r.-l.-l.-l.l.l.l.-r.-l.-l.-l.l.l.l.r.-l.-l.-l.l.l.l.-r.-l.-l.-l.l.l.l.r.-l.-l.-l.l.l.l.-r.-l.-l.-l.l.l.l.r.-l.-l.-l.l.l.l.-r.-l.-l.-l.l.l.l.r.-l.-l.-l.l.l.l.-r.-l.-l.-l.l.l.l.r.-l.-l.-l.l.l.l.-r.-l.-l.-l.l.l.l.r.-l.-l.-l.l.l.l.-r.-l.-l.-l.l.l.l.r.-l.-l.-l.l.l.l.-r.-l.-l.-l.l.l.l.r.-l.-l.-l.l.l.l.-r.-l.-l.-l.l.l.l.r.-l.-l.-l.l.l.l.-r.-l.-l.-l.l.l.l.r.-l.-l.-l.l.l.l.-r.-l.-l.-l.l.l.l.r.-l.-l.-l.l.l.l.-r.-l.-l.-l.l.l.l.r.-l.-l.-l.l.l.l.-r.-l.-l.-l.l.l.l.r.-l.-l.-r.l.l.r.-r.l.r.l.-r.-l.r.-l.-r.l.r.l.-l.-l.l.-l.l.-r.-l.r.r.-r.l.-r.-l.l.r.-l.-r.-l.r.l.l.-r.-l.r.l.-l.-l.-r.r.-l.-l.l.-r.l.-l.-r.l.r.r.-l.-r.-r.r.l.l.-r.-l.-l.-r.l.r.r.-l.-r.-l.r.-l.-r.-l.l.l.l.r.-l.-r.-l.-l.-l.-l.-l.-l.-l.-l.-l.-r.r.-r.r.-r.r.-r.r.-r.r.-r.r.-r.r.-r.r.-r.r.-r.r.-r.r.-r.r.-r.r.-r.r.-r.r.-r.r.-r.r.-r.r.-r.r.-r.r.-r.r.-r.r.-r.r.-l.l.-r.r.-r.r.-r.r.-r.r.-r.r.-r.r.-r.r.-l.l.-r.r.-r.-r.r.r.-r.r.-r.r.r.-r.-r.r.-r.l.l.r.-l.l.-r.l.-l.r.-r.-r.r.l.-l.r.-r.-l.-l.r.l.-l.l.r.-l.-r.l.r.-l.l.-l.-r.l.r.-l.-r.-r.-r.r.l.l.r.-r.l.-l.-l.-l.-r.r.r.-l.-r.l.r.-l.r.-r.-r.r.l.l.-r.r.l.-r.-r.r.r.-r.l.-r.r.r.-l.-r.l.r.-r.l.l.l.l.l.l.l.l.l.r.-l.l.l.l.l.l.l.l.l.l.l.r.-r.-l.l.l.l.l.-l.-l.-l.-l.l.l.l.l.-l.-l.-l.-l.l.l.l.l.-l.-l.-l.-l.l.l.l.r.-r.-l.-l.-l.l.l.l.r.-r.-l.-l.-l.l.l.l.r.-r.-l.-l.-l.l.l.l.r.-r.-l.-l.-l.l.l.l.r.-r.-l.-l.-l.l.l.l.r.-r.-l.-l.-l.l.l.l.-r.r.-l.-l.-l.l.l.l.-r.r.-l.-l.-l.l.l.l.l.-l.-l.-l.-l.l.l.l.-r.r.-l.-l.-l.l.l.l.r.-r.-l.-l.-l.l.l.l.-r.r.-l.-l.-l.l.l.l.r.-r.-l.-l.-l.l.l.l.-r.r.-l.-l.-l.l.l.l.r.-r.-l.-l.-l.l.l.l.r.-r.-l.-l.-l.l.l.l.r.-r.-l.-l.-l.l.l.l.r.-r.-l.-l.-l.l.l.l.-r.r.-l.-l.-l.l.l.l.r.-r.-l.-l.-l.l.l.l.r.-r.-l.-l.-l.l.l.l.l.-l.-l.-l.-l.l.l.l.-r.r.-l.-l.-l.l.l.l.l.-l.-l.-l.-l.l.l.l.-r.r.-l.-l.-l.l.l.l.l.-l.-l.-l.-l.l.l.l.-r.r.-l.-l.-l.l.l.l.r.-r.-l.-l.-l.l.l.l.r.-r.-l.-l.-l.l.l.l.r.-r.-l.-l.-l.l.l.l.l.-l.-l.-l.-l.l.l.l.r.-r.-l.-l.-l.-l.-l.-r.r.-l.-l.-l.-l.-l.-l.-l.l.l.l.l.l.l.l.l.l.l.r.-r.l.r.-r.-r.-l.r.l.-r.l.-r.-r.-l.l.-r.-r.-r.-r.-l.l.-l.-l.-l.l.l.l.r.r.-l.-r.-r.l.r.r.-r.r.-l.-r.-l.-r.r.-r.l.r.-l.r.-l.-l.-l.-r.r.-l.-l.-r.r.-l.-l.-l.-l.-r.-l.l.l.l.-r.r.-l.-l.-l.l.l.l.-r.r.-l.-l.-l.l.l.l.r.-r.-l.-l.-l.l.l.l.r.-r.-l.-l.-l.l.l.l.l.-l.-l.-l.-l.l.l.l.r.-r.-l.-l.-l.l.l.l.-r.r.-l.-l.-l.l.l.l.r.-r.-l.-l.-l.l.l.l.r.-r.-l.-l.-l.l.l.l.-r.r.-l.-l.-l.l.l.l.r.-r.-l.-l.-l.l.l.l.r.-r.-l.-l.-l.l.l.l.r.-r.-l.-l.-l.l.l.l.l.-l.-l.-l.-l.l.l.l.-r.r.-l.-l.-l.l.l.l.-r.r.-l.-l.-l.l.l.l.-r.r.-l.-l.-l.l.l.l.r.-r.-l.-l.-l.l.l.l.r.-r.-l.-l.-l.l.l.l.l.-l.-l.-l.-l.l.l.l.l.-l.-l.-l.-l.l.l.l.r.-r.-l.-l.-l.l.l.l.r.-r.-l.-l.-l.l.l.l.r.-r.-l.-l.-l.l.l.l.-r.r.-l.-l.-l.l.l.l.-r.r.-l.-l.-l.l.l.l.r.-r.-l.-l.-l.l.l.l.-r.r.-l.-l.-l.l.l.l.r.-r.-l.-l.-l.l.l.l.-r.r.-l.-l.-l.l.l.l.-r.r.-l.-l.-l.l.l.l.r.-r.-l.-l.-l.l.l.l.l.-l.-l.-l.-l.l.l.l.-r.r.-l.-l.-l.l.l.l.r.-r.-l.-l.-l.l.l.l.l.l.-l.-l.-l.-l.-r.-r.-r.-r.-r.l.l.r.-r.l.l.-r.-r.-r.-r.-r.-l.-l.-l.l.l.l.-r.r.-l.-l.-l.l.l.l.r.-r.-l.-l.-l.l.l.l.r.-r.-l.-l.-l.l.l.l.r.-r.-l.-l.-l.l.l.l.l.-l.-l.-l.-l.l.l.l.-r.r.-l.-l.-l.l.l.l.l.-l.-l.-l.-l.l.l.l.l.-l.-l.-l.-l.l.l.l.-r.r.-l.-l.-l.l.l.l.r.-r.-l.-l.-l.l.l.l.r.-r.-l.-l.-l.l.l.l.-r.r.-l.-l.-l.l.l.l.r.-r.-l.-l.-l.l.l.l.-r.r.-l.-l.-l.l.l.l.l.-l.-l.-l.-l.l.l.l.r.-r.-l.-l.-l.l.l.l.r.-r.-l.-l.-l.l.l.l.r.-r.-l.-l.-l.l.l.l.l.-l.-l.-l.-l.l.l.l.l.-l.-l.-l.-l.l.l.l.-r.r.-l.-l.-l.l.l.l.-r.r.-l.-l.-l.l.l.l.l.-l.-l.-l.-l.l.l.l.r.-r.-l.-l.-l.l.l.l.l.-l.-l.-l.-l.l.l.l.r.-r.-l.-l.-l.l.l.l.r.-r.-l.-l.-l.l.l.l.-r.r.-l.-l.-l.l.l.l.l.-l.-l.-l.-l.l.l.l.r.-r.-l.-l.-l.l.l.l.-r.r.-l.-l.-l.l.l.l.r.-r.-l.-l.-l.l.l.l.r.-r.-r.r.-l.-l.-l.l.l.l.r.-r.-l.l.l.l.l.l.l.l.l.r.-r.l.-l.-l.-l.l.l.-r.r.-l.l.l.l.l.l.l.l.l.l.l.-r.r.-l.l.l.l.l.l.l.l.l.l.r.-r.-l.l.l.l.l.l.l.l.l.r.-r.l.l.r.-r.-l.-r.-l.l.r.-l.l.-r.r.l.l.-l.l.-r.r.-r.-l.r.l.-r.-l.-l.-l.-l.-l.-l.-l.-l.-l.-l.l.l.l.l.l.l.l.r.-r.-l.l.l.l.l.-r.r.-l.-l.-l.-l.l.l.l.l.r.-r.-l.-l.-l.-l.l.l.l.l.-r.r.-l.-l.-l.-l.l.l.l.l.r.-r.-l.-l.-l.-l.l.l.l.l.l.-l.-l.-l.-l.-l.l.l.l.l.r.-r.-l.-l.-l.-l.l.l.l.l.r.-r.-l.-l.-l.-l.l.l.l.l.r.-r.-l.-l.-l.-l.l.l.l.l.r.-r.-l.-l.-l.-l.l.l.l.l.r.-r.-l.-l.-l.-l.l.l.l.l.r.-r.-l.-l.-l.-l.l.l.l.l.l.-l.-l.-l.-l.-l.l.l.l.l.r.-r.-l.-l.-l.-l.l.l.l.l.-r.r.-l.-l.-l.-l.l.l.l.l.r.-r.-l.-l.-l.-l.l.l.l.l.-r.r.-l.-l.-l.-l.l.l.l.l.-r.r.-l.-l.-l.-l.l.l.l.l.l.-l.-l.-l.-l.-l.l.l.l.l.r.-r.-l.-l.-l.-l.l.l.l.l.r.-r.-l.-l.-l.-l.l.l.l.l.-r.r.-l.-l.-l.-l.l.l.l.l.-r.r.-l.-l.-l.-l.l.l.l.l.-r.r.-l.-l.-l.-l.l.l.l.l.-r.r.-l.-l.-l.-l.l.l.l.l.-r.r.-l.-l.-l.-l.l.l.l.l.-r.r.-l.-l.-l.-l.l.l.l.l.r.-r.-l.-l.-l.-l.l.l.l.l.r.-r.-l.-l.-l.-l.l.l.l.l.l.-l.-l.-l.-l.-l.l.l.l.l.-r.r.-l.-l.-l.-l.l.l.l.l.r.-r.-l.-l.-l.-l.l.l.l.l.-r.r.-l.-l.-l.-l.l.l.l.l.r.-r.-l.l.l.l.l.l.l.l.r.-r.-l.-l.l.-l.-l.l.l.l.-r.r.-l.l.l.l.l.l.l.l.l.l.l.-r.r.-l.l.l.l.l.l.l.l.l.l.l.l.-l.r.r.l.-l.r.r.-r.-r.-r.-r.r.-l.r.l.-r.-l.-r.-r.-l.r.l.-r.r.-r.-l.-r.r.l.-r.-l.r.l.-r.l.-l.-r.-l.r.l.-r.-l.-r.r.-r.-l.-r.r.r.l.-r.-l.-l.l.l.l.l.l.l.l.l.l.l.l.-l.-l.l.l.l.l.l.l.l.l.l.l.r.-r.-l.-l.-l.l.l.l.r.-r.-l.-l.-l.l.l.l.r.-r.-l.-l.-l.l.l.l.l.-l.-l.-l.-l.l.l.l.r.-r.-l.-l.-l.l.l.l.-r.r.-l.-l.-l.l.l.l.-r.r.-l.-l.-l.l.l.l.-r.r.-l.-l.-l.l.l.l.r.-r.-l.-l.-l.l.l.l.r.-r.-l.-l.-l.l.l.l.-r.r.-l.-l.-l.l.l.l.-r.r.-l.-l.-l.l.l.l.r.-r.-l.-l.-l.l.l.l.r.-r.-l.-l.-l.l.l.l.r.-r.-l.-l.-l.l.l.l.r.-r.-l.-l.-l.l.l.l.-r.r.-l.-l.-l.l.l.l.r.-r.-l.-l.-l.l.l.l.r.-r.-l.-l.-l.l.l.l.r.-r.-l.-l.-l.l.l.l.l.-l.-l.-l.-l.l.l.l.r.-r.-l.-l.-l.l.l.l.-r.r.-l.-l.-l.l.l.l.-r.r.-l.-l.-l.l.l.l.l.-l.-l.-l.-l.l.l.l.-r.r.-l.-l.-l.l.l.l.r.-r.-l.-l.-l.l.l.l.l.-l.-l.-l.-l.l.l.l.r.-r.-l.-l.-l.l.l.l.r.-r.-l.-l.-l.l.l.l.r.-r.-l.-l.-l.l.l.l.r.-r.-l.-l.-l.l.l.l.-r.r.-l.r.-r.-l.-l.l.l.l.-r.r.-l.-l.-r.-r.-l.l.l.l.r.-r.-l.l.l.l.l.l.l.l.r.r.-l.l.l.l.l.l.l.l.l.l.l.r.-r.-l.l.l.l.l.l.l.l.l.l.l.r.-r.-r.l.-r.-l.r.l.r.-r.-l.r.l.-r.-l.-l.-r.l.-r.r.r.-l.-r.-r.l.-l.-l.r.l.-r.-l.l.r.-l.-r.l.r.r.-r.-l.-r.r.r.l.-r.-l.r.l.-l.r.-l.r.l.-r.r.-r.-l.-r.-r.-l.r.r.l.-r.-r.l.-l.-l.r.-l.l.r.l.r.-l.-r.l.r.-r.-r.r.l.l.l.l.l.l.l.l.l.-r"
    sol = sol.split(".")

    for p in sol:
        wreath.permutate(p)
    print(wreath)
    while True:
        wreath.permutate(input())
        print(wreath)


def test(args):
    solution = read_solution(filename=args.initial_solution_file)

    puzzle = PUZZLES[args.puzzle_id]
    puzzle_type = puzzle.type

    print(puzzle._id)
    print(puzzle_type)

    print(f"Puzzle type '{puzzle_type}' permutations:")

    permutations = {
        k: sympy.combinatorics.Permutation(p)
        for k, p in PUZZLE_INFO[puzzle_type].items()
    }

    for i in permutations.keys():
        for j in permutations.keys():
            if i == j:
                continue
            if permutations[i].commutes_with(permutations[j]):
                print(f"Permutation {i}, {j} are commutative")
            pij = permutations[i] * permutations[j]
            if pij == ~pij:
                print(f"Permutations {i}.{j}, -{i}.-{j} are inverses")
        print()

    for i in permutations.keys():
        if permutations[i] == ~permutations[i]:
            print(f"Permutation {i} is its own inverse")

    for i in permutations.keys():
        for j in permutations.keys():
            if i == j:
                continue
            if permutations[i] == ~permutations[j]:
                print(f"Permutation {i},{j} are inverses")

    print(f"Puzzle type '{puzzle_type}' permutations:")
    for i, p in permutations.items():
        print(f"Permutation {i}:")
        print(p)
        print(p.array_form)
        print(p.cyclic_form)
        print()

    if puzzle_type.startswith("globe"):
        print_globe(puzzle_type)

    key_values = list(permutations.items())
    for k, p in key_values:
        permutations[f"-{k}"] = ~p

    mismatches = [puzzle.count_mismatches]
    for move in solution[puzzle._id]:
        puzzle.permutate(move)
        mismatches.append(puzzle.count_mismatches)

    # for i in range(0, len(mismatches) - 1, 10):
    #     print(".".join(solution[puzzle._id][i:i+10]))
    #     print(" ".join([str(m) for m in mismatches[i:i+10]]))

    # print("Solution:")
    # curr_mismatches = puzzle.count_mismatches
    # print(f"Initial mismatches: {curr_mismatches}")
    # solution_blocks = []
    # curr_block = []
    # i = 0
    # while i < len(solution[puzzle._id]):
    #     while puzzle.count_mismatches >= curr_mismatches:
    #         move = solution[puzzle._id][i]
    #         curr_block.append(move)
    #         puzzle.permutate(move)
    #         i += 1
    #     print(puzzle.count_mismatches, curr_block)
    #     solution_blocks.append(curr_block)
    #     curr_block = []
    #     curr_mismatches = puzzle.count_mismatches

    p = permutations[solution[puzzle._id][0]]
    ct = sum([1 if i != j else 0 for i, j in enumerate(p.array_form)])
    ct_series = [ct]
    for i in range(1, len(solution[puzzle._id])):
        p *= permutations[solution[puzzle._id][i]]
        ct = sum([1 if i != j else 0 for i, j in enumerate(p.array_form)])
        ct_series.append(ct)
    print("ct_series:", ct_series)

    min_ct = 100
    n = len(solution[puzzle._id])
    for i in range(n):
        p = permutations[solution[puzzle._id][i]]
        for j in range(i + 1, n):
            p *= permutations[solution[puzzle._id][j]]
            ct = sum([1 if i != j else 0 for i, j in enumerate(p.array_form)])
            if ct <= 4:
                print(f"{i}:{j}")
                print(solution[puzzle._id][i : j + 1])
                print(p)
                print(p.array_form)
            min_ct = min(min_ct, ct)
        print(f"min_ct: {min_ct}", end="\r")
    print()

    # initial_state = puzzle._initial
    # goal = puzzle._solution
    # goal_map = {goal[i]: i for i in range(len(goal))}
    # solution_permutation_array = [
    #     goal_map[initial_state[i]] for i in range(len(initial_state))
    # ]
    # solution_permutation = sympy.combinatorics.Permutation(solution_permutation_array)
    #
    # print("Solution permutation:")
    # print(solution_permutation)

    for seq in [
        "f4",
        "r0",
        "-r0.f4",
        "-r0.f4.r0",
        "-r0.-r0.f4.r0.r0",
        "-r1.f4.r1",
        "-r1.-r1.f4.r1.r1",
        "-r2.f4.r2",
        "-r1.f4.r1.-r2.f4.r2",
        "-r1.-r1.f4.r1.r1.-r2.f4.r2",
    ]:
        print(seq)
        moves = seq.split(".")
        p = permutations[moves[0]]
        for move_id in moves[1:]:
            p *= permutations[move_id]
        print(p)

    exit()

    for seq in (
        [f"r0.f{i}.-r0.-f{i}" for i in range(0, 8)]
        + [f"f{i}.r0.-f{i}.-r0" for i in range(0, 8)]
        + [f"f0.f{i}" for i in range(0, 8)]
        + [
            "f0.f2",
            "f0.f2.r3",
            "f0.f2.r3.f2",
            "f0.f2.r3.f2.f0",
            "f0.f2.r3.f2.f0.f0.f2.r3.f2.f0",
            "f0.f2.r3.f2.f0.r0",
            "f0.f2.r3.f0.f2",
            "f0.f2.r3.f0.f2.-r3",
        ]
    ):
        print(seq)
        moves = seq.split(".")
        p = permutations[moves[0]]
        for move_id in moves[1:]:
            p *= permutations[move_id]
        print(p)
    exit()
    G = sympy.combinatorics.PermutationGroup(list(permutations.values()))
    G.schreier_sims()
    print(f"Group '{puzzle_type}' base (len(base)={len(G.base)}):")
    print(G.base)
    print(
        f"Group '{puzzle_type}' generators (len(G.strong_gens)={len(G.strong_gens)}):"
    )
    for i, p in enumerate(G.strong_gens):
        print(i, p)

    print(G.stabilizer(0))

    print()
    p1 = permutations[0]
    p2 = permutations[1]
    print(p1)
    print(p2)
    print(p1 * p2 * (~p1))

    # puzzle_type = "cube_5/5/5"
    #     print(f"Puzzle type '{puzzle_type}' permutations:")
    #     puzzle_size = [puzzle for puzzle in puzzles if puzzle.type == puzzle_type][0].size()
    #     # Create graph of size puzzle_size


def fast_identities(args):
    solution = read_solution(filename=args.initial_solution_file)

    puzzle_type = args.puzzle_type
    if puzzle_type == "all":
        puzzle_types = PUZZLE_TYPES
        puzzle_types.pop()
    else:
        puzzle_types = [puzzle_type]

    all_shortest_permutations = {
        puzzle_type: get_fast_identities(
            all_puzzle_info[puzzle_type], args.depth, args.max_time
        )
        for puzzle_type in puzzle_types
    }
    a = list(list(all_shortest_permutations.values())[0].keys())
    a = sorted(a, key=lambda x: sum([-1 if i != j else 0 for i, j in enumerate(x)]))
    # for m in a:
    #     print(m, sum([1 if i != j else 0 for i, j in enumerate(m)]))
    # exit()

    new_solution = {}
    try:
        for puzzle in PUZZLES:
            if puzzle.type not in puzzle_types:
                new_solution[puzzle._id] = solution[puzzle._id]
                continue

            permutations = [
                Permutation(PUZZLE_INFO[puzzle.type][move_id], move_id)
                for move_id in solution[puzzle._id]
            ]
            shortest_permutations = all_shortest_permutations[puzzle.type]
            initial_len = len(permutations)
            print(
                f"Searching puzzle {puzzle._id} ({puzzle.type}) Size {len(permutations)}                                        ",
                end="\r",
            )

            i = 0
            while i < len(permutations):
                p = permutations[i]
                j = i + 1
                while j < len(permutations):
                    print(
                        f"Searching puzzle {puzzle._id} position {i:5d}-{j:5d}/{len(permutations):6d}             ",
                        end="\r",
                    )
                    p *= permutations[j]
                    id = p.mapping
                    if id in shortest_permutations:
                        if shortest_permutations[id] < p:
                            permutations[i : j + 1] = shortest_permutations[id].split(
                                all_puzzle_info[puzzle.type]
                            )
                            print(
                                f"Searching puzzle {puzzle._id} ({puzzle.type}) [{i}:{j}](size={j-i+1}) ({initial_len})->({len(permutations)})"
                            )
                            print(f"Map size: {len(shortest_permutations):0,}")

                            j = i + len(shortest_permutations[id]) - 1

                        elif p < shortest_permutations[id]:
                            shortest_permutations[id] = p
                        else:
                            shortest_permutations[id] = p
                    else:
                        shortest_permutations[id] = p
                    j += 1
                i += 1

            permutations_list = []
            for p in permutations:
                permutations_list += p.move_ids
            solution[puzzle._id] = permutations_list
    except KeyboardInterrupt:
        pass
    except Exception as e:
        raise e
    export_solution(PUZZLES, solution)


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
    new_solution = []

    for id, sol in solution.items():
        puzzle = PUZZLES[id]
        if puzzle._num_wildcards == 0:
            new_solution.append(solution[puzzle._id])
            continue
        p = puzzle.clone()
        print(puzzle._id, puzzle._num_wildcards, len(solution[puzzle._id]))
        for move_idx, move_id in enumerate(solution[puzzle._id]):
            p.permutate(move_id)
            if move_idx < len(solution[puzzle._id]) - 1 and p.is_solved:
                print(f"Found wildcard {len(solution[puzzle._id])}->{move_idx+1}")
                break
        new_solution.append(p.permutations)
    export_solution(PUZZLES, new_solution)


def search_branching_shortcuts(puzzle, permutation, pattern_map, depth=1):
    candidate_shortcuts = []
    p = sympy.combinatorics.Permutation(list(range(puzzle.size)))
    allowed_moves = {
        move_id: sympy.combinatorics.Permutation(puzzle._allowed_moves[move_id])
        for move_id in puzzle.current_allowed_move_ids
    }
    for idx1, move_id in enumerate(permutation):
        p *= allowed_moves[move_id]
        for move_id2 in puzzle.current_allowed_move_ids:
            p_new = p * allowed_moves[move_id2]
            p_new_hash = p_new
            if p_new_hash in pattern_map:
                idx2 = pattern_map[p_new_hash][0]
                if idx2 - idx1 > depth:
                    candidate_shortcuts.append((idx1, idx2, move_id2))
    if len(candidate_shortcuts) == 0:
        return None

    _, longest_shortcut = argmax(candidate_shortcuts, key=lambda x: x[1] - x[0])

    idx1, idx2, move_id = longest_shortcut
    new_permutation = permutation[: idx1 + 1] + [move_id] + permutation[idx2 + 1 :]

    return new_permutation


def shortcut(args):
    solution = read_solution(filename=args.initial_solution_file)

    if args.puzzle_id is None and args.puzzle_type is None:
        print("Must specify either --puzzle_id/-p or --puzzle_type/-t")
        print("Available puzzle types:")
        for p in PUZZLE_TYPES:
            print(f"  '{p}'")
        return

    if args.puzzle_type is not None:
        if args.puzzle_type not in PUZZLE_TYPES:
            print(f"Invalid puzzle type {args.puzzle_type}")
            return

    for puzzle in PUZZLES:
        if args.puzzle_id is not None and puzzle._id != args.puzzle_id:
            continue

        if (
            args.puzzle_type is not None
            and args.puzzle_type != "all"
            and not puzzle.type.startswith(args.puzzle_type)
        ):
            continue
        start = time.time()

        print(
            f"Searching shortcuts for puzzle {puzzle._id} ({puzzle.type})                               "
        )
        has_shortcut = True
        while has_shortcut:
            has_shortcut = False
            p = puzzle.clone()
            pattern_map = {p.current_pattern_hash: [0]}
            for move_idx, move_id in enumerate(solution[puzzle._id]):
                p.permutate(move_id)
                id = p.current_pattern_hash
                pattern_map[id] = pattern_map.get(id, []) + [move_idx]
            candidate_shortcuts = [
                positions for positions in pattern_map.values() if len(positions) > 1
            ]
            if len(candidate_shortcuts) == 0 and len(solution[puzzle._id]) > 5000:
                continue

            if len(candidate_shortcuts) == 0:
                shortcut_solution = search_branching_shortcuts(
                    puzzle.clone(), solution[puzzle._id], pattern_map
                )
                if shortcut_solution:
                    print(
                        f"Searching branching shortcuts for {puzzle._id}: {len(solution[puzzle._id])}->{len(shortcut_solution)}"
                    )
                    solution[puzzle._id] = shortcut_solution
                    export_solution(PUZZLES, solution)
                    has_shortcut = True
                continue
            has_shortcut = True
            _, longest_shortcut = argmax(
                candidate_shortcuts, key=lambda x: max(x) - min(x)
            )
            idx1 = min(longest_shortcut)
            idx2 = max(longest_shortcut)
            new_permutation = solution[puzzle._id][: idx1 + 1] + solution[puzzle._id][idx2 + 1 :]
            print(
                f"Searching shortcuts for {puzzle._id}: {len(solution[puzzle._id])}->{len(new_permutation)}"
            )
            solution[puzzle._id] = new_permutation
            export_solution(PUZZLES, solution)

        print(f"Shortcut {puzzle._id} time: {time.time()-start:0.2f}s")


def is_solution_file(filepath):
    if filepath.suffix != ".csv":
        return False
    with open(filepath, "r") as f:
        if f.readline().startswith("id,moves"):
            return True
    return False


def ensemble(args):
    all_files = []
    for filename in args.solution_files_or_folders:
        print(filename)
        # Check if a folder was passed
        if Path(filename).is_dir():
            for file in os.listdir(filename):
                all_files.append(Path(filename) / file)
        else:
            all_files.append(Path(filename))

    print(all_files)
    solution_files = [filename for filename in all_files if is_solution_file(filename)]
    print(solution_files)
    solutions = []
    for filename in solution_files:
        solutions.append(read_solution(filename))

    print(f"Loaded {len(solutions)} solutions:")
    for i, solution in enumerate(solutions):
        print(f"Solution {i} - {solution_files[i]}: {calculate_score(solution):0,}")
    print(
        "*Note: No validation is done on the solutions, only moves are counted. "
        "Assure they are valid using `evaluate`"
    )
    ensemble_solution = []

    for i in range(398):
        sol_lengths = [len(sol.get(i, [])) for sol in solutions]
        print(i, sol_lengths, end=" ")
        ensemble_solution.append(
            sorted(
                [sol.get(i) for sol in solutions if sol.get(i)], key=lambda x: len(x)
            )[0]
        )
        print(len(ensemble_solution[-1]))
    export_solution(PUZZLES, ensemble_solution)


def stitch(args):
    all_files = []
    for filename in args.solution_files_or_folders:
        # Check if a folder was passed
        if Path(filename).is_dir():
            for file in os.listdir(filename):
                all_files.append(Path(filename) / file)
        else:
            all_files.append(Path(filename))

    solution_files = [filename for filename in all_files if is_solution_file(filename)]
    solutions = []
    for filename in solution_files:
        solutions.append(read_solution(filename))

    print(f"Loaded {len(solutions)} solutions:")
    for i, solution in enumerate(solutions):
        print(f"Solution {i} - {solution_files[i]}: {calculate_score(solution):0,}")
    print(
        "*Note: No validation is done on the solutions, only moves are counted. "
        "Assure they are valid using `evaluate`"
    )
    stitched_solution = {}

    for puzzle in PUZZLES:
        skip = False
        if args.puzzle_id is not None and puzzle._id != args.puzzle_id:
            skip = True

        if (
            args.puzzle_type is not None
            and args.puzzle_type != "all"
            and not puzzle.type.startswith(args.puzzle_type)
        ):
            skip = True

        all_solutions = []
        for filename, sol in zip(solution_files, solutions):
            if puzzle._id in sol:
                all_solutions.append(sol[puzzle._id])
        if len(all_solutions) == 0:
            continue

        all_solutions = sorted(all_solutions, key=lambda x: len(x))
        if skip:
            stitched_solution[puzzle._id] = all_solutions[0]
            continue

        unique_solutions = set()
        for sol in all_solutions:
            if args.cleanup:
             unique_solutions.add(tuple(clean_solution(puzzle, sol)))
            else:
                unique_solutions.add(tuple(sol))

        unique_solutions = list(sorted(unique_solutions, key=lambda x: len(x)))
        stitched_solution[puzzle._id] = unique_solutions[0]

        # if len(unique_solutions) > 5:
        #     unique_solutions = unique_solutions[:3] + unique_solutions[-2:]
        print(f"Stitching {puzzle.type} puzzle {puzzle._id} with {len(unique_solutions)} unique solutions", end="")

        pattern_map = {}
        for j, sol in enumerate(unique_solutions):
            p = puzzle.clone()
            for move_idx, move_id in enumerate(sol):
                p.permutate(move_id)
                id = p.current_pattern_hash
                pattern_map[id] = pattern_map.get(id, []) + [(j, move_idx+1)]

        candidate_shortcuts = [
            positions for positions in pattern_map.values() if len(positions) > 1
        ]
        if len(candidate_shortcuts) == 0:
            print(", no intersection patterns found")
            continue

        candidate_solutions = set(
            pt[0] for c in candidate_shortcuts for pt in c
        )
        solution_pts = {
            j: []
            for j in candidate_solutions
        }
        for c in candidate_shortcuts:
            for pt in c:
                solution_pts[pt[0]].append(pt[1])
        for j, idxs in solution_pts.items():
            solution_pts[j] = sorted(idxs)

        if args.debug:
            print()
            for j, idxs in solution_pts.items():
                print(f"Solution {j} indices: {idxs}")

        stitch_pts = [(-1, 0), (-1, 1)]
        if args.debug:
            print(f"Pattern map:")
        for v in candidate_shortcuts:
            if args.debug:
                print(v)
            for pt in v:
                stitch_pts.append(pt)

        pattern_graph = nx.DiGraph()
        pattern_graph.add_nodes_from(stitch_pts)
        for j, idxs in solution_pts.items():
            pattern_graph.add_edge((-1, 0), (j, idxs[0]), weight=idxs[0])
            for idx1, idx2 in zip(idxs[:-1], idxs[1:]):
                pattern_graph.add_edge((j, idx1), (j, idx2), weight=idx2 - idx1)
            if puzzle.clone().full_permutation(unique_solutions[j]).is_solved:
                # Accept invalid solutions, but won't generate an invalid solution
                pattern_graph.add_edge((j, idxs[-1]), (-1, 1), weight=len(unique_solutions[j])-idxs[-1])
        for pts in candidate_shortcuts:
            for pt1 in pts:
                for pt2 in pts:
                    if pt1 == pt2:
                        continue
                    pattern_graph.add_edge(
                        pt1, pt2, weight=0
                    )
                    pattern_graph.add_edge(
                        pt2, pt1, weight=0
                    )

        if args.debug:
            print(f"Search graph number of nodes: {pattern_graph.number_of_nodes()}")
            print(f"Search graph number of edges: {pattern_graph.number_of_edges()}")

        shortest_path = nx.algorithms.shortest_paths.weighted.single_source_dijkstra(
            pattern_graph, (-1, 0), (-1, 1)
        )
        previous_length = min(len(sol) for sol in unique_solutions)
        new_length = shortest_path[0]
        if args.debug:
            print(f"Shortest path: {shortest_path}")
        if new_length < previous_length:
            sol, idx = shortest_path[1][1]
            new_solution = unique_solutions[sol][:idx]
            curr_sol = sol
            curr_idx = idx
            for sol, idx in shortest_path[1][2:-1]:
                w = pattern_graph.get_edge_data((curr_sol, curr_idx), (sol, idx))["weight"]
                if w > 0:
                    # not a shortcut
                    assert curr_sol == sol
                    new_solution += unique_solutions[sol][curr_idx:idx]
                curr_sol = sol
                curr_idx = idx
            new_solution += unique_solutions[sol][curr_idx:]
            print(len(new_solution), new_length)
            assert puzzle.clone().full_permutation(new_solution).is_solved
            assert len(new_solution) == new_length
            print(f"\t{previous_length}->{new_length} ****improved*****")
            stitched_solution[i] = new_solution
            export_solution(PUZZLES, stitched_solution)
        else:
            print(f", shortest path ({new_length}) >= current best ({previous_length}) (no improvement)")
