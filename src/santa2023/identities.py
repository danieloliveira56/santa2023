import math
import os
import time
from pathlib import Path
from typing import Iterable

import networkx as nx
import sympy.combinatorics

from santa2023.puzzle import Permutation, read_puzzle_info, read_puzzles
from santa2023.utils import (
    CSV_BASE_PATH,
    PUZZLE_TYPES,
    calculate_score,
    export_solution,
    get_inverse,
    read_solution,
)


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
    puzzles = read_puzzles(CSV_BASE_PATH / "puzzles.csv")
    puzzle_info = read_puzzle_info(CSV_BASE_PATH / "puzzle_info.csv")
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
    export_solution(puzzles, new_solution)


def test(args):
    solution = read_solution(filename=args.initial_solution_file)
    puzzles = read_puzzles(CSV_BASE_PATH / "puzzles.csv")
    all_puzzle_info = read_puzzle_info(CSV_BASE_PATH / "puzzle_info.csv")
    for p in puzzles:
        p.initialize_move_list(all_puzzle_info[p.type])

    puzzle_type = "globe_2/6"
    puzzle_type = "cube_5/5/5"
    print(f"Puzzle type '{puzzle_type}' permutations:")
    puzzle_size = [puzzle for puzzle in puzzles if puzzle.type == puzzle_type][0].size()
    # Create graph of size puzzle_size
    position_graph = nx.DiGraph()
    position_graph.add_nodes_from(range(puzzle_size))

    taboo_list = {3: {}}
    taboo_list[3][""] = [move_id for move_id in all_puzzle_info[puzzle_type].keys()]
    for move_id in all_puzzle_info[puzzle_type].keys():
        taboo_list[3][""].append(f"-{move_id}")

    for f in "fdr":
        for i in range(3):
            taboo_list[3][f"{f}{i}"] = (
                [f"{f}{j}" for j in range(i, 3)]
                + [f"{d}{j}" for j in range(3) for d in "fdr" if d != f]
                + [f"-{f}{j}" for j in range(i + 1, 3)]
                + [f"-{d}{j}" for j in range(3) for d in "fdr" if d != f]
            )
            taboo_list[3][f"-{f}{i}"] = (
                [f"{f}{j}" for j in range(i + 1, 3)]
                + [f"{d}{j}" for j in range(3) for d in "fdr" if d != f]
                + [f"-{f}{j}" for j in range(i + 1, 3)]
                + [f"-{d}{j}" for j in range(3) for d in "fdr" if d != f]
            )
    print(taboo_list)

    for key, value in all_puzzle_info[puzzle_type].items():
        p = sympy.combinatorics.Permutation(value)
        print(f"{key}:", p)
        # print("\t", sympy.combinatorics.Permutation(value).array_form)
        for group in p.cyclic_form:
            if len(group) > 1:
                for i in range(len(group)):
                    position_graph.add_edge(group[i], group[(i + 1) % len(group)])

    print(f"Graph size: {len(position_graph)}")
    print(f"Graph edges: {len(position_graph.edges)}")
    # Find connected components
    connected_components = list(nx.weakly_connected_components(position_graph))
    groups = []
    group_moves = {}
    group_reindex = {}
    group_cost_database = {}
    for i, component in enumerate(connected_components):
        group_moves[i] = {}
        print(f"Component {i}: {len(component)}")
        print(component)
        groups.append(component)
        group_reindex[i] = {k: j for j, k in enumerate(component)}
        # print(group_reindex[i])
        move_ct = 0
        for move_id, move_mapping in all_puzzle_info[puzzle_type].items():
            # for j in component:
            #     # print(j, end="->")
            #     # print(move_mapping[j], end="->")
            #     print(group_reindex[i][move_mapping[j]])
            group_moves[i][move_id] = [
                group_reindex[i][move_mapping[j]] for j in component
            ]
            group_moves[i][f"-{move_id}"] = get_inverse(group_moves[i][move_id])
            if list(group_moves[i][move_id]) != list(range(len(component))):
                print(f"\t{move_id}: {group_moves[i][move_id]}")
                move_ct += 1
        print(f"\tMove count: {move_ct}")

    for puzzle in puzzles:
        if puzzle.type != puzzle_type:
            continue
        print(f"Testing puzzle {puzzle._id}")
        for i, group in enumerate(groups):
            group_solution = [group_reindex[i][j] for j in solution[puzzle._id]]
            print(f"Group {i}: {group_solution}")
            print(f"Cost: {group_cost_database[i][group_solution]}")

    keys = list(all_puzzle_info[puzzle_type].keys())

    permutations = [
        sympy.combinatorics.Permutation(p)
        for p in all_puzzle_info[puzzle_type].values()
    ]

    for i in range(len(permutations)):
        for j in range(i + 1, len(permutations)):
            if permutations[i] * permutations[j] == permutations[j] * permutations[i]:
                print(f"Permutation {keys[i]},{keys[j]} are commutative")

    for i in range(len(permutations)):
        if permutations[i] == ~permutations[i]:
            print(f"Permutation {keys[i]} is its own inverse")

    for i in range(len(permutations)):
        for j in range(i + 1, len(permutations)):
            if permutations[i] == ~permutations[j]:
                print(f"Permutation {keys[i]},{keys[j]} are inverses")
    exit()

    print(f"Puzzle type '{puzzle_type}' permutations:")
    for i, p in enumerate(permutations):
        print(f"Permutation {i}:")
        print(p)
        print(p.array_form)
        print(p.cyclic_form)
        print()

    G = sympy.combinatorics.PermutationGroup(permutations)

    print(f"Group '{puzzle_type}' base:")
    print(G.base)
    print(f"Group '{puzzle_type}' generators:")
    for p in G.strong_gens:
        print(p)

    print()
    p1 = permutations[0]
    p2 = permutations[1]
    print(p1)
    print(p2)
    print(p1 * p2 * (~p1))


def fast_identities(args):
    solution = read_solution(filename=args.initial_solution_file)
    puzzles = read_puzzles(CSV_BASE_PATH / "puzzles.csv")
    all_puzzle_info = read_puzzle_info(CSV_BASE_PATH / "puzzle_info.csv")
    for p in puzzles:
        p.initialize_move_list(all_puzzle_info[p.type])

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
    for m in a:
        print(m, sum([1 if i != j else 0 for i, j in enumerate(m)]))
    exit()

    new_solution = []
    try:
        for permutation, puzzle in zip(solution, puzzles):
            if puzzle.type not in puzzle_types:
                new_solution.append(permutation)
                continue

            permutations = [
                Permutation(all_puzzle_info[puzzle.type][move_id], move_id)
                for move_id in permutation
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
            new_solution.append(permutations_list)

    except KeyboardInterrupt:
        new_solution = new_solution + solution[len(new_solution) :]
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
    puzzles = read_puzzles(CSV_BASE_PATH / "puzzles.csv")
    puzzle_info = read_puzzle_info(CSV_BASE_PATH / "puzzle_info.csv")
    for p in puzzles:
        p.initialize_move_list(puzzle_info[p.type])
    new_solution = []

    for puzzle in puzzles:
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
    export_solution(puzzles, new_solution)


def search_branching_shortcuts(puzzle, permutation, pattern_map, depth=1):
    candidate_shortcuts = []
    for idx1, move_id in enumerate(permutation):
        puzzle.permutate(move_id)
        for move_id2 in puzzle.current_allowed_move_ids:
            p = puzzle.clone()
            p.permutate(move_id2)
            if p.current_pattern_hash in pattern_map:
                idx2 = pattern_map[p.current_pattern_hash][0]
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
    puzzles = read_puzzles(CSV_BASE_PATH / "puzzles.csv")
    puzzle_info = read_puzzle_info(CSV_BASE_PATH / "puzzle_info.csv")
    for p in puzzles:
        p.initialize_move_list(puzzle_info[p.type])
    new_solution = []
    for puzzle in puzzles:
        print(f"Searching shortcuts for {puzzle._id}", end="\r")
        new_permutation = solution[puzzle._id].copy()
        has_shortcut = True
        while has_shortcut and len(new_permutation) < 3000:
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
                if len(new_permutation) > 3000:
                    continue
                shortcut_solution = search_branching_shortcuts(puzzle.clone(), new_permutation, pattern_map)
                if shortcut_solution:
                    print(
                        f"Searching branching shortcuts for {puzzle._id}: {len(solution[puzzle._id])}->{len(shortcut_solution)}"
                    )
                    new_permutation = shortcut_solution
                    has_shortcut = True
                continue
            has_shortcut = True
            _, longest_shortcut = argmax(
                candidate_shortcuts, key=lambda x: max(x) - min(x)
            )
            idx1 = min(longest_shortcut)
            idx2 = max(longest_shortcut)
            new_permutation = new_permutation[: idx1 + 1] + new_permutation[idx2 + 1 :]
            print(
                f"Searching shortcuts for {puzzle._id}: {len(solution[puzzle._id])}->{len(new_permutation)}"
            )

        new_solution.append(new_permutation)

    export_solution(puzzles, new_solution)


def is_solution_file(filepath):
    if filepath.suffix != ".csv":
        return False
    with open(filepath, "r") as f:
        if f.readline().startswith("id,moves"):
            return True
    return False

def ensemble(args):
    puzzles = read_puzzles(CSV_BASE_PATH / "puzzles.csv")

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
    print("*Note: No validation is done on the solutions, only moves are counted. "
          "Assure they are valid using `evaluate`")
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
    export_solution(puzzles, ensemble_solution)
