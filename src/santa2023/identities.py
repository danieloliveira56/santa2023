import math
import time
from typing import Iterable

from santa2023.puzzle import Permutation, read_puzzle_info, read_puzzles
from santa2023.utils import PUZZLE_TYPES, export_solution, get_inverse, read_solution, calculate_score
import sympy.combinatorics

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
    puzzles = read_puzzles("puzzles.csv")
    all_puzzle_info = read_puzzle_info("puzzle_info.csv")
    for p in puzzles:
        p.initialize_move_list(all_puzzle_info[p.type])

    puzzle_type = 'globe_2/6'
    print(f"Puzzle type '{puzzle_type}' permutations:")

    for key, value in all_puzzle_info[puzzle_type].items():
        print(f"{key}:", sympy.combinatorics.Permutation(value))
        print("\t", sympy.combinatorics.Permutation(value).array_form)

    keys = list(all_puzzle_info[puzzle_type].keys())

    permutations = [
        sympy.combinatorics.Permutation(p)
        for p in all_puzzle_info[puzzle_type].values()
    ]

    for i in range(len(permutations)):
        for j in range(i+1, len(permutations)):
            if permutations[i]*permutations[j] == permutations[j]* permutations[i]:
                print(f"Permutation {keys[i]},{keys[j]} are commutative")

    for i in range(len(permutations)):
        if permutations[i] == ~permutations[i]:
            print(f"Permutation {keys[i]} is its own inverse")


    for i in range(len(permutations)):
        for j in range(i+1, len(permutations)):
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
    print(p1*p2*(~p1))

def fast_identities(args):
    solution = read_solution(filename=args.initial_solution_file)
    puzzles = read_puzzles("puzzles.csv")
    all_puzzle_info = read_puzzle_info("puzzle_info.csv")
    for p in puzzles:
        p.initialize_move_list(all_puzzle_info[p.type])

    # max_len = 0
    # for p in puzzles:
    #     max_len = max(max_len, len(p._solution))
    # print(f"Max solution length: {max_len}")
    # exit()

    puzzle_type = args.puzzle_type
    if puzzle_type == "all":
        puzzle_types = PUZZLE_TYPES
        puzzle_types.pop()
    else:
        puzzle_types = [puzzle_type]

    # for puzzle_type, puzzle_info in all_puzzle_info.items():
    #     print(puzzle_type, len(puzzle_info))
    # exit()

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

            # size = len(permutations)
            # size = 3
            # print(f"size: {size}")
            # while size < len(permutations) and time.time() - start < int(args.max_time):
            #     for i in range(len(permutations) - size):
            start = time.time()
            i = 0
            while i < len(permutations):
                p = permutations[i]
                j = i+1
                while j < len(permutations):
                    print(f'Searching puzzle {puzzle._id} position {i:5d}-{j:5d}/{len(permutations):6d}             ', end='\r')
                    p *= permutations[j]
                    id = p.mapping
                    if id in shortest_permutations:
                        if shortest_permutations[id] < p:
                            permutations[i:j+1] = shortest_permutations[id].split(all_puzzle_info[puzzle.type])
                            print(
                                f"Searching puzzle {puzzle._id} ({puzzle.type}) [{i}:{j}](size={j-i+1}) ({initial_len})->({len(permutations)})"
                            )
                            print(f"Map size: {len(shortest_permutations):0,}")

                            j = i + len(shortest_permutations[id]) - 1

                            # p_check = permutations[i]
                            # for k in range(i+1, j+1):
                            #     p_check *= permutations[k]
                            # assert p_check.mapping == shortest_permutations[id].mapping
                        elif p < shortest_permutations[id]:
                            shortest_permutations[id] = p
                        else:
                            shortest_permutations[id] = p
                    else:
                        shortest_permutations[id] = p
                    j += 1
                i += 1

                    #     if shortest_permutations[id] < p:
                    #         permutations[i : i + size + 1] = shortest_permutations[
                    #             id
                    #         ].split(all_puzzle_info[puzzle.type])
                    #         print(
                    #             f"Searching puzzle {puzzle._id} ({puzzle.type}) [{i}:{i+size+1}](size={size}) ({initial_len})->({len(permutations)})"
                    #         )
                    #     elif p < shortest_permutations[id]:
                    #         shortest_permutations[id] = p
                    # else:
                    #     shortest_permutations[id] = p
                # size += 1

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
    puzzles = read_puzzles("puzzles.csv")
    puzzle_info = read_puzzle_info("puzzle_info.csv")
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


def shortcut(args):
    solution = read_solution(filename=args.initial_solution_file)
    puzzles = read_puzzles("puzzles.csv")
    puzzle_info = read_puzzle_info("puzzle_info.csv")
    for p in puzzles:
        p.initialize_move_list(puzzle_info[p.type])
    new_solution = []
    for puzzle in puzzles:
        print(f"Searching shortcuts for {puzzle._id}", end="\r")
        new_permutation = solution[puzzle._id].copy()
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
            print(
                f"Searching shortcuts for {puzzle._id}: {len(solution[puzzle._id])}->{len(new_permutation)}"
            )

        new_solution.append(new_permutation)

    export_solution(puzzles, new_solution)

    # def sorting():
    #     solution = read_solution(filename=args.initial_solution_file)
    #     puzzles = read_puzzles("puzzles.csv")
    #     puzzle_info = read_puzzle_info("puzzle_info.csv")
    #     for p in puzzles:
    #         p.initialize_move_list(puzzle_info[p.type])
    #
    #     new_solution = []
    #     for permutation, puzzle in zip(solution, puzzles):
    #         if "cube" not in puzzle.type:
    #             new_solution.append(permutation)
    #             continue
    #         print(f"Sorting {puzzle.type} puzzle {puzzle._id}", end="\r")
    #         new_permutation = []
    #         current_letter = permutation[0].replace('-', '')[0]
    #         current_group = []
    #         for move in permutation:
    #             if current_letter in move:
    #                 current_group.append(move)
    #             else:
    #                 current_group

    export_solution(puzzles, new_solution)


def ensemble(args):
    puzzles = read_puzzles("puzzles.csv")
    solutions = []
    for filename in args.solution_files:
        solutions.append(read_solution(filename))

    print(f"Loaded {len(solutions)} solutions:")
    for i, solution in enumerate(solutions):
        print(f"Solution {i} - {args.solution_files[i]}: {calculate_score(solution):0,}")
    print()
    ensemble_solution = []

    for i in range(398):
        sol_lengths = [len(sol.get(i, [])) for sol in solutions]
        print(i, sol_lengths, end=" ")
        ensemble_solution.append(sorted([sol.get(i) for sol in solutions if sol.get(i)], key=lambda x: len(x))[0])
        print(len(ensemble_solution[-1]))
    export_solution(puzzles, ensemble_solution)