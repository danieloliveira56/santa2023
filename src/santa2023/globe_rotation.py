import json
import os
import time
from typing import List, Literal

import sympy.combinatorics

from santa2023.puzzle import Permutation, Puzzle
from santa2023.utils import (cache_translation, clean_globe_solution,
                             move_letter, print_solution)


@cache_translation
def get_globe_rotation_translations(
    puzzle_type, allowed_moves: dict[str, Permutation]
) -> dict[str, dict[str, str]]:
    assert puzzle_type.startswith("globe"), "Not a globe puzzle type"

    start = time.time()
    print(f"Getting {puzzle_type} rotation translation")
    globe_long_size = int(puzzle_type.split("_")[1].split("/")[1])
    globe_lat_size = int(puzzle_type.split("_")[1].split("/")[0])
    permutations = {
        move_id: sympy.combinatorics.Permutation(allowed_moves[move_id])
        for move_id in allowed_moves.keys()
    }
    rotation_translations = {}
    for i in range(globe_long_size):
        move1 = f"f{i}"
        move2 = f"f{i + globe_long_size}"
        rotation_translations[f"{move1}.{move2}"] = {}
        rotation = permutations[move1] * permutations[move2]
        for id1, p1 in permutations.items():
            for id2, p2 in permutations.items():
                if id2.startswith("-f"):
                    continue
                if rotation * p2 * ~rotation == p1:
                    rotation_translations[f"{move1}.{move2}"][id1] = id2
    rotation_translations[f"r"] = {}
    r_rotation = permutations["r0"]
    for i in range(1, globe_lat_size + 1):
        r_rotation *= permutations[f"r{i}"]

    for direction in range(-2 * globe_long_size, 2 * globe_long_size + 1):
        if direction == 0:
            continue
        # r and -r movements clearly won't change
        # rotation_translations["r"][str(direction)] = {
        #     f"r{i}": f"r{i}" for i in range(globe_lat_size + 1)
        # } | {f"-r{i}": f"-r{i}" for i in range(globe_lat_size + 1)}

        rotation_translations["r"][str(direction)] = {}
        rotation = r_rotation
        for _ in range(1, abs(direction)):
            rotation *= r_rotation
        if direction < 0:
            rotation = ~rotation
        for id1, p1 in permutations.items():
            # if move_letter(id1) == "r":
            #     continue
            for id2, p2 in permutations.items():
                if id2.startswith("-f"):
                    continue
                if rotation * p2 * ~rotation == p1:
                    rotation_translations["r"][str(direction)][id2] = id1

    print(
        f"Got {puzzle_type} rotation translation in {time.time() - start:.2f} seconds"
    )

    return rotation_translations


def f_rotate_globe_solution(
    solution: List[str],
    globe_long_size: int,
    longitude: int,
    sequence_start: int,
    sequence_end: int,
    rotation_transformation: dict[str, dict[str, str]],
) -> List[str]:
    """
    Rotates a sequence of moves in a globe solution

    :param solution: solution to rotate
    :param globe_long_size: size of the globe, e.g. 3 for a globe_3/3 puzzle
    :param longitude: longitude to rotate around
    :param sequence_start: start index of the sequence to rotate
    :param sequence_end: end index of the sequence to rotate
    :param rotation_transformation: dictionary of rotation transformations for each axis/direction
    :return: new solution with the rotated sequence
    """
    new_solution = solution.copy()
    new_solution.insert(sequence_end, f"f{longitude}")
    new_solution.insert(sequence_end, f"f{longitude + globe_long_size}")

    new_solution.insert(sequence_start, f"f{longitude}")
    new_solution.insert(sequence_start, f"f{longitude + globe_long_size}")

    for i in range(
        sequence_start + 2,
        sequence_end + 2,
    ):
        new_solution[i] = rotation_transformation[
            f"f{longitude}.{f'f{longitude + globe_long_size}'}"
        ][new_solution[i]]

    return new_solution


def r_rotate_globe_solution(
    solution: List[str],
    globe_lat_size: int,
    direction: int,
    sequence_start: int,
    sequence_end: int,
    rotation_transformation: dict[str, dict[str, str]],
) -> List[str]:
    """
    Rotates a sequence of moves in a globe solution

    :param solution: solution to rotate
    :param globe_long_size: size of the globe, e.g. 3 for a globe_3/3 puzzle
    :param longitude: longitude to rotate around
    :param sequence_start: start index of the sequence to rotate
    :param sequence_end: end index of the sequence to rotate
    :param rotation_transformation: dictionary of rotation transformations for each axis/direction
    :return: new solution with the rotated sequence
    """
    new_solution = solution.copy()

    for i in range(globe_lat_size + 1):
        for _ in range(abs(direction)):
            if direction > 0:
                new_solution.insert(sequence_end, f"-r{i}")
            else:
                new_solution.insert(sequence_end, f"r{i}")

    for i in range(globe_lat_size + 1):
        for _ in range(abs(direction)):
            if direction > 0:
                new_solution.insert(sequence_start, f"r{i}")
            else:
                new_solution.insert(sequence_start, f"-r{i}")

    for i in range(
        sequence_start + (globe_lat_size + 1) * abs(direction),
        sequence_end + (globe_lat_size + 1) * abs(direction),
    ):
        new_solution[i] = rotation_transformation["r"][str(direction)][new_solution[i]]
    return new_solution


def eliminate_globe_rotations(solution: List[str], puzzle: Puzzle, debug=False):
    globe_lat_size = int(puzzle.type.split("_")[1].split("/")[0])
    globe_long_size = int(puzzle.type.split("_")[1].split("/")[1])
    temp_solution = solution.copy()

    rotation_transformation = get_globe_rotation_translations(
        puzzle.type, puzzle.allowed_moves
    )

    try:
        has_rotation = True
        while has_rotation:
            has_rotation = False
            for i in range(globe_long_size):
                if debug:
                    print(
                        f"Eliminating globe {puzzle._id} rotation f{i}.f{i + globe_long_size}"
                    )

                rotation_ids = {
                    f"f{i}",
                    f"f{i + globe_long_size}",
                    f"-f{i}",
                    f"-f{i + globe_long_size}",
                }
                j = 0
                group_ct_map = {}
                while j < len(temp_solution):
                    while (
                        j < len(temp_solution) and temp_solution[j] not in rotation_ids
                    ):
                        j += 1

                    current_group_idx = j
                    group_ct_map[current_group_idx] = 0
                    while j < len(temp_solution) and temp_solution[j] in rotation_ids:
                        group_ct_map[current_group_idx] += 1
                        j += 1

                if debug:
                    print_solution(temp_solution, debug=True)
                    print("group_ct_map")
                    for k, v in group_ct_map.items():
                        print(f"|  {k}: {v}")

                insertion_costs = []
                for insertion_idx1 in group_ct_map.keys():
                    for insertion_idx2 in group_ct_map.keys():
                        if insertion_idx1 >= insertion_idx2:
                            continue
                        net_cost1 = 2 - 2 * group_ct_map[insertion_idx1]
                        net_cost2 = 2 - 2 * group_ct_map[insertion_idx2]
                        cost = net_cost1 + net_cost2

                        if cost > 0:
                            continue
                        # Attempts to push extra moves to the end of the sequence
                        if cost == 0 and net_cost2 <= 0:
                            continue
                        insertion_costs.append((insertion_idx1, insertion_idx2, cost))
                if len(insertion_costs) == 0:
                    continue

                insertion_costs = sorted(insertion_costs, key=lambda x: x[2])

                if debug:
                    print("Insertion cost:")
                    print("insertion_idx1, insertion_idx2, direction, cost")
                    for insertion_idx1, insertion_idx2, cost in insertion_costs:
                        print(f"{insertion_idx1:14d}, {insertion_idx2:14d}, {cost:4d}")

                insertion_idx1, insertion_idx2, cost = insertion_costs[0]
                print(
                    f"  f{i}.f{i+globe_long_size}-rotating solution[{insertion_idx1:5d}, {insertion_idx2:5d}]",
                    end="",
                )
                print(
                    f", expected diff: {cost:2d}, actual: {len(temp_solution)}",
                    end="->",
                )

                if debug:
                    solved_puzzle = puzzle.clone().full_permutation(temp_solution)
                    assert (
                        solved_puzzle.is_solved
                    ), f"Initially Unsolved:\n{solved_puzzle}"

                previous_length = len(temp_solution)
                temp_solution = f_rotate_globe_solution(
                    temp_solution,
                    globe_long_size,
                    longitude=i,
                    sequence_start=insertion_idx1,
                    sequence_end=insertion_idx2,
                    rotation_transformation=rotation_transformation,
                )

                if debug:
                    solved_puzzle = puzzle.clone().full_permutation(solution)
                    assert (
                        solved_puzzle.is_solved
                    ), f"Unsolved rotated cube:\n{solved_puzzle}"

                print(len(temp_solution), end="->")
                temp_solution = clean_globe_solution(puzzle, temp_solution)
                print(len(temp_solution), end="")
                actual_diff = len(temp_solution) - previous_length
                print(f" ({actual_diff}) ", end="")
                if cost != actual_diff:
                    print("\u274c")
                else:
                    print("\u2713")

                if debug:
                    solved_puzzle = puzzle.clone().full_permutation(temp_solution)
                    assert (
                        solved_puzzle.is_solved
                    ), f"Unsolved rotated + cleaned-up cube:\n{solved_puzzle}"

                if len(temp_solution) <= len(solution):
                    solution = temp_solution.copy()
                    has_rotation = True
                    break

            if has_rotation:
                continue

            if debug:
                print(f"Eliminating globe {puzzle._id} rotation r")
            group_ct_map = {}
            total_group_ct_map = {}
            j = 0
            while j < len(temp_solution):
                while j < len(temp_solution) and move_letter(temp_solution[j]) != "r":
                    j += 1

                current_group_idx = j
                group_ct = {}
                total_group_ct_map[current_group_idx] = 0
                while j < len(temp_solution) and move_letter(temp_solution[j]) == "r":
                    move_id = temp_solution[j]
                    group_ct[move_id] = group_ct.get(move_id, 0) + 1
                    total_group_ct_map[current_group_idx] += 1
                    j += 1

                group_ct_map[current_group_idx] = group_ct

            if debug:
                print_solution(temp_solution, debug=True)
                print("group_ct_map")
                for k, v in group_ct_map.items():
                    print(f"|  {k}: {v}")

            insertion_costs = []
            for insertion_idx1 in group_ct_map.keys():
                for insertion_idx2 in group_ct_map.keys():
                    if insertion_idx1 >= insertion_idx2:
                        continue
                    if total_group_ct_map[insertion_idx1] + total_group_ct_map[
                        insertion_idx2
                    ] < 2 * (globe_lat_size + 1):
                        continue

                    directions_to_try = set()
                    for move_id, ct in group_ct_map[insertion_idx1].items():
                        if move_id.startswith("r"):
                            directions_to_try.add(-ct)
                            directions_to_try.add(2 * globe_long_size - ct)
                        if move_id.startswith("-r"):
                            directions_to_try.add(ct)
                            directions_to_try.add(-2 * globe_long_size + ct)
                    for move_id, ct in group_ct_map[insertion_idx2].items():
                        if move_id.startswith("r"):
                            directions_to_try.add(-ct)
                            directions_to_try.add(2 * globe_long_size - ct)
                        if move_id.startswith("-r"):
                            directions_to_try.add(ct)
                            directions_to_try.add(-2 * globe_long_size + ct)

                    for direction in directions_to_try:
                        net_cost1 = (globe_lat_size + 1) * abs(direction)
                        net_cost2 = (globe_lat_size + 1) * abs(direction)

                        for move_id, ct in group_ct_map[insertion_idx1].items():
                            if move_id.startswith("r") and direction < 0:
                                net_cost1 -= 2 * min(ct, abs(direction))
                            # if move_id.startswith("-r") and direction > 0 and ct + direction > globe_lat_size:
                            #    net_cost1 -= globe_lat_size
                            if move_id.startswith("-r") and direction > 0:
                                net_cost1 -= 2 * min(ct, abs(direction))
                            # if move_id.startswith("-r") and direction < 0 and ct + abs(direction) > globe_lat_size:
                            #    net_cost1 -= globe_lat_size
                        for move_id, ct in group_ct_map[insertion_idx2].items():
                            if move_id.startswith("r") and direction > 0:
                                net_cost2 -= 2 * min(ct, abs(direction))
                            # if move_id.startswith("r") and direction < 0 and ct + abs(direction) > globe_lat_size:
                            #    net_cost2 -= globe_lat_size
                            if move_id.startswith("-r") and direction < 0:
                                net_cost2 -= 2 * min(ct, abs(direction))
                            # if move_id.startswith("-r") and direction > 0 and ct + abs(direction) > globe_lat_size:
                            #     net_cost2 -= globe_lat_size

                        cost = net_cost1 + net_cost2

                        if cost > 0:
                            continue
                        # Attempts to push extra moves to the end of the sequence if cost == 0 is accepted above
                        if cost == 0 and net_cost2 <= 0:
                            continue

                        insertion_costs.append(
                            (insertion_idx1, insertion_idx2, direction, cost)
                        )

            insertion_costs = sorted(
                insertion_costs, key=lambda x: -x[0] + (x[3] / 100_000)
            )

            if debug:
                for insertion_idx1, insertion_idx2, direction, cost in insertion_costs:
                    print(insertion_idx1, insertion_idx2, direction, cost)
            lowest_insert = len(temp_solution) + 1
            for insertion_idx1, insertion_idx2, direction, cost in insertion_costs:
                assert (
                    insertion_idx1 <= lowest_insert
                ), f"insertion_idx1: {insertion_idx1}, lowest_insert: {lowest_insert}"
                if insertion_idx2 >= lowest_insert:
                    continue
                lowest_insert = insertion_idx1
                print(
                    f"  {direction:3d}r-rotating {puzzle.type} puzzle {puzzle._id} solution[{insertion_idx1:5d}, {insertion_idx2:5d}]",
                    end="",
                )
                print(
                    f", expected diff: {cost:2d}, actual: {len(temp_solution)}",
                    end="->",
                )

                if debug:
                    solved_puzzle = puzzle.clone().full_permutation(temp_solution)
                    assert (
                        solved_puzzle.is_solved
                    ), f"Initially Unsolved:\n{solved_puzzle}"

                previous_length = len(temp_solution)
                temp_solution = r_rotate_globe_solution(
                    temp_solution,
                    globe_lat_size=globe_lat_size,
                    direction=direction,
                    sequence_start=insertion_idx1,
                    sequence_end=insertion_idx2,
                    rotation_transformation=rotation_transformation,
                )

                if debug:
                    solved_puzzle = puzzle.clone().full_permutation(temp_solution)
                    assert (
                        solved_puzzle.is_solved
                    ), f"Unsolved rotated cube:\n{solved_puzzle}"

                print(len(temp_solution), end="->")
                temp_solution = clean_globe_solution(puzzle, temp_solution)
                actual_diff = len(temp_solution) - previous_length
                print(f"{len(temp_solution)} ({actual_diff}) ", end="")
                if cost != actual_diff:
                    print("\u274c")
                else:
                    print("\u2713")

                if debug:
                    solved_puzzle = puzzle.clone().full_permutation(temp_solution)
                    assert (
                        solved_puzzle.is_solved
                    ), f"Unsolved rotated + cleaned-up cube:\n{solved_puzzle}"

                if len(temp_solution) <= len(solution):
                    solution = temp_solution.copy()
                    has_rotation = True

    except KeyboardInterrupt:
        return solution
        pass
    except Exception as e:
        raise e

    return solution
