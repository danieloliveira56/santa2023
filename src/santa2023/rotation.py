from typing import List, Literal

from santa2023.puzzle import Permutation, Puzzle
from santa2023.utils import clean_cube_solution, print_solution


def cache_rotation_transformation(fun):
    cache_rotation_transformation.cache_ = {}

    def inner(cube_size, allowed_moves):
        if cube_size not in cache_rotation_transformation.cache_:
            cache_rotation_transformation.cache_[cube_size] = fun(
                cube_size, allowed_moves
            )
        return cache_rotation_transformation.cache_[cube_size]

    return inner


@cache_rotation_transformation
def get_rotation_transformation(
    cube_size: int, allowed_moves: dict[str, Permutation]
) -> dict[str, dict[str, str]]:
    """
    Returns a dictionary of the form:
    rotation_transformation[axis][move_id] = new_move_id
    where new_move_id is the move_id after rotating the cube 90 degrees around the axis,
    axis can be one of "f", "d", "r", "-f", "-d", "-r", which correspond to rotations clockwise and counterclockwise

    :param cube_size: size of the cube, e.g. 3 for a cube_3/3/3 puzzle
    :param allowed_moves: dictionary of allowed moves, including inverses
    :return dictionary of rotation transformations for each axis,
            e.g. rotation_transformation["f"]["r0"] = "d0", since f0.f1.r0.-f0.-f1 == d0
    """
    permutations = {
        move_id: Permutation(allowed_moves[move_id]) for move_id in allowed_moves.keys()
    }
    rotation_transformation = {}
    for axis in ["f", "d", "r"]:
        rotation_transformation[axis] = {}
        for direction in [-1, 1, 2]:
            rotation_transformation[axis][direction] = {}
            rotation: Permutation = permutations[f"{axis}0"]
            for i in range(1, cube_size):
                rotation *= permutations[f"{axis}{i}"]
            if abs(direction) == 2:
                rotation *= rotation
            if direction < 0:
                rotation = ~rotation

            for id1, p1 in permutations.items():
                for id2, p2 in permutations.items():
                    if rotation * p2 * ~rotation == p1:
                        rotation_transformation[axis][direction][id1] = id2

    # print("Rotation transformation:")
    # for axis in ["f", "d", "r"]:
    #     for direction in [-2, -1, 1, 2]:
    #         print(f"{direction}{axis}:")
    #         for k, v in rotation_transformation[axis][direction].items():
    #             print(f"| {direction}{k} -> {v}")
    return rotation_transformation


def rotate_cube_solution(
    solution: List[str],
    cube_size: int,
    axis: Literal["f", "d", "r"],
    direction: Literal[-1, 1, 2],
    sequence_start: int,
    sequence_end: int,
    rotation_transformation: dict[str, dict[int, dict[str, str]]],
) -> List[str]:
    """
    Rotates a sequence of moves in a cube solution

    :param solution: solution to rotate
    :param cube_size: size of the cube, e.g. 3 for a cube_3/3/3 puzzle
    :param axis: axis to rotate around, one of "f", "d", "r"
    :param direction: direction to rotate, one of -2, -1, 1, 2,
           e.g. -2 means rotate 180 degrees using -f, -r, or -d moves
    :param sequence_start: start index of the sequence to rotate
    :param sequence_end: end index of the sequence to rotate
    :param rotation_transformation: dictionary of rotation transformations for each axis/direction
    :return: new solution with the rotated sequence
    """
    new_solution = solution.copy()
    for i in range(cube_size):
        if direction == 1:
            new_solution.insert(sequence_end, f"-{axis}{i}")
        elif direction == 2:
            new_solution.insert(sequence_end, f"-{axis}{i}")
            new_solution.insert(sequence_end, f"-{axis}{i}")
        elif direction == -1:
            new_solution.insert(sequence_end, f"{axis}{i}")
        elif direction == -2:
            new_solution.insert(sequence_end, f"{axis}{i}")
            new_solution.insert(sequence_end, f"{axis}{i}")

    for i in range(cube_size):
        if direction == 1:
            new_solution.insert(sequence_start, f"{axis}{i}")
        elif direction == 2:
            new_solution.insert(sequence_start, f"{axis}{i}")
            new_solution.insert(sequence_start, f"{axis}{i}")
        elif direction == -1:
            new_solution.insert(sequence_start, f"-{axis}{i}")
        elif direction == -2:
            new_solution.insert(sequence_start, f"-{axis}{i}")
            new_solution.insert(sequence_start, f"-{axis}{i}")

    for i in range(
        sequence_start + abs(direction) * cube_size,
        sequence_end + abs(direction) * cube_size,
    ):
        new_solution[i] = rotation_transformation[axis][direction][new_solution[i]]
    return new_solution


def eliminate_cube_rotations(solution: List[str], puzzle: Puzzle, debug=False):
    """
    Eliminates cube rotations from the solution
    f rotations will replace r0 by rn and d0 by -dn
    r rotations will replace f0 by fn and d0 by -dn
    d rotations will replace f0 by -fn and r0 by rn

    """
    cube_size = int(puzzle.type.split("_")[1].split("/")[0])
    temp_solution = solution.copy()

    rotation_transformation = get_rotation_transformation(
        cube_size, puzzle.allowed_moves
    )

    for group_letter in "fdr":
        if debug:
            print(f"Eliminating {group_letter} rotations")
        while True:
            j = 0
            group_ct_map1 = {}
            group_ct_map_neg1 = {}
            group_ct_map2 = {}
            while j < len(temp_solution):
                while (
                    j < len(temp_solution)
                    and temp_solution[j].replace("-", "")[0] != group_letter
                ):
                    j += 1

                current_group_idx = j
                group_ct = {}
                while (
                    j < len(temp_solution)
                    and temp_solution[j].replace("-", "")[0] == group_letter
                ):
                    move_id = temp_solution[j]
                    group_ct[move_id] = group_ct.get(move_id, 0) + 1
                    j += 1
                group_ct_map1[current_group_idx] = sum(
                    1
                    for move_id, ct in group_ct.items()
                    if ct == 1 and not move_id.startswith("-")
                )
                group_ct_map_neg1[current_group_idx] = sum(
                    1
                    for move_id, ct in group_ct.items()
                    if ct == 1 and move_id.startswith("-")
                )
                group_ct_map2[current_group_idx] = sum(
                    1 for move_id, ct in group_ct.items() if ct == 2
                )

            if debug:
                print_solution(temp_solution, debug=True)
                print("group_ct_map1")
                for k, v in group_ct_map1.items():
                    print(f"|  {k}: {v}")
                print("group_ct_map_neg1")
                for k, v in group_ct_map_neg1.items():
                    print(f"|  {k}: {v}")
                print("group_ct_map2")
                for k, v in group_ct_map2.items():
                    print(f"|  {k}: {v}")

            insertion_cost = []
            for insertion_idx1 in group_ct_map1.keys():
                for insertion_idx2 in group_ct_map1.keys():
                    if insertion_idx1 >= insertion_idx2:
                        continue
                    # Example for cube_size=3:
                    # Initial solution: (f0.f1.f1).r1.(f0.f0.f1.f2.f2)

                    # Rotating 180 degrees using f-moves:
                    # f0.f0.f1.f1.f2.f2.(f0.f1.f1).r1.(f0.f0.f2.f2)
                    net_cost1 = 2 * cube_size

                    # f0.f0.f0 is replaced by -f0
                    # -f0.f1.f1.f2.f2.(f1.f1).r1.(f0.f0.f1.f2.f2)
                    net_cost1 -= 2 * group_ct_map1[insertion_idx1]

                    # f1.f1.f1.f1 is canceled out
                    # -f0.f2.f2.r1.(f0.f0.f1.f2.f2)
                    net_cost1 -= 4 * group_ct_map2[insertion_idx1]

                    # Undoing rotation
                    # -f0.f2.f2.r1.(f0.f0.f1.f2.f2).f0.f0.f1.f1.f2.f2
                    net_cost2 = 2 * cube_size

                    # f1.f1.f1 is replaced by -f1
                    # -f0.f2.f2.r1.(f0.f0.f2.f2).f0.f0.-f1.f2.f2
                    net_cost2 -= 2 * group_ct_map1[insertion_idx2]

                    # f0.f0.f0.f0 and f2.f2.f2.f2 are canceled out
                    # -f0.f2.f2.r1.-f1
                    net_cost2 -= 4 * group_ct_map2[insertion_idx2]

                    cost = net_cost1 + net_cost2

                    # Rotating 90 degrees using f-moves:
                    # f0.f1.f2.(f0.f1.f1).r1.(f0.f0.f1.f2.f2).-f0.-f1.-f2
                    cost90 = (
                        cube_size
                        - 2 * group_ct_map_neg1[insertion_idx1]
                        - 2 * group_ct_map2[insertion_idx1]
                    )
                    cost90 += (
                        cube_size
                        - 2 * group_ct_map1[insertion_idx2]
                        - 2 * group_ct_map2[insertion_idx2]
                    )

                    if cost90 < 0:
                        insertion_cost.append(
                            (insertion_idx1, insertion_idx2, 1, cost90)
                        )
                        # print(insertion_cost)
                        # exit()

                    # Rotating -90 degrees using f-moves:
                    # -f0.-f1.-f2.(f0.f1.f1).r1.(f0.f0.f1.f2.f2).f0.f1.f2
                    cost90 = (
                        cube_size
                        - 2 * group_ct_map1[insertion_idx1]
                        - 2 * group_ct_map2[insertion_idx1]
                    )
                    cost90 += (
                        cube_size
                        - 2 * group_ct_map_neg1[insertion_idx2]
                        - 2 * group_ct_map2[insertion_idx2]
                    )

                    if cost90 < 0:
                        insertion_cost.append(
                            (insertion_idx1, insertion_idx2, -1, cost90)
                        )

                    if cost > 0:
                        continue
                    # Attempts to push extra moves to the end of the sequence
                    if cost == 0 and net_cost2 <= 0:
                        continue
                    insertion_cost.append((insertion_idx1, insertion_idx2, 2, cost))
            if len(insertion_cost) == 0:
                break

            insertion_cost = sorted(insertion_cost, key=lambda x: x[3])

            if debug:
                print("Insertion cost:")
                print("insertion_idx1, insertion_idx2, direction, cost")
                for insertion_idx1, insertion_idx2, direction, cost in insertion_cost:
                    print(
                        f"{insertion_idx1:14d}, {insertion_idx2:14d}, {direction:9d}, {cost:4d}"
                    )

            insertion_idx1, insertion_idx2, direction, cost = insertion_cost[0]
            print(
                f"{direction}{group_letter}-rotating solution[{insertion_idx1}, {insertion_idx2}]",
                end="",
            )
            print(f", expected diff: {cost}, actual: {len(temp_solution)}", end="->")

            if debug:
                solved_puzzle = puzzle.clone().full_permutation(temp_solution)
                assert solved_puzzle.is_solved, f"Initially Unsolved:\n{solved_puzzle}"

            previous_length = len(temp_solution)
            temp_solution = rotate_cube_solution(
                temp_solution,
                cube_size,
                axis=group_letter,
                direction=direction,
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
            temp_solution = clean_cube_solution(puzzle, temp_solution)
            print(len(temp_solution), end="")
            actual_diff = len(temp_solution) - previous_length
            print(f" ({actual_diff})", end="")
            if cost != actual_diff:
                print("\u274c")
            else:
                print("\u2713")

            if len(temp_solution) <= len(solution):
                solution = temp_solution.copy()

            if debug:
                solved_puzzle = puzzle.clone().full_permutation(solution)
                assert (
                    solved_puzzle.is_solved
                ), f"Unsolved  rotated + cleaned-up cube:\n{solved_puzzle}"

    return solution
