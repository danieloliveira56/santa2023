from random import choice, randrange, sample

from puzzle import read_puzzle_info, read_puzzles
from utils import read_solution, remove_identity


def crossover(permutations1, permutations2):
    i = randrange(len(permutations1[0]))
    j = randrange(len(permutations2[0]))
    return permutations1[0][:i] + permutations2[0][j:]


def mutate(permutation, allowed_move_ids):
    mutated = False
    while not mutated:
        if len(permutation) > 1 and randrange(0, 2):
            i = randrange(len(permutation))
            del permutation[i]
            mutated = True
            mutation_type = "delete-mutation"

        if not mutated and randrange(0, 2):
            i = randrange(len(permutation) + 1)
            new_move = "-" * randrange(2) + choice(allowed_move_ids)
            permutation.insert(i, new_move)
            mutated = True
            mutation_type = "insert-mutation"

        if not mutated and randrange(0, 2):
            i = randrange(len(permutation))
            new_move = "-" * randrange(2) + choice(allowed_move_ids)
            while new_move == permutation[i]:
                new_move = "-" * randrange(2) + choice(allowed_move_ids)
            permutation[i] = new_move
            mutated = True
            mutation_type = "replace-mutation"
    return permutation, mutation_type


def genetic(args):
    initial_solution = read_solution(filename=args.solution)
    puzzles = read_puzzles("puzzles.csv")
    puzzle_info = read_puzzle_info("puzzle_info.csv")
    for p in puzzles:
        p.set_allowed_moves(puzzle_info[p.type])

    num_iterations = args.num_iterations
    size_population = args.size_population
    num_survivors = size_population * args.survival_rate
    num_crossovers = args.num_crossovers
    num_mutations = args.num_mutations

    solution_score = {
        "original": 0,
        "crossover": 0,
        "replace-mutation": 0,
        "delete-mutation": 0,
        "insert-mutation": 0,
    }

    for puzzle_idx, p in enumerate(puzzles):
        initial_permutation = initial_solution[puzzle_idx]
        current_score = len(initial_permutation)
        initial_permutations = [
            p.random_solution(len(initial_permutation)) for _ in range(size_population)
        ]
        initial_permutations.append(initial_permutation)

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
            pool = pool[: (size_population - num_survivors)] + sample(
                pool[(size_population - num_survivors) :], k=num_survivors
            )
            new_score = pool[0][0].score
            if new_score < current_score:
                solution_score[pool[0][1]] += current_score - new_score
                current_score = new_score

            print(
                f"Searching {puzzle_idx}/{len(puzzles)}, "
                f"End of iteration {i + 1}/{num_iterations}, "
                f"Pool size: {len(pool)} "
                f"Score: {len(initial_permutation)}->{new_score}"
            )
        if pool[0][0].is_solved:
            print(f"***{pool[0][0].submission}")
        else:
            print("No solution found")
        print(solution_score)
