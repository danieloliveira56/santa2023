from random import choice, randrange, sample

from santa2023.puzzle import read_puzzle_info, read_puzzles
from santa2023.utils import export_solution, read_solution, remove_identity


def crossover(permutations1, permutations2):
    i = randrange(len(permutations1))
    j = randrange(len(permutations2))
    return permutations1[:i] + permutations2[j:]


def mutate(permutation, allowed_move_ids):
    mutated = False
    while not mutated:
        mutation_idx = randrange(4)
        if mutation_idx == 0:
            ij = (randrange(len(permutation) + 1), randrange(len(permutation) + 1))
            i = min(ij)
            j = max(ij)
            deletion_size = j - i
            if deletion_size < len(permutation):
                del permutation[i:j]
                mutated = True
                mutation_type = "delete-range-mutation"

        if mutation_idx == 1 and len(permutation) > 1:
            i = randrange(len(permutation))
            del permutation[i]
            mutated = True
            mutation_type = "delete-mutation"

        if mutation_idx == 2:
            i = randrange(len(permutation) + 1)
            new_move = choice(allowed_move_ids)
            permutation.insert(i, new_move)
            mutated = True
            mutation_type = "insert-mutation"

        if mutation_idx == 3:
            i = randrange(len(permutation))
            new_move = choice(allowed_move_ids)
            while new_move == permutation[i]:
                new_move = choice(allowed_move_ids)
            permutation[i] = new_move
            mutated = True
            mutation_type = "replace-mutation"
    return permutation, mutation_type


def genetic(args):
    initial_solution = read_solution(filename=args.initial_solution_file)
    puzzles = read_puzzles("puzzles.csv")
    puzzle_info = read_puzzle_info("puzzle_info.csv")
    for p in puzzles:
        p.initialize_move_list(puzzle_info[p.type])

    num_iterations = args.num_iterations
    size_population = args.size_population
    num_survivors = int(size_population * args.survival_rate)
    num_crossovers = args.num_crossovers
    num_mutations = args.num_mutations
    cases = None
    if args.puzzles:
        cases = args.puzzles[0]

    solution_score = {
        "original": 0,
        "crossover": 0,
        "replace-mutation": 0,
        "delete-mutation": 0,
        "delete-range-mutation": 0,
        "insert-mutation": 0,
    }

    new_solution = []
    try:
        for puzzle_idx, p in enumerate(puzzles):
            print(puzzle_idx, end="\r")
            initial_permutation = initial_solution[puzzle_idx]
            if cases and p._id not in cases:
                new_solution.append(initial_permutation)
                continue
            initial_permutations = [
                p.random_solution(len(initial_permutation))
                for _ in range(size_population)
            ]
            initial_permutations.append(initial_permutation)

            pool = [
                (p.clone().full_permutation(permutation))
                for permutation in initial_permutations
            ]

            for i in range(num_iterations):
                for j in range(num_crossovers):
                    new_p = crossover(*sample(pool, 2))
                    pool.append(
                        (
                            p.clone().full_permutation(
                                remove_identity(new_p)
                            )
                        )
                    )
                for j in range(num_mutations):
                    new_p, mutation_type = mutate(
                        choice(pool).permutations, p.allowed_move_ids
                    )
                    pool.append(
                        (
                            p.clone().full_permutation(
                                remove_identity(new_p)
                            )
                        )
                    )

                pool = sorted(pool, key=lambda x: x.score)
                pool = pool[: (size_population - num_survivors)] + sample(
                    pool[(size_population - num_survivors) :], k=num_survivors
                )

                print(
                    f"Searching {puzzle_idx}/{len(puzzles)} ({p._type}), "
                    f"End of iteration {i + 1}/{num_iterations}, "
                    f"Pool size: {len(pool)} "
                    f"Score: {len(initial_permutation)}->{pool[0].score}",
                    end="                                           \r",
                )
            print(f"{p._id}: {len(initial_permutation)}->{pool[0].score}", " " * 200)
            new_solution.append(pool[0].permutations)

    except KeyboardInterrupt:
        new_solution = new_solution + initial_solution[len(new_solution) :]
        pass
    except Exception as e:
        raise e

    print(solution_score)
    export_solution(puzzles, new_solution)
