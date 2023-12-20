from random import choice, randrange


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
