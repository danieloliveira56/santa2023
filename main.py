import os
import itertools
import json
import random
def get_inverse(permutation):
    return [p[0] for p in sorted([(i, v) for i, v in enumerate(permutation)], key=lambda x: x[1])]
class Puzzle():
    def __init__(self, id, puzzle_type, solution, initial, num_wildcards):
        self.id = id
        self.type = puzzle_type
        self.initial = initial.split(";")
        self.current = initial.split(";")
        self.solution = solution.split(";")
        self.num_wildcards = int(num_wildcards)
        self.permutations = []

    def set_allowed_moves(self, allowed_moves):
        self.allowed_moves = allowed_moves

    @property
    def allowed_move_ids(self):
        return list(self.allowed_moves.keys())

    def random_solution(self, size):
        permutations = random.choices(self.allowed_move_ids, k=size)
        return [ '-'*random.randrange(2) + p for p in permutations ]

    def permutate(self, move_id, inverse=False):
        permutation = self.allowed_moves[move_id]
        if inverse:
            permutation = get_inverse(permutation)
            self.permutations.append(f"-{move_id}")
        else:
            self.permutations.append(move_id)

        self.current = [
            self.current[i] for i in permutation
        ]
        return self

    def full_permutation(self, permutation_list):
        for move_id in permutation_list:
            self.permutate(move_id.strip("-"), move_id.startswith("-"))
        return self

    def clone(self):
        cloned_puzzle = Puzzle(self.id, self.type, "", "", self.num_wildcards)
        cloned_puzzle.solution = self.solution.copy()
        cloned_puzzle.current = self.current.copy()
        cloned_puzzle.initial = self.initial.copy()
        cloned_puzzle.allowed_moves = self.allowed_moves.copy()
        return cloned_puzzle

    @property
    def score(self):
        return 5 * sum([c1 != c2 for c1, c2 in zip(self.current, self.solution)]) + len(self) + (0 if self.is_solved else 2)

    @property
    def is_solved(self):
        return self.current == self.solution

    @property
    def submission(self):
        return f"{self.id},{'.'.join(self.permutations)}"

    def __len__(self):
        return len(self.permutations)

    def __getitem__(self, item):
        return self.permutations[item]

    def __repr__(self):
        return(
            "----------\n"
            f"{self.id}: "
            f"{self.type} "
            f"[{self.num_wildcards}]\n"
            f"{''.join(self.initial)}\n"
            f"{''.join(self.current)}[{self.score}]\n"
            f"{''.join(self.solution)}\n"
            f"{self.submission}\n"
            "----------"
           )

def get_puzzles(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    return [
        Puzzle(*l.strip().split(","))
        for l in lines[1:]
    ]

def get_puzzle_info(filename):
    global l
    with open(filename, 'r') as f:
        lines = f.readlines()
    type_moves = [l.strip().split(",", maxsplit=1) for l in lines[1:]]
    return {
        type: json.loads(moves.strip("\"").replace("'", "\""))
        for type, moves in type_moves
    }

def crossover(permutations1, permutations2):
    i = random.randrange(len(permutations1))
    j = random.randrange(len(permutations2))
    return permutations1[:i] + permutations2[j:]

def mutate(permutation, allowed_move_ids):
    mutated = False
    while not mutated:
        if len(permutation) > 1 and random.randrange(0,2):
            i = random.randrange(len(permutation))
            del permutation[i]
            mutated = True

        if not mutated and random.randrange(0,2):
            i = random.randrange(len(permutation)+1)
            new_move = '-'*random.randrange(2) + random.choice(allowed_move_ids)
            permutation.insert(i,  new_move)
            mutated = True

        if not mutated and random.randrange(0,2):
            i = random.randrange(len(permutation))
            new_move = '-'*random.randrange(2) + random.choice(allowed_move_ids)
            while new_move == permutation[i]:
                new_move = '-' * random.randrange(2) + random.choice(allowed_move_ids)
            permutation[i] = new_move
            mutated = True
    return permutation

def remove_identity(permutation):
    for i in range(len(permutation) - 1, 0):
        if permutation[i] == f"-{permutation[i + 1]}" or permutation[i + 1] == f"-{permutation[i]}":
            permutation.pop(i)
            permutation.pop(i+1)
    return permutation

def get_identities(puzzle_info):
    move_ids = list(puzzle_info.keys())
    for id in move_ids:
        puzzle_info[f"-{id}"] = get_inverse(puzzle_info[id])
    move_ids = list(puzzle_info.keys())
    puzzle_size = len(list(puzzle_info.values())[0])
    permutation_map = {
        tuple(range(puzzle_size)): [()]
    }
    for n in range(1,4):
        for permutations in itertools.permutations(move_ids, r=n):
            result = puzzle_info[permutations[0]]
            for move_id in permutations[1:]:
                result = [result[i] for i in puzzle_info[move_id]]
            id = tuple(result)
            print(id)
            if id in permutation_map:
                permutation_map[id].append(permutations)
            else:
                permutation_map[id] = [permutations]
    return permutation_map

if __name__ == '__main__':
    num_iterations = 1000
    size_population = 100
    lucky_survivors = 20
    num_crossovers = 20
    num_mutations = 50

    puzzle_info = get_puzzle_info("puzzle_info.csv")
    puzzles = get_puzzles("puzzles.csv")
    for p in puzzles:
        p.set_allowed_moves(puzzle_info[p.type])

    # permutation_map = get_identities(puzzle_info['cube_2/2/2'])
    # for k, v in permutation_map.items():
    #     print(k, v)
    # exit()

    for puzzle_idx, p in enumerate(puzzles):
        pool = [p.clone() for _ in range(size_population)]
        pool_solutions = [p.random_solution(20) for _ in range(size_population)]
        for puzzle, permutation in zip(pool, pool_solutions):
            puzzle.full_permutation(remove_identity(permutation))

        for i in range(num_iterations):
            for j in range(num_crossovers):
                new_p = crossover(*random.sample(pool,2))
                remove_identity(new_p)
                pool.append(p.clone().full_permutation(new_p))
                print(p.clone())
            for j in range(num_mutations):
                new_p = mutate(random.choice(pool).permutations, p.allowed_move_ids)
                remove_identity(new_p)
                pool.append(p.clone().full_permutation(new_p))

            pool = sorted(pool, key=lambda x: x.score)
            pool = pool[:(size_population-lucky_survivors)] + random.sample(pool[(size_population-lucky_survivors):], k=lucky_survivors)
            # print(f"Searching {puzzle_idx}/{len(puzzles)}, End of iteration {i+1}/{num_iterations}, Pool size: {len(pool)} Current score: {pool[0].score}")

        if pool[0].is_solved:
            print(pool[0])
            print(f"***{pool[0].submission}")
        else:
            print("No solution found")






