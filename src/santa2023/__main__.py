import argparse

from .genetic import genetic
from .identities import identities, shortcut, simple_wildcards
from .puzzle import read_puzzle_info, read_puzzles
from .utils import read_solution, PUZZLE_TYPES


def evaluate(args):
    total_score = 0
    solution = read_solution(args.solution_file)
    puzzles = read_puzzles("puzzles.csv")
    puzzle_info = read_puzzle_info("puzzle_info.csv")
    for p in puzzles:
        p.initialize_move_list(puzzle_info[p.type])

    for i, (permutation, puzzle) in enumerate(zip(solution, puzzles)):
        print(i, puzzle._id)
        puzzle.full_permutation(permutation)
        assert puzzle.is_solved
        total_score += puzzle.score
        print(total_score)

    return total_score


parser = argparse.ArgumentParser(description="Santa 2023 Solver")
subparsers = parser.add_subparsers(
    title="subcommands",
)

evaluate_parser = subparsers.add_parser("evaluate")
evaluate_parser.add_argument("solution_file")
evaluate_parser.set_defaults(func=evaluate)

genetic_parser = subparsers.add_parser("genetic")
genetic_parser.add_argument("initial_solution_file")
genetic_parser.add_argument('-p', "--puzzles", type=int, action='append', nargs='*')
genetic_parser.add_argument("-i", "--num_iterations", type=int, default=1000)
genetic_parser.add_argument("-n", "--size_population", type=int, default=100)
genetic_parser.add_argument("-s", "--survival_rate", type=float, default=0.1)
genetic_parser.add_argument("-c", "--num_crossovers", type=int, default=50)
genetic_parser.add_argument("-m", "--num_mutations", type=int, default=50)
genetic_parser.set_defaults(func=genetic)

identities_parser = subparsers.add_parser("identities")
identities_parser.add_argument("initial_solution_file")
identities_parser.add_argument(
    "--puzzle_type", choices=PUZZLE_TYPES, default=PUZZLE_TYPES[0]
)
identities_parser.add_argument("-d", "--depth", type=int, default=5)
identities_parser.set_defaults(func=identities)

shortcut_parser = subparsers.add_parser("shortcut")
shortcut_parser.add_argument("initial_solution_file")
shortcut_parser.set_defaults(func=shortcut)

wildcards_parser = subparsers.add_parser("wildcards")
wildcards_parser.add_argument("initial_solution_file")
wildcards_parser.set_defaults(func=simple_wildcards)

args = parser.parse_args()
args.func(args)
