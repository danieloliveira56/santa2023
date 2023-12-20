import argparse

from genetic import genetic
from identities import identities
from puzzle import read_puzzle_info, read_puzzles
from utils import read_solution, PUZZLE_TYPES


def evaluate(args):
    total_score = 0
    solution = read_solution(args.solution_file)
    puzzles = read_puzzles("puzzles.csv")
    puzzle_info = read_puzzle_info("puzzle_info.csv")
    for p in puzzles:
        p.set_allowed_moves(puzzle_info[p.type])
    for permutation, puzzle in zip(solution, puzzles):
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
evaluate_parser.add_argument("initial_solution_file")
evaluate_parser.set_defaults(func=genetic)
genetic_parser.add_argument("-i", "--num_iterations", type=int, default=1000)
genetic_parser.add_argument("-n", "--size_population", type=int, default=100)
genetic_parser.add_argument("-s", "--survival_rate", type=int, default=0)
genetic_parser.add_argument("-c", "--num_crossovers", type=int, default=20)
genetic_parser.add_argument("-m", "--num_mutations", type=int, default=200)

identities_parser = subparsers.add_parser("identities")
identities_parser.add_argument("initial_solution_file")
identities_parser.add_argument('--puzzle_type', choices=PUZZLE_TYPES, default=PUZZLE_TYPES[0])
identities_parser.add_argument('-d', '--depth', type=int, default=5)
identities_parser.set_defaults(func=identities)

args = parser.parse_args()
args.func(args)
