import argparse

from .genetic import genetic
from .identities import fast_identities, identities, shortcut, simple_wildcards, test, ensemble
from .puzzle import read_puzzle_info, read_puzzles
from .utils import PUZZLE_TYPES, read_solution, export_solution


def plot(args):
    solution = read_solution(args.solution_file)
    puzzles = read_puzzles("puzzles.csv")
    puzzle_info = read_puzzle_info("puzzle_info.csv")
    for p in puzzles:
        p.initialize_move_list(puzzle_info[p.type])

    puzzle_to_plot = puzzles[args.puzzle_id]
    permutations = solution[puzzle_to_plot._id]
    mismatch_series = [puzzle_to_plot.count_mismatches]
    for perm in permutations:
        puzzle_to_plot.permutate(perm)
        mismatch_series.append(puzzle_to_plot.count_mismatches)

    import matplotlib.pyplot as plt
    plt.plot(mismatch_series)
    plt.show()




def evaluate(args):
    total = 0
    solution = read_solution(args.solution_file)
    puzzles = read_puzzles("puzzles.csv")
    puzzle_info = read_puzzle_info("puzzle_info.csv")
    for p in puzzles:
        p.initialize_move_list(puzzle_info[p.type])

    print(
        "Puzzle\t"
        "Type\t"
        "num_characters\t"
        "num_moves\t"
        "num_wildcards\t"
        "initial_mismatches\t"
        "sol_length\t"
        "initial_mismatches/sol_length\t"
        "final_mismatches"
    )
    for puzzle in puzzles:
        if puzzle.type.startswith("globe"):
            solution[puzzle._id] = [p.replace("-f", "f") for p in solution[puzzle._id]]
        elif puzzle.type.startswith("cube"):
            current_group = solution[puzzle._id][:1]
            sorted_solution = []
            for move in solution[puzzle._id][1:]:
                if move.replace("-", "")[0] == current_group[-1].replace("-", "")[0]:
                    current_group.append(move)
                else:
                    sorted_solution += sorted(current_group, key=lambda x: x.replace("-", ""))
                    current_group = [move]
            sorted_solution += sorted(current_group, key=lambda x: -0.1 if x.startswith("-") else 0.1 + int(x.replace("-", "")[1:]))
            solution[puzzle._id] = sorted_solution

        print(
            f"{puzzle._id}\t"
            f"{puzzle.type}\t"
            f"{len(puzzle._initial)}\t"
            f"{len(puzzle._allowed_moves)}\t"
            f"{puzzle._num_wildcards}\t"
            f"{puzzle.count_mismatches}\t"
            f"{len(solution[puzzle._id])}\t"
            f"{puzzle.count_mismatches/len(solution[puzzle._id])}\t",
            end=""
        )

        if args.fast and (puzzle.type == "cube_19/19/19" or puzzle.type == "cube_33/33/33"):
            print(f"skipped validation")
        else:
            puzzle.full_permutation(solution[puzzle._id])
            assert puzzle.is_solved, f"Unsolved:\n{puzzle}"
            print(f"{puzzle.count_mismatches}")
        total += len(solution[puzzle._id])

    print(f"Solution value: {total}")

    export_solution(puzzles, solution)


parser = argparse.ArgumentParser(description="Santa 2023 Solver")
subparsers = parser.add_subparsers(
    title="subcommands",
)

evaluate_parser = subparsers.add_parser("evaluate")
evaluate_parser.add_argument("solution_file")
evaluate_parser.add_argument("-f", "--fast", action="store_true")
evaluate_parser.set_defaults(func=evaluate)

genetic_parser = subparsers.add_parser("genetic")
genetic_parser.add_argument("initial_solution_file")
genetic_parser.add_argument("-p", "--puzzles", type=int, action="append", nargs="*")
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
identities_parser.add_argument("-f", "--fast", action="store_true")
identities_parser.add_argument("-t", "--max_time")
identities_parser.set_defaults(func=identities)

shortcut_parser = subparsers.add_parser("shortcut")
shortcut_parser.add_argument("initial_solution_file")
shortcut_parser.set_defaults(func=shortcut)

wildcards_parser = subparsers.add_parser("wildcards")
wildcards_parser.add_argument("initial_solution_file")
wildcards_parser.set_defaults(func=simple_wildcards)

test_parser = subparsers.add_parser("test")
test_parser.add_argument("initial_solution_file")
test_parser.set_defaults(func=test)

ensemble_parser = subparsers.add_parser("ensemble")
ensemble_parser.add_argument("solution_files", nargs="+")
ensemble_parser.set_defaults(func=ensemble)

plot_parser = subparsers.add_parser("plot")
plot_parser.add_argument("solution_file")
plot_parser.add_argument("-p", "--puzzle_id", type=int, default=0)
plot_parser.set_defaults(func=plot)

args = parser.parse_args()
args.func(args)
