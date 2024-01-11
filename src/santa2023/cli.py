import argparse

from santa2023.rotation import eliminate_cube_rotations

from .genetic import genetic
from .ida import ida
from .identities import ensemble, identities, shortcut, simple_wildcards, test
from .puzzle import read_puzzle_info, read_puzzles
from .utils import (
    CSV_BASE_PATH,
    PUZZLE_TYPES,
    export_solution,
    read_solution,
    sorted_solution,
)


def plot(args):
    solution = read_solution(args.solution_file)
    puzzles = read_puzzles(CSV_BASE_PATH / "puzzles.csv")
    puzzle_info = read_puzzle_info(CSV_BASE_PATH / "puzzle_info.csv")
    for p in puzzles:
        p.initialize_move_list(puzzle_info[p.type])

    puzzle_to_plot = puzzles[args.puzzle_id]
    permutations = solution[puzzle_to_plot._id]
    mismatch_series = [puzzle_to_plot.count_mismatches]
    for perm in permutations:
        puzzle_to_plot.permutate(perm)
        mismatch_series.append(puzzle_to_plot.count_mismatches)

    import matplotlib.pyplot as plt

    plt.scatter(range(len(mismatch_series)), mismatch_series)
    plt.plot(
        [0, len(mismatch_series)],
        [puzzle_to_plot._num_wildcards] * 2,
        color="red",
        linestyle="dashed",
    )
    plt.show()


def diff_solutions(sol1, sol2):
    diff_start = 0
    diff_end = -1
    if isinstance(sol1, str):
        sol1 = sol1.split(".")
    if isinstance(sol2, str):
        sol2 = sol2.split(".")

    while diff_start < len(sol2) and sol1[diff_start] == sol2[diff_start]:
        diff_start += 1

    if diff_start < len(sol2):
        while diff_end > 0 and sol1[diff_end] == sol2[diff_end]:
            diff_end -= 1

    diff_end += 1
    print(diff_start, diff_end)
    diff = ".".join(sol1[:diff_start]) + "."
    offset = len(diff)
    diff += "\n"
    diff += " " * offset
    diff += ".".join(sol1[diff_start:diff_end]) + "\n"
    diff += " " * offset
    if len(sol2[diff_start:diff_end]):
        diff += ".".join(sol2[diff_start:diff_end]) + "\n"
    else:
        diff += "<blank>\n"
    offset = len(".".join(sol1[:diff_end]) + ".")
    diff += " " * offset + "."
    diff += ".".join(sol1[diff_end:]) + "\n"
    diff += " " * offset + "."
    diff += ".".join(sol2[diff_end:])
    return diff


def evaluate(args):
    total = 0
    solution = read_solution(args.solution_file)
    puzzles = read_puzzles(CSV_BASE_PATH / "puzzles.csv")
    puzzle_info = read_puzzle_info(CSV_BASE_PATH / "puzzle_info.csv")
    for p in puzzles:
        p.initialize_move_list(puzzle_info[p.type])

    cases = None
    if args.puzzles:
        cases = args.puzzles[0]
    print("Evaluating " + args.solution_file)
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
        if puzzle.type.startswith("cube"):
            for i in range(len(solution[puzzle._id]) - 1):
                if solution[puzzle._id][i] == solution[puzzle._id][i + 1] and solution[
                    puzzle._id
                ][i].startswith("-"):
                    solution[puzzle._id][i] = solution[puzzle._id][i][1:]
                    solution[puzzle._id][i + 1] = solution[puzzle._id][i + 1][1:]

            if (
                puzzle.type.startswith("cube")
                and not args.fast
                and (cases is None or puzzle._id in cases)
            ):
                solution[puzzle._id] = eliminate_cube_rotations(
                    solution[puzzle._id], puzzle, debug=args.debug
                )

            solution[puzzle._id] = sorted_solution(puzzle, solution[puzzle._id])

        print(
            f"{puzzle._id}\t"
            f"{puzzle.type}\t"
            f"{len(puzzle._initial)}\t"
            f"{len(puzzle._allowed_moves)}\t"
            f"{puzzle._num_wildcards}\t"
            f"{puzzle.count_mismatches}\t"
            f"{len(solution[puzzle._id])}\t"
            f"{puzzle.count_mismatches/len(solution[puzzle._id])}\t",
            end="",
        )

        if args.fast and (
            puzzle.type == "cube_19/19/19" or puzzle.type == "cube_33/33/33"
        ):
            print("skipped validation")
        else:
            puzzle.full_permutation(solution[puzzle._id])
            assert puzzle.is_solved, f"Unsolved:\n{puzzle}"
            print(f"{puzzle.count_mismatches}")
        total += len(solution[puzzle._id])

    print(f"Solution value: {total}")

    export_solution(puzzles, solution)


def main():
    parser = argparse.ArgumentParser(description="Santa 2023 Solver and Utilities")
    parser.set_defaults(func=lambda x: parser.print_help())
    subparsers = parser.add_subparsers(
        title="Available commands",
    )

    evaluate_parser = subparsers.add_parser(
        "evaluate", aliases=["eval"], description="Evaluate a solution file."
    )
    evaluate_parser.add_argument("solution_file")
    evaluate_parser.add_argument(
        "-p", "--puzzles", type=int, action="append", nargs="*"
    )
    evaluate_parser.add_argument(
        "-f",
        "--fast",
        help="skip validation of cube_19/19/19 and cube_33/33/33 puzzles",
        action="store_true",
    )
    evaluate_parser.add_argument(
        "-d", "--debug", help="print debugging information", action="store_true"
    )
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

    identities_parser = subparsers.add_parser(
        "identities",
        description="Replace sequences of moves with shorter sequences based on a mappings->moves hash table.",
    )
    identities_parser.add_argument("initial_solution_file")
    identities_parser.add_argument(
        "--puzzle_type",
        help="puzzle types to solve",
        choices=PUZZLE_TYPES,
        default=PUZZLE_TYPES[0],
    )
    identities_parser.add_argument(
        "-d", "--depth", help="how many moves to enumerate", type=int, default=5
    )
    identities_parser.add_argument("-f", "--fast", action="store_true")
    identities_parser.add_argument("-t", "--max_time")
    identities_parser.set_defaults(func=identities)

    shortcut_parser = subparsers.add_parser(
        "shortcuts",
        aliases=["sc", "shortcut"],
        description="Find identical patterns in a sequence of moves.",
    )
    shortcut_parser.add_argument("initial_solution_file")
    shortcut_parser.set_defaults(func=shortcut)

    wildcards_parser = subparsers.add_parser(
        "wildcards",
        aliases=["wc", "wildcard"],
        description="Terminate sequences of moves earlier if a pattern has fewer mismatches than wildcards.",
    )
    wildcards_parser.add_argument("initial_solution_file")
    wildcards_parser.set_defaults(func=simple_wildcards)

    ensemble_parser = subparsers.add_parser(
        "ensemble",
        description="Ensemble multiple solutions into a single best solution.",
    )
    ensemble_parser.add_argument(
        "solution_files_or_folders",
        help="solution files or folders containing solution files (all csv files in a folder will be considered)",
        nargs="+",
    )
    ensemble_parser.set_defaults(func=ensemble)

    plot_parser = subparsers.add_parser("plot")
    plot_parser.add_argument("solution_file")
    plot_parser.add_argument("-p", "--puzzle_id", type=int, required=True)
    plot_parser.set_defaults(func=plot)

    ida_parser = subparsers.add_parser(
        "ida", description="[Experimental and Buggy] Run IDA* search on a puzzle."
    )
    ida_parser.add_argument("initial_solution_file")
    ida_parser.add_argument("-p", "--puzzle_id", type=int, default=0)
    ida_parser.add_argument("-c", "--max_cost", type=int, default=15)
    ida_parser.add_argument("-b", "--bound", type=int, default=30)
    ida_parser.set_defaults(func=ida)

    test_parser = subparsers.add_parser("test")
    test_parser.add_argument("initial_solution_file")
    test_parser.add_argument("-p", "--puzzle_id", type=int, required=True)
    test_parser.set_defaults(func=test)

    args = parser.parse_args()
    args.func(args)
