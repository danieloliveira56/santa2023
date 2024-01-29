import time

from santa2023.data import PUZZLE_INFO, PUZZLES
from santa2023.pricer import SolutionPricer
from santa2023.puzzle import Permutation


def search(puzzle, allowed_moves, pricer, bound):
    print(f"Trying bound {bound}")
    start = time.time()
    queue = [Permutation(range(puzzle.size()))]
    while True:
        if len(queue) == 0:
            return None
        p1 = queue.pop()
        result_str = "".join([puzzle._initial[i] for i in p1.mapping])
        if result_str == pricer._goal_state:
            return p1
        p1_estimate = pricer.estimate(result_str)
        print(
            f"Bound: {bound}, Queue size: {len(queue):0,}, Cost: {len(p1)}/{p1_estimate}, Time: {time.time() - start:.1f}, curr_solution: {result_str}",
            end="\r",
        )
        if len(p1) + p1_estimate > bound:
            continue
        for move_id, p2 in allowed_moves.items():
            if len(p1) == 0 or move_id not in pricer._taboo_list[p1.move_ids[-1]]:
                queue.append(p1 * p2)


def ida(args):
    print("Running IDA*")

    puzzle = PUZZLES[args.puzzle_id]
    allowed_moves = {
        key: Permutation(value, [key])
        for key, value in PUZZLE_INFO[puzzle.type].items()
    }

    pricer = SolutionPricer(
        puzzle._solution,
        allowed_moves=allowed_moves,
        taboo_list=puzzle.taboo_list,
        max_cost=args.max_cost,
    )

    solution = search(puzzle, allowed_moves, pricer, args.bound)

    if solution is None:
        print("No solution found")
        return

    print(f"Solution found: {solution}")
