import networkx as nx
import sympy.combinatorics

from .utils import get_inverse


class SolutionPricer:
    def __init__(self, goal_state, allowed_moves, taboo_list, max_cost):
        self._goal_state = goal_state
        self._allowed_moves = allowed_moves
        self._taboo_list = taboo_list
        self._max_cost = max_cost

        self.compute_groups()

    @staticmethod
    def get_cost_database(solution, allowed_moves, taboo_list, max_cost):
        solution = "".join(solution)
        print(f"Building cost database for {solution} of length {len(solution)}")
        print(f"allowed_moves: {allowed_moves}")
        print(f"taboo_list: {taboo_list}")
        print(f"max_cost: {max_cost}")
        print()
        cost = {}
        max_mapped_cost = 0
        cost[solution] = 0
        queue = [(solution, ("", ""))]
        while len(queue) > 0 and max_mapped_cost < max_cost:
            print(
                f"Queue size: {len(queue):0,}, Max mapped cost {max_mapped_cost}/{max_cost}",
                end="\r",
            )
            current, last_move = queue.pop(0)
            for move_id in allowed_moves:
                if move_id in (
                    taboo_list.get(last_move, []) + taboo_list.get(last_move[-1], [])
                ):
                    continue
                new = "".join([current[i] for i in allowed_moves[move_id]])
                if new not in cost:
                    cost[new] = cost[current] + 1
                    max_mapped_cost = max(max_mapped_cost, cost[new])
                    queue.append((new, (last_move[-1], move_id)))
        # print(cost)
        return cost

    def compute_groups(self):
        position_graph = nx.DiGraph()
        position_graph.add_nodes_from(range(self.size()))

        for key, value in self._allowed_moves.items():
            p = sympy.combinatorics.Permutation(value.mapping)
            print(f"{key}:", p)
            # print("\t", sympy.combinatorics.Permutation(value).array_form)
            for group in p.cyclic_form:
                if len(group) > 1:
                    for i in range(len(group)):
                        position_graph.add_edge(group[i], group[(i + 1) % len(group)])

        connected_components = list(nx.weakly_connected_components(position_graph))
        self._partitions = []
        self._partition_moves = {}
        self._partition_reindex = {}
        self._pattern_database = {}
        for i, component in enumerate(connected_components):
            print(f"Component {i}: {len(component)}")
            print(component)
            self._partitions.append(component)

        for i, partition in enumerate(self._partitions):
            self._partition_reindex[i] = {k: j for j, k in enumerate(partition)}
            print(self._partition_reindex[i])
            for move_id, permutation in self._allowed_moves.items():
                print(f"Move {move_id}: {permutation}")
                for j in partition:
                    print(j, end="->")
                    print(permutation.mapping[j], end="->")
                    print(self._partition_reindex[i][permutation.mapping[j]])
                self._partition_moves[i][move_id] = [
                    self._partition_reindex[i][permutation.mapping[j]]
                    for j in partition
                ]
                self._partition_moves[i][f"-{move_id}"] = get_inverse(
                    self._partition_moves[i][move_id]
                )
            self._pattern_database[i] = self.get_cost_database(
                "".join([self._goal_state[j] for j in partition]),
                self._partition_moves[i],
                self._taboo_list,
                self._max_cost,
            )
            print()

    def size(self):
        return len(self._goal_state)

    def estimate(self, solution):
        total_cost = 0
        for i, group in enumerate(self._partitions):
            # print(f"Group {i}: {group}")
            total_cost += self._pattern_database[i].get(
                "".join([solution[j] for j in group]), self._max_cost
            )
            # print(f"Group {i}: {total_cost}")
        return total_cost

    @property
    def max_cost(self):
        return self._max_cost
