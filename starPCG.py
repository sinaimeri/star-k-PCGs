from itertools import combinations
import networkx as nx
from docplex.cp.model import CpoModel


def generate_all_graphs(n):
    """Returns a list of all possible graphs with n nodes.
    The graphs are represented as a set of edges (i, j) with 1 <= i < j <= n."""
    node_list = [x for x in range(1, n + 1)]
    edge_list = list(combinations(node_list, 2))
    m = len(edge_list)
    graphs = []
    for num_edges in range(m + 1):
        for edges in combinations(edge_list, num_edges):
            G = nx.Graph()
            G.add_nodes_from(node_list)
            G.add_edges_from(edges)
            graphs.append(G)
    return graphs


def generate_non_isomorphic_graphs(n):
    """Returns a list of all non-isomorphic graphs with n nodes.
    The graphs are represented as a set of edges (i, j) with 1 <= i < j <= n."""
    graphs = generate_all_graphs(n)
    non_isomorphic_graphs = []
    for G in graphs:
        is_isomorphic = False
        for G1 in non_isomorphic_graphs:
            if nx.is_isomorphic(G, G1):
                is_isomorphic = True
                break
        if not is_isomorphic:
            non_isomorphic_graphs.append(G)
    return [list(x.edges()) for x in non_isomorphic_graphs]


def solve_single_interval(E, a):
    """Given a graph E (as a list of edges), this function checks if it can be recognized
    using a single interval with vertex weights from 1 to a."""
    mdl = CpoModel()

    # Calculate the number of nodes in the graph E
    n = max(max(edge) for edge in E)
    print(E)  # Print the edges of the graph

    # Create the complementary graph E1
    E1 = [(i, j) for i in range(1, n) for j in range(i + 1, n + 1) if (i, j) not in E]

    # Variables for interval bounds
    d_min = mdl.integer_var(1, name="d_min")
    d_max = mdl.integer_var(1, name="d_max")
    mdl.add_constraint(d_min <= d_max)

    # Create variables for vertex weights
    w = mdl.integer_var_dict((i for i in range(1, n + 1)), min=1, max=a)

    # Constraints for edges in E
    for i, j in E:
        mdl.add_constraint(mdl.logical_and(d_min <= w[i] + w[j], w[i] + w[j] <= d_max))

    # Constraints for edges in the complementary graph E1
    for i, j in E1:
        mdl.add_constraint(mdl.logical_or(w[i] + w[j] < d_min, w[i] + w[j] > d_max))

    # Solve the model
    msol = mdl.solve(log_output=None)

    if msol:
        a1 = msol[d_min]
        a2 = min(2 * a, msol[d_max])
        print(f'Interval 1: [{a1}, {a2}]')
        print('############')
        print('Vertex weights:')
        for i in range(1, n + 1):
            print(f'   weight v[{i}] =', msol[w[i]])
        print('############')
        print('Edge weights:')
        for i, j in E:
            print(f'   weight edge({i}, {j}) =', msol[w[i]] + msol[w[j]])
        print('############')
        return True
    else:
        print("No solution found")
        return False



def solve_two_intervals(E, a):
    """Given a graph E (as a list of edges), this function checks if it can be recognized
    using one or two intervals with vertex weights from 1 to a."""
    mdl = CpoModel()

    # Calculate the number of nodes in the graph E
    n = max(max(edge) for edge in E)
    print(E)  # Print the edges of the graph

    # Create the complementary graph E1
    E1 = [(i, j) for i in range(1, n) for j in range(i + 1, n + 1) if (i, j) not in E]

    # Variables for the two intervals
    d_min = mdl.integer_var(1, name="d_min")
    d_max = mdl.integer_var(1, name="d_max")
    d_min1 = mdl.integer_var(1, name="d_min1")
    d_max1 = mdl.integer_var(1, name="d_max1")

    # Constraints for the intervals
    mdl.add_constraint(d_min <= d_max)
    mdl.add_constraint(d_min1 <= d_max1)
    mdl.add_constraint(d_max + 1 < d_min1)

    # Create variables for vertex weights
    w = mdl.integer_var_dict((i for i in range(1, n + 1)), min=1, max=a)

    # Constraints for edges in E
    for i, j in E:
        mdl.add_constraint(
            mdl.logical_or(
                mdl.logical_and(d_min1 <= w[i] + w[j], w[i] + w[j] <= d_max1),
                mdl.logical_and(d_min <= w[i] + w[j], w[i] + w[j] <= d_max)
            )
        )

    # Constraints for edges in the complementary graph E1
    for i, j in E1:
        mdl.add_constraint(
            mdl.logical_or(
                mdl.logical_or(w[i] + w[j] < d_min, w[i] + w[j] > d_max1),
                mdl.logical_and(d_max < w[i] + w[j], w[i] + w[j] < d_min1 - 1)
            )
        )

    # Solve the model
    msol = mdl.solve(log_output=None)

    if msol:
        print(f'Interval 1: [{msol[d_min]}, {msol[d_max]}]')
        if msol[d_max] < msol[d_min1]:
            print(f'Interval 2: [{msol[d_min1]}, {msol[d_max1]}]')
        print('############')
        print('Vertex weights:')
        for i in range(1, n + 1):
            print(f'   weight v[{i}] =', msol[w[i]])
        print('############')
        print('Edge weights:')
        for i, j in E:
            print(f'   weight edge({i}, {j}) =', msol[w[i]] + msol[w[j]])
        print('############')
        return True
    else:
        print("No solution found")
        return False



def test_case_single_interval():
    # Test case for single interval graph recognition
    graph_edges = [(1, 2), (2, 3), (3, 4), (4, 5), (1, 3), (2,5)]
    max_weight = 8
    print("Testing single interval graph recognition:")
    result = solve_single_interval(graph_edges, max_weight)
    if result:
        print("Solution found!")
    else:
        print("No solution found.")

def test_case_two_intervals():
    # Test case for two intervals graph recognition 
    graph_edges = [(1, 2), (2, 3), (3, 4), (4, 5), (1,5)]
    max_weight = 8
    print("Testing star-2-graph recognition:")
    result = solve_two_intervals(graph_edges, max_weight)
    if result:
        print("Solution found!")
    else:
        print("No solution found.")





# Running test cases
test_case_single_interval()
test_case_two_intervals()
