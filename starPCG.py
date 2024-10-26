from itertools import combinations
from itertools import permutations
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


def check_2_FP(vertices, edges):
    """
    Check if there exist 5 pairs of vertices (a_1, a_2), (a_3, a_4), (a_5, a_6), 
    (a_7, a_8), (a_9, a_10) such that:
    - (a_1, a_2), (a_5, a_6), (a_9, a_10) are in the edge list.
    - (a_3, a_4), (a_7, a_8) are not in the edge list.
    - The pairs are distinct.
    - w(a_1) + w(a_2) < w(a_3) + w(a_4) < w(a_5) + w(a_6) < w(a_7) + w(a_8) < w(a_9) + w(a_10).
    
    Parameters:
    - vertices: A list of vertices ordered by their weight (i.e., w(v_1) < w(v_2) < ... < w(v_n)).
    - edges: A set of edges represented as tuples of vertex pairs (v_j, v_k) where j < k.
    
    Returns:
    - True if the condition holds for any 5 such pairs, False otherwise.
    """
    
    from itertools import combinations
    
    # Generate all possible distinct pairs of vertices
    all_pairs = list(combinations(vertices, 2))
    
    # Iterate over all possible sets of 5 distinct pairs
    for (a1, a2), (a3, a4), (a5, a6), (a7, a8), (a9, a10) in combinations(all_pairs, 5):
        # Check if the edges (a1, a2), (a5, a6), (a9, a10) exist in the edge list
        if ((a1, a2) in edges or (a2, a1) in edges) and \
           ((a5, a6) in edges or (a6, a5) in edges) and \
           ((a9, a10) in edges or (a10, a9) in edges):
           
            # Check if the pairs (a3, a4) and (a7, a8) are NOT in the edge list
            if ((a3, a4) not in edges and (a4, a3) not in edges) and \
               ((a7, a8) not in edges and (a8, a7) not in edges):
                
                # Now we check the total order using vertex indices (since vertices is ordered by weight)
                index_a1_a2 = (vertices.index(a1), vertices.index(a2))
                index_a3_a4 = (vertices.index(a3), vertices.index(a4))
                index_a5_a6 = (vertices.index(a5), vertices.index(a6))
                index_a7_a8 = (vertices.index(a7), vertices.index(a8))
                index_a9_a10 = (vertices.index(a9), vertices.index(a10))
                
                # Sort the pairs to ensure we are comparing the lower weights first
                index_a1_a2 = sorted(index_a1_a2)
                index_a3_a4 = sorted(index_a3_a4)
                index_a5_a6 = sorted(index_a5_a6)
                index_a7_a8 = sorted(index_a7_a8)
                index_a9_a10 = sorted(index_a9_a10)
                
                # Compare the pairs in order, based on their indices
                if index_a1_a2 < index_a3_a4 < index_a5_a6 < index_a7_a8 < index_a9_a10:
                    print(f"Found with indices: {index_a1_a2} < {index_a3_a4} < {index_a5_a6} < {index_a7_a8} < {index_a9_a10}")

                    return True
    
    return False


def check_2FP_for_all_total_orders(vertices,edges):
    # Define vertices (which will be permuted to represent different total orders by weight)
    #vertices are represented as a list in V
    #Edges are reppresented as pairs (i,j) with i<j
    
    # Generate all possible total orders of the vertices (permutations)
    total_orders = permutations(vertices)
    #if for ALL total orders we always found a 2FP then we are sure the graph is not a star-2-PCG
    star_2_PCG=False

    # Check each total order
    for order in total_orders:
        print(f"Testing order: {order}")
        result = check_2_FP(order, edges)
        if result:
            print(f"Condition satisfied with order: {order}")
        else:
            print(f"This can be a candidate order to prove the star-2-PCG: {order}")
            star_2_PCG=True
            break #STOP further testing 
    if not star_2_PCG:
        print("The graph is NOT a star-2-PCG")

def test_case_check_2FP():
     # Test case for the graph G536
     graph_edges=[(1, 2), (2, 3), (3, 4), (4, 5), (1, 5), (1,6), (2,6), (3,6), (4,6), (5,6), (6,7)]
     graph_vertices=[1,2,3,4,5,6,7]
     check_2FP_for_all_total_orders(graph_vertices,graph_edges)


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
#test_case_single_interval()
#test_case_two_intervals()
test_case_check_2FP()
