# The star-number of graphs with at most 7 vertices


This page provides the experimental results concerning the paper "On star-k-PCGs: Exploring class boundaries for small k values" by A. Monti and B. Sinaimeri [https://arxiv.org/abs/2209.11860]. 


### Graph Interval Recognition Solver

We provide the funcionts to determine whether a graph can be recognized using one or two intervals by assigning weights to its vertices and ensuring that the sum of weights for each edge falls within certain the interval(s).

### Methods

1. **Graph Generation:**
   - `generate_all_graphs(n)`: Generates all possible graphs with `n` nodes.
   - `generate_non_isomorphic_graphs(n)`: Generates all non-isomorphic graphs with `n` nodes.

2. **Single Interval Solver:**
   - `solve_single_interval(E, a)`: Determines if a graph `E` (represented by a list of edges) can be recognized using one interval with vertex weights between 1 and `a`.

3. **Two Interval Solver:**
   - `solve_two_intervals(E, a)`: Determines if a graph `E` can be recognized using one or two intervals with vertex weights between 1 and `a`.
  
4. **2-FP Checker:**
   - `check_2_FP(vertices, edges)`: Checks if there exist 5 distinct pairs of vertices that form a 2FP.

## Prerequisites

The following Python libraries are required to run the code:
- `networkx`: For graph generation and manipulation.
- `docplex`: For solving the optimization problems related to interval recognition.

You can install the required libraries using pip:

```bash
pip install networkx
pip install docplex
