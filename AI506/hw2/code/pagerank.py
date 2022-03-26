from collections import defaultdict
from typing import DefaultDict, Dict, List, TextIO
from graph import Graph_disk, Graph_memory

### You may import any Python's standard library here (Do not import other external libraries) ###

### Import End ###


# Compute the distance between two dictionaries based on L1 norm
def l1_distance(x: DefaultDict[int, float], y: DefaultDict[int, float]) -> float:
    err: float = 0.0
    for k in x.keys():
        err += abs(x[k] - y[k])
    return err


################################################################################
# Run the pagerank algorithm iteratively using the memory-based graph          #
#  parameters                                                                  #
#    - graph : Memory-based graph (Graph_memory object)                        #
#    - damping_factor : Damping factor                                         #
#    - preference : Preference vector                                          #
#    - maxiters : The maximum number of iterations                             #
#    - tol : Tolerance threshold to check the convergence                      #
################################################################################
def pagerank_memory(
    graph: Graph_memory,
    damping_factor: float,
    preference: Dict[int, float],
    maxiters: int,
    tol: float,
) -> Dict[int, float]:
    vec: DefaultDict[int, float] = defaultdict(float)  # Pagerank vector

    ############### TODO: Implement the pagerank algorithm #####################
    
    vec_old = defaultdict()
    N = len(graph._nodes)
    for node in graph._nodes:
        vec_old[node] = 1/N

    for itr in range(maxiters):
        vec = defaultdict(lambda: 0.)
        for node in graph._nodes:
            if node not in graph.out_neighbor.keys():
                continue
            for out_neighbor in graph.out_neighbor[node]:
                vec[out_neighbor] += damping_factor * vec_old[node] / graph._out_degree[node]       
        
        S = sum(vec.values())
        for node in graph._nodes:
            vec[node] += (1-S) * preference[node]

        #### Check the convergence ###
        # Stop the iteration if L1norm[PR(t) - PR(t-1)] < tol
        delta: float = 0.0
        delta = l1_distance(vec, vec_old)
        print(f"[Iter {itr}]\tDelta = {delta}")

        if delta < tol:
            break
        
        vec_old = vec

    ########################### Implementation end #############################

    return dict(vec)


################################################################################
# Run the pagerank algorithm iteratively using the disk-based graph            #
#  parameters                                                                  #
#    - graph : Disk-based graph (Graph_disk) object                            #
#    - damping_factor : Damping factor                                         #
#    - preference : Preference vector                                          #
#    - maxiters : The maximum number of iterations                             #
#    - tol : Tolerance threshold to check the convergence                      #
################################################################################
def pagerank_disk(
    graph: Graph_disk,
    damping_factor: float,
    preference: Dict[int, float],
    maxiters: int,
    tol: float,
) -> Dict[int, float]:
    vec: DefaultDict[int, float] = defaultdict(float)  # Pagerank vector

    ############### TODO: Implement the pagerank algorithm #####################

    vec_old = defaultdict()
    N = len(graph._nodes)
    for node in graph._nodes:
        vec_old[node] = 1/N

    for itr in range(maxiters):
        vec = defaultdict(lambda: 0.)
        graph.setBOF()
        while True:
            edge = graph.readEdge()
            if edge is None:
                break
            src, dst = edge[0], edge[1]
            vec[dst] += damping_factor * vec_old[src] / graph._out_degree[src]
         
        S = sum(vec.values())
        for node in graph._nodes:
            vec[node] += (1-S) * preference[node]

        #### Check the convergence ###
        # Stop the iteration if L1norm[PR(t) - PR(t-1)] < tol
        delta: float = 0.0
        delta = l1_distance(vec, vec_old)
        print(f"[Iter {itr}]\tDelta = {delta}")

        if delta < tol:
            break

        vec_old = vec

    ########################### Implementation end #############################

    return dict(vec)
