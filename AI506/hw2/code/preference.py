from typing import Dict
from graph import Graph

### You may import any Python's standard library here (Do not import other external libraries) ###

### Import End ###


def preference_uniform(graph: Graph) -> Dict[int, float]:
    ################## TODO: Fill out the preference vector ####################
    #  Prob(node i is choosen) should be the constant                          #
    ############################################################################

    preference: Dict[int, float] = dict()
    nodes = list(graph._nodes)
    num_nodes = len(nodes)
    for node in nodes:
        preference[node] = 1 / num_nodes

    ########################### Implementation end #############################

    return preference


def preference_onehot(graph: Graph, node: int) -> Dict[int, float]:
    assert node in graph.nodes

    ################## TODO: Fill out the preference vector ####################
    #  Prob(node i is choosen) = | 1   if i is given as a parameter            #
    #                            | 0   otherwise                               #
    ############################################################################

    preference: Dict[int, float] = dict()
    nodes = list(graph._nodes)
    for _node in nodes:
        if not _node == node:
            preference[_node] = 0
        else:
            preference[_node] = 1

    ########################### Implementation end #############################

    return preference


def preference_degree(graph: Graph) -> Dict[int, float]:
    ################## TODO: Fill out the preference vector ####################
    #  Prob(node i is choosen) ∝ 1 + in-degree(i)                             #
    ############################################################################

    preference: Dict[int, float] = dict()
    unnormalized_preference = dict()
    nodes = list(graph._nodes)
    num_nodes = len(nodes)
    num_in_degrees = sum(graph._in_degree.values())
    total_sum = num_nodes + num_in_degrees

    for node in nodes:
        if node in graph._in_degree.keys():
            in_degree = graph.in_degree[node]
        else:
            in_degree = 0

        preference[node] = (1 + in_degree) / total_sum

    ########################### Implementation end #############################

    return preference
