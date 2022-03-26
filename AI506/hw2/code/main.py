import argparse
import heapq
from typing import Dict, Union
from graph import Graph_disk, Graph_memory
from pagerank import pagerank_disk, pagerank_memory
from preference import preference_uniform, preference_onehot, preference_degree


if __name__ == "__main__":
    ############################################################################
    # Each line of the file looks like:                                        #
    #         <SOURCE ID>\t<DESTINATION ID>                                    #
    # s.t. two integers are separated by tab                                   #
    ############################################################################
    fileName = "../data/web-Stanford.edgelist"

    beta = 0.85  # damping factor
    maxiters = 1000
    tolerance = 1e-6
    n = 10  # Print Top-n pageranks

    # Parse the arguments
    parser = argparse.ArgumentParser(description="Run the pagerank algorithm.")
    parser.add_argument(
        "-d", "--disk", action="store_true", help="Use disk based pagerank"
    )
    parser.add_argument(
        "-p",
        "--pref",
        action="store",
        default="uniform",
        type=str,
        help="Select the preference vector: uniform, onehot, degree",
    )
    parser.add_argument(
        "-t",
        "--target",
        action="store",
        default=1,
        type=int,
        help="Select the target node when you are using one-hot preference vector",
    )
    args = parser.parse_args()

    # Initialize the graph
    graph: Union[Graph_disk, Graph_memory]
    if args.disk:
        graph_disk: Graph_disk = Graph_disk(fileName)
        graph = graph_disk
    else:
        graph_memory: Graph_memory = Graph_memory(fileName)
        graph = graph_memory

    # Select the preference vector
    preference: Dict[int, float]
    if args.pref == "degree":
        preference = preference_degree(graph)
    elif args.pref == "onehot":
        preference = preference_onehot(graph, args.target)
    else:
        preference = preference_uniform(graph)

    # Run the pagerank algorithm
    vec: Dict[int, float]
    if args.disk:
        vec = pagerank_disk(graph_disk, beta, preference, maxiters, tolerance)
    else:
        vec = pagerank_memory(graph_memory, beta, preference, maxiters, tolerance)

    # Find top-n pageranks
    topN = [(k, vec[k]) for k in heapq.nlargest(n, vec, key=lambda k: vec[k])]

    print(f"-----Top {n} pageranks-----")
    for node, pagerank in topN:
        print(f"{node}\t{pagerank}")
