import networkx as nx

#### Task1. Load mysterious_graph.txt using the networkx library ####
g = None # The variable "g" is an undirected graph object in networkx, and it contains mysterious_graph.txt.

###        TODO: Implement       ###
g = nx.read_edgelist('mysterious_graph.txt', nodetype=int)
###     Implementation end       ###

print("------------      Task1      ------------")
print("A number of nodes: %d"%g.number_of_nodes())
print("A number of edges: %d"%g.number_of_edges())
print("-----------------------------------------")
print()
#####################################################################


#######     Task2. Find the maximum degree node in the graph    #######
max_deg_node = None # The variable "max_deg_node" contains the maximum degree node ID in the graph "g".
max_deg = None # The variable "max_deg" contains the degree of the "max_deg_node".

###        TODO: Implement       ###
max_deg = 0
for node in g.nodes:
    degree = g.degree[node]
    if degree > max_deg:
        max_deg_node = node
        max_deg = degree
###     Implementation end       ###

print("------------      Task2      ------------")
print("The maximum degree node: %d"%max_deg_node)
print("The degree of the node %d: %d"%(max_deg_node, max_deg))
print("-----------------------------------------")
print()
#####################################################################


######    Task3. Find the maximum pagerank node in the graph    #######
max_pr_node = None # The variable "max_pr_node" contains the maximum pagerank node ID in the graph "g".
max_pr = None # The variable "max_pr" contains the pagerank of the "max_pr_node".

###        TODO: Implement       ###
max_pr = 0
pr = nx.pagerank(g, alpha=0.85)

for node in pr.keys():
    rank = pr[node]
    if rank > max_pr:
        max_pr_node = node
        max_pr = rank
###     Implementation end       ###

print("------------      Task3      ------------")
print("The maximum PageRank node: %d"%max_pr_node)
print("The PageRank of the node %d: %.2f"%(max_pr_node, max_pr))
print("-----------------------------------------")
print()
#####################################################################


## Task4. Find the maximum betweenness centrality edge in the graph  ##
max_bc_edge = None # The variable "max_bc_edge" contains a maximum betweeness edge in the graph "g".
                   # An edge should be a tuple composed of two node IDs (increasing order).
max_bc = None # The variable "max_bc" is the betweeness of the "max_bc_edge".

###        TODO: Implement       ###
max_bc = 0
bc = nx.algorithms.centrality.edge_betweenness_centrality(g)

for edge in bc.keys():
    centrality = bc[edge]
    if centrality > max_bc:
        max_bc_edge = edge
        max_bc = centrality
###     Implementation end       ###

print("------------      Task4      ------------")
print("The maximum betweenness centrality edge: %s"%(max_bc_edge,))
print("The betweenness centrality of the node %s: %.2f"%(max_bc_edge, max_bc))
print("-----------------------------------------")
print()
#####################################################################


###  Task5. Find the nodes with the maximum authority and hub score. ###
max_aut_node = None # The variable "max_aut_node" contains the maximum authority score node ID in the graph "g".
max_hub_node = None # The variable "max_hub_node" contains the maximum hub score node ID in the graph "g".
max_aut = None # The variable "max_aut" contains the authority score of the "max_aut_node".
max_hub = None # The variable "max_hub" contains the hub score of the "max_hub_node".

###        TODO: Implement       ###
max_aut = max_hub = 0
hubs, authorities = nx.hits(g)

for node in g.nodes:
    hub_score = hubs[node]
    aut_score = authorities[node]
    if hub_score > max_hub:
        max_hub = hub_score
        max_hub_node = node
    if aut_score > max_aut:
        max_aut = aut_score
        max_aut_node = node
###     Implementation end       ###

print("------------      Task5      ------------")
print("The maximum autority score node: %d"%max_aut_node)
print("The autority score of the node %d: %.2f"%(max_aut_node, max_aut))
print("The maximum hub score node: %d"%max_hub_node)
print("The hub score of the node %d: %.2f"%(max_hub_node, max_hub))
print("-----------------------------------------")
print()
#####################################################################


# Task6. Find the two communities in the graph using the Girvan Newman method #
# The variable "first_community" is the list of node ids in a community in the graph "g" obtained by the Girvan Newman method.
# The variable "second_community" is the list of node ids in the other community in the graph "g" obtained by the Girvan Newman method.
# More details are provided in the "S20_AI506_HW4.pdf".    
first_community = None 
second_community = None 

###        TODO: Implement       ###
comp = nx.algorithms.community.centrality.girvan_newman(g)
(first_community, second_community) = (sorted(c) for c in next(comp))
if len(first_community) == len(second_community) and 0 in second_community:
    c = second_community
    second_community = first_community
    first_community = c
###     Implementation end       ###

print("------------      Task6      ------------")
print("First community: ")
print(first_community)
print("Second community: ")
print(second_community)
print("-----------------------------------------")
print()
#####################################################################


#  Task7. Find the cut and normalized cut values when partitioning the graph "g" into two communities obtained in Task 6.                  #
cut = None # The variable "cut" contains the cut of two communities obtained in Task 6.
normalized_cut = None # The variable "normalized_cut" contains the normalized cut of the two communities obtained in Task 6.

###        TODO: Implement       ###
cut = nx.cut_size(g, first_community, second_community)
normalized_cut = nx.normalized_cut_size(g, first_community, second_community)
###     Implementation end       ###

print("------------      Task6      ------------")
print("The cut: %d"%cut)
print("The normalized cut: %.10f"%normalized_cut)
print("-----------------------------------------")
print()
#####################################################################
