import sys
import os
import time
import networkx as nx

# Setting up system paths
# This includes the project directories to the system path to ensure the custom modules can be imported without issue.
# TODO: Ensure the paths specified below are correct for the current setup.
sys.path.append(os.path.abspath(r"\Users\Alexm\OneDrive - Universität Wien\01_WINF\Praktikum1\SigMA_Alex_modifications"))
sys.path.append(os.path.abspath(r"\Users\Alexm\OneDrive - Universität Wien\01_WINF\Praktikum1\SigMA_Alex_modifications\alex_workspace"))

# Importing custom classes
# These classes handle various functionalities such as clustering, graph manipulation, and plotting.
from ClusterHandler import ClusteringHandler
from GraphCreator import GraphCreator
from Tree import Custom_Tree
from NxGraphAssistant import NxGraphAssistant
from PlotHandler import PlotHandler

# Creating instances of custom classes
clusterMaster = ClusteringHandler()  # Manages cluster operations
orion_index =  0  # Index for selecting specific data or configuration; needs to be defined based on context.
graph_base, tmp = GraphCreator().get_get_orion(orion_index)  # Retrieves a base graph and temporary data for the Orion dataset.

# Graph manipulation
# Removing edges from the graph that do not meet certain criteria based on weight and analyzing cliques within the graph.
#FIXME optimize me
graph = NxGraphAssistant.remove_edges_with_minor(graph_base, 'weight', 0.3, 0.7)
#FIXME optimize me
graph = NxGraphAssistant.analyze_cliques(graph, 0.95)

graph = graph_base



# Measure time for clustering operations
# Timing the execution of clustering operations to compare the efficiency of different clustering strategies.
start_time1 = time.time()
T = clusterMaster.do_all(graph, Custom_Tree.alex_optimal)  # Performs clustering using a specific tree structure strategy.
end_time1 = time.time()

start_time2 = time.time()
Z = clusterMaster.do_all(graph, Custom_Tree.alex_optimal_alternative)  # An alternative clustering strategy.
end_time2 = time.time()

start_time3 = time.time()
A = clusterMaster.do_all(graph, Custom_Tree.tree_size_problem_handler_create_multiple_trees_on_conflict)  # Handles size conflicts in trees during clustering.
end_time3 = time.time()

# Print elapsed times
elapsed_time1 = end_time1 - start_time1
elapsed_time2 = end_time2 - start_time2
elapsed_time3 = end_time3 - start_time3
print("Elapsed time for the first code snippet:", elapsed_time1, "seconds")
print("Elapsed time for the second code snippet:", elapsed_time2, "seconds")
print("Elapsed time for the third code snippet:", elapsed_time3, "seconds")
output = clusterMaster.tree_as_cluster_list(T,tmp)
print("----------------------------------------------------")
print(output)
    
