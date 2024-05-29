import sys
import os

sys.path.append(
    os.path.abspath(r"\Users\Alexm\OneDrive - Universität Wien\01_WINF\Praktikum1\SigMA_Alex_modifications"))
sys.path.append(os.path.abspath(
    r"\Users\Alexm\OneDrive - Universität Wien\01_WINF\Praktikum1\SigMA_Alex_modifications\alex_workspace"))

from ClusterHandler import ClusteringHandler
from GraphCreator import GraphCreator
from Tree import Custom_Tree
from NxGraphAssistant import NxGraphAssistant
from PlotHandler import PlotHandler

clusterMaster = ClusteringHandler()
orion_index = 0
graph_base, tmp = GraphCreator().get_get_orion(orion_index)

import time

graph = NxGraphAssistant.remove_edges_with_minor(graph_base, 'weight', 0.3, 0.8)
graph = NxGraphAssistant.analyze_cliques(graph, 0.8)

graph2 = NxGraphAssistant.remove_edges(graph_base, similarity='weight', threshold=0.3)
graph2 = NxGraphAssistant.analyze_cliques(graph2, 0.7)

# Measure time for the second code snippet
start_time1 = time.time()
T = clusterMaster.do_all(graph_base, Custom_Tree.alex_optimal)
end_time1 = time.time()
start_time2 = time.time()
Z = clusterMaster.do_all(graph_base, Custom_Tree.alex_optimal_alternative)
end_time2 = time.time()
start_time3 = time.time()
A = clusterMaster.do_all(graph_base, Custom_Tree.tree_size_problem_handler_create_multiple_trees_on_conflict)
end_time3 = time.time()

# Calculate the elapsed time for each code snippet
elapsed_time1 = end_time1 - start_time1
elapsed_time2 = end_time2 - start_time2
elapsed_time3 = end_time3 - start_time3

# Print the elapsed time for each code snippet
print("Elapsed time for the first code snippet:", elapsed_time1, "seconds")
print("Elapsed time for the second code snippet:", elapsed_time2, "seconds")
print("Elapsed time for the third code snippet:", elapsed_time3, "seconds")





