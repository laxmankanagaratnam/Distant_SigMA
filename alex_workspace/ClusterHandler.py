import networkx as nx
from alex_workspace.Tree import Custom_Tree
import numpy as np
from collections import defaultdict, Counter

class ClusteringHandler:
    """
    A handler for clustering and creating custom trees from a NetworkX graph.

    This class provides methods to generate custom trees from a graph's connected components, 
    utilizing different solvers for analyzing subgraphs, and printing the generated trees.
    """

    def __init__(self):
        """
        Initializes a new instance of ClusteringHandler.
        """
        pass

    def do_all(self, G, solver):
        """
        Creates custom trees from each connected component in the graph.

        This method generates a list of trees by extracting connected components from the given graph 
        and applying a solver to find complete subgraphs within each component.

        Args:
            G (nx.Graph): The graph to process.
            solver (callable): A function or method used to analyze and process the subgraphs.

        Returns:
            list[Custom_Tree]: A list of custom trees generated from the graph's connected components.
        """
        trees = []
        # Iterate over connected components and create trees
        for g in nx.connected_components(G):
            tree = Custom_Tree()
            graph = G.subgraph(g)
            tree.find_complete_subgraphs_in_connected_graph(graph, graph, None, solver)
            trees.append(tree)
        return trees

    def do_all_and_print(self, G, solver):
        """
        Creates custom trees from each connected component and prints them.

        This method generates a list of trees, applying the given solver to each connected component,
        then prints the trees to the console.

        Args:
            G (nx.Graph): The graph to process.
            solver (callable): A function or method used to analyze and process the subgraphs.

        Returns:
            list[Custom_Tree]: A list of custom trees, printed and returned.
        """
        trees = []
        for g in nx.connected_components(G):
            tree = Custom_Tree()
            graph = G.subgraph(g)
            tree.find_complete_subgraphs_in_connected_graph(graph, graph, None, solver)
            trees.append(tree)
        
        # Print each tree and return the list of trees
        self.print(trees)
        return trees

    def print(self, trees):
        """
        Prints all trees in the provided list.

        This method iterates through the list of custom trees and prints each tree with a separator for clarity.

        Args:
            trees (list[Custom_Tree]): A list of custom trees to print.

        Returns:
            list[Custom_Tree]: The same list of custom trees, for chaining or further processing.
        """
        for index, tree in enumerate(trees):
            print(f"Tree number {index + 1}")
            tree.print_tree()
            print("---")  # Separator for readability
        return trees
    
    def tree_as_cluster_list(self,trees,tmp):
        output_nodes = []
        cluster_label_list = []
        # if trees not list
        if not isinstance(trees, list):
            trees = [trees]
        if len(trees) == 0:
            return []
        for tree in trees:
            output_nodes.extend(tree.get_leaf_nodes())
        for node in output_nodes:   
            cluster_label_list.append(self.count_label_node(node,tmp))
        final_list = []
        for index in range(len(cluster_label_list[0])):
            # get maximum value of all lists at index
            max_value = max([cluster_label_list[i][index] for i in range(len(cluster_label_list))])
            if max_value == 0:
                final_list.append(-1)
                continue
            # also provide the i of the list with the maximum value
            max_index = [i for i in range(len(cluster_label_list)) if cluster_label_list[i][index] == max_value]
            # only insert the index of the first list with the maximum value
            # convert the index from 1 to 1.0 for example, float
            final_list.append(max_index[0])
        return np.array(final_list)

    def full_pipline_tree_hierachy(self,graph,solver,tmp,plotter,threshold,merge_option = 0):
        trees = self.do_all(graph, solver)
        for tree in trees:
            tree.merge(plotter,threshold,merge_option)
        labels = self.tree_as_cluster_list(trees,tmp)
        return labels
            

                
                
    def count_label_node(self,node,tmp):
        """
        Extracts and processes label data for a single node based on the temporary data list.

        Args:
            node (Custom_tree_node): A node from which labels are extracted.

        Returns:
            numpy.ndarray: An array of label data processed from the node.
        """
        if node.name == "root":
            return np.array([])
        local_lists = []  # Store lists for each sub-node
        for sub_node in node.name.split("+"):
            local_lists.append(tmp[int(sub_node)])

        # Transpose to work with columns (indices)
        transposed_lists = np.array(local_lists).T 

        # Count '1' occurrences in each column
        occurrence_counts = np.sum(transposed_lists == True, axis=1)

        return occurrence_counts
