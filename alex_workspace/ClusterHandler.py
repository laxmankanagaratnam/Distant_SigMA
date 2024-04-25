import networkx as nx
from Tree import Custom_Tree

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
