import networkx as nx
from Tree import Custom_Tree

class ClusteringHandler():
    def __init__(self):
            pass
    def do_all(self,G,solver):
        trees = []
        for g in nx.connected_components(G):
            tree = Custom_Tree()
            graph = G.subgraph(g)
            tree.find_complete_subgraphs_in_connected_graph(graph, graph,None,solver)
            trees.append(tree)
        return trees
    def do_all_and_print(self,G,solver):
        trees = []
        for g in nx.connected_components(G):
            tree = Custom_Tree()
            graph = G.subgraph(g)
            tree.find_complete_subgraphs_in_connected_graph(graph, graph,None,solver)
            trees.append(tree)
        trees.print(trees)
        return trees
            
    def print(self,trees):
        # Print all trees
        for tree in trees:
            print("Tree number", trees.index(tree) + 1)
            tree.print_tree()
            print("---")
        return trees  


    
