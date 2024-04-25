import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import sys
import os

sys.path.append(
    os.path.abspath(r"\Users\Alexm\OneDrive - Universität Wien\01_WINF\Praktikum1\SigMA_Alex_modifications"))
from ConcensusClustering.consensus import ClusterConsensus


def get_orion_data(num=1):
    # change to the location of the directory containing the label data
    label_path = r'C:\Users\Alexm\OneDrive - Universität Wien\01_WINF\Praktikum1\SigMA_Alex_modifications\alex_workspace\Grouped_solution_labels\Grouped_solution_labels/'

    # Orion is split into 5 regions (numbered 0 - 4)
    ## Region 2 is the largest (22 groups)
    ## Regions 0 and 4 are the smallest

    regions = [f'Region_{i}/' for i in range(5)]

    # pick the region you want to work with
    r = 1

    region = regions[r]
    grouped_labels = pd.read_csv(label_path + region + f'grouped_solutions_chunk_{r}.csv',
                                 header=None).to_numpy()  # load labels
    print(f"There are {grouped_labels.shape[0]} grouped solutions for region {r}.")
    # density = pd.read_csv(label_path+region+f'Density_chunk_{r}.csv', header=None).to_numpy() # load density (for cc.remove_edges_density)
    # rho =density.reshape(len(density),)

    # create graph
    cc = ClusterConsensus(*grouped_labels)

    # reanme the similarity in every edge to "weight"
    for i, j, d in cc.G.edges(data=True):
        d['weight'] = d['similarity']
        d['weight_minor'] = d['similarity_minor']
        del d['similarity']
        del d['similarity_minor']
    return cc.G


import networkx as nx
import random
import networkx as nx
import matplotlib.pyplot as plt


class GraphCreator():
    def __init__(self):
        pass

    # easy example Sebastian
    def create_easyGraph(self):
        G = nx.Graph()
        G.add_nodes_from(['A1', 'A2', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3', 'C4'])
        G.add_edges_from([
            ('A1', 'B1'), ('A1', 'B2'), ('A1', 'C1'), ('A1', 'C2'), ('A1', 'C3'), ('B1', 'C1'),
            ('B2', 'C2'), ('B2', 'C3'),
            ('A2', 'C4'), ('A2', 'B3'), ('B3', 'C4')
        ])

        # Add random weights to all edges
        for u, v in G.edges():
            G[u][v]['weight'] = round(random.uniform(0.1, 1.0), 3)

        return G

    # advanced example Sebastian
    def create_advanced_graph(self):
        G = nx.Graph()
        G.add_nodes_from(
            ['A1', 'A2', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3', 'C4', 'D1', 'D2', 'D3', 'D4', 'D5', 'E1', 'E2', 'E3', 'E4',
             'E5'])
        G.add_edges_from([
            ('A1', 'B1'), ('A1', 'B2'), ('A1', 'C1'), ('A1', 'C2'), ('A1', 'C3'), ('A1', 'D1'), ('A1', 'D2'),
            ('A1', 'D3'), ('A1', 'D4'), ('A1', 'E1'), ('A1', 'E2'), ('A1', 'E3'), ('A1', 'E4'),
            ('A2', 'B3'), ('A2', 'C4'), ('A2', 'D5'), ('A2', 'E5'),
            ('B1', 'C1'), ('B1', 'D1'), ('B1', 'E1'),
            ('B2', 'C2'), ('B2', 'D2'), ('B2', 'E2'), ('B2', 'C3'), ('B2', 'D3'), ('B2', 'D4'), ('B2', 'E3'),
            ('B2', 'E4'),
            ('B3', 'C4'), ('B3', 'D5'), ('B3', 'E5'),
            ('C1', 'D1'), ('C1', 'E1'),
            ('C2', 'D3'), ('C2', 'D4'), ('C2', 'E4'), ('C2', 'E2'),
            ('C3', 'D2'), ('C3', 'E2'), ('C3', 'E3'),
            ('C4', 'D5'), ('C4', 'E5'),
            ('D1', 'E1'),
            ('D2', 'E2'), ('D2', 'E3'),
            ('D3', 'E2'),
            ('D4', 'E4'),
            ('D5', 'E5'), ]

        )
        # Add random weights to all edges that simulate a pair of jaccardian similarity and jaccardian similiarity
        for u, v in G.edges():
            G[u][v]['weight'] = round(random.uniform(0.1, 1.0), 3)

        return G

    # martin example 0
    def create_GraphX0(self):
        G = nx.Graph()
        G.add_nodes_from([1, 3, 4, 6, 10, 11])
        G.add_edges_from([
            (1, 3), (1, 4), (1, 6), (1, 10), (1, 11),
            (3, 6), (3, 10), (3, 11),
            (4, 10), (4, 11),
            (6, 10), (6, 11)
        ])
        # Add weights to all edges
        for u, v in G.edges():
            G[u][v]['weight'] = 0.5
        return G

    # random graph
    def create_random_graph(num_nodes, edge_probability):
        G = nx.Graph()
        nodes = list(range(1, num_nodes + 1))  # Nodes from 1 to num_nodes
        G.add_nodes_from(nodes)

        # Add random edges
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if random.random() < edge_probability:  # Edge probability as parameter
                    G.add_edge(nodes[i], nodes[j])

        # Add weights to all edges
        for u, v in G.edges():
            G[u][v]['weight'] = 0.5

        return G

    # random graph with weights
    def create_random_graph_with_weights(num_nodes, edge_probability):
        G = nx.Graph()
        nodes = list(range(1, num_nodes + 1))  # Nodes from 1 to num_nodes
        G.add_nodes_from(nodes)

        # Add random edges
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if random.random() < edge_probability:  # Edge probability as parameter
                    G.add_edge(nodes[i], nodes[j])

        # Add weights to all edges
        for u, v in G.edges():
            G[u][v]['weight'] = round(random.uniform(0.1, 1.0), 3)

        return G

    # martin graph 1
    def create_GraphX1(self):
        G = nx.Graph()
        G.add_nodes_from([1, 3, 4, 6, 10, 11])
        G.add_edges_from([
            (1, 3), (1, 4), (1, 6), (1, 10), (1, 11),
            (3, 6), (3, 10), (3, 11),
            (4, 10), (4, 11),
            (6, 10), (6, 11)
        ])
        # Add weights to all edges
        predefined_w = [1.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0, 1.0 / 2.0, 1.0 / 2.0,
                        1.0 / 2.0, 1.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0,
                        2.0 / 5.0, 2.0 / 5.0]
        uv = 0
        for u, v in G.edges():
            G[u][v]['weight'] = predefined_w[uv]
            uv += 1
        return G

    # marin graph 2
    def create_GraphX2(self):
        G = nx.Graph()
        G.add_nodes_from([1, 3, 4, 6, 10, 11])
        G.add_edges_from([
            (1, 3), (1, 4), (1, 6), (1, 10), (1, 11),
            (3, 6), (3, 10), (3, 11),
            (4, 10), (4, 11),
            (6, 10), (6, 11)
        ])
        # Add weights to all edges
        predefined_w = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 2.0, 1.0 / 2.0,
                        1.0 / 3.0, 1.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0,
                        1.0 / 4.0, 1.0 / 4.0]
        uv = 0
        for u, v in G.edges():
            G[u][v]['weight'] = predefined_w[uv]
            uv += 1
        return G

    def create_GraphX3(self):
        G = nx.Graph()
        G.add_nodes_from([1, 3, 4, 6, 10, 11])
        G.add_edges_from([
            (1, 3), (1, 4), (1, 6), (1, 10), (1, 11),
            (3, 6), (3, 10), (3, 11),
            (4, 10), (4, 11),
            (6, 10), (6, 11)
        ])
        # Add weights to all edges
        predefined_w = [1.0, 0.001, 0.7, 1.0, 0.001,
                        0.7, 1.0, 0.001, 0.001, 0.9,
                        0.7, 0.001]
        uv = 0
        for u, v in G.edges():
            G[u][v]['weight'] = predefined_w[uv]
            uv += 1
        return G

    def get_get_orion(self, num=1):
        return get_orion_data()


class NxGraphAssistant():

    def __init__(self):
        pass

    def is_complete_graph(G):
        """
        Check if a graph is complete.

        Parameters:
        - G: NetworkX graph

        Returns:
        - True if the graph is complete, False otherwise
        """
        for node in G:
            if len(G[node]) != len(G) - 1:
                return False
        return True

    def is_same_connections(graph, node1, node2):
        """
        Checks if two nodes have the same connections in a graph.

        Parameters:
        - graph: NetworkX graph
        - node1: node identifier
        - node2: node identifier

        Returns:
        - bool: True if nodes have the same connections, False otherwise
        """
        try:
            return (graph.neighbors(node1)) == set(graph.neighbors(node2))
        except:
            return False

    def all_connection_similar_but_to_each_other(graph, node1, node2):
        """
        Checks if two nodes have the same connections in a graph but are not connected to each other.

        Parameters:
        - graph: NetworkX graph
        - node1: node identifier
        - node2: node identifier

        Returns:
        - bool: True if nodes have the same connections but are not connected to each other, False otherwise
        """
        try:
            set1 = set(graph.neighbors(node1))
            set2 = set(graph.neighbors(node2))

            if node2 in set1:
                set1.remove(node2)
            if node1 in set2:
                set2.remove(node1)

            return set1 == set2
        except Exception as e:
            print(f"An error occurred: {e}")
            return False

    def connected(graph, node1, node2):
        """
        Checks if two nodes are connected in a graph.

        Parameters:
        - graph: NetworkX graph
        - node1: node identifier
        - node2: node identifier

        Returns:
        - bool: True if nodes are connected, False otherwise
        """
        try:
            return nx.has_path(graph, node1, node2)
        except:
            return False

    @staticmethod
    def all_most_connected_nodes(entire_graph: nx.Graph, sub_graph):
        try:
            most_connected_nodes = [max(sub_graph, key=lambda x: len(entire_graph[x]))]
            for node in sub_graph:
                if len(entire_graph[node]) == len(entire_graph[most_connected_nodes[0]]) and node != \
                        most_connected_nodes[0]:
                    most_connected_nodes.append(node)

            return most_connected_nodes
        except:
            return []

    @staticmethod
    def analyze_cliques(G, treshold=0.9):
        # Find cliques with Jaccardian similarity higher than 0.9

        cliques = [clique for clique in nx.find_cliques(G) if
                   all(G.has_edge(u, v) and G[u][v]['weight'] > treshold for u, v in nx.utils.pairwise(clique))]

        # Calculate average Jaccardian value for each clique
        avg_jaccard_values = {}
        for clique in cliques:
            total_jaccard_value = sum(G[u][v]['weight'] for u, v in nx.utils.pairwise(clique))
            avg_jaccard_value = total_jaccard_value / len(clique)
            avg_jaccard_values[tuple(clique)] = avg_jaccard_value

        # Sort cliques by average Jaccardian value in descending order
        sorted_cliques = sorted(avg_jaccard_values.keys(), key=lambda x: avg_jaccard_values[x], reverse=True)
        print("Sorted Cliques:")
        print(sorted_cliques)
        print("Avg Jaccard Values:")
        print(avg_jaccard_values)
        # Keep track of assigned nodes
        assigned_nodes = set()

        # remove all cliques that have nodes that are already assigned
        selected_cliques = []
        for clique in sorted_cliques:
            if not assigned_nodes.intersection(clique):
                selected_cliques.append(clique)
                assigned_nodes.update(clique)

        print("Selected Cliques:")
        print(selected_cliques)

        # create copy of graph G
        new_graph = G.copy()
        # iterate all cliques
        for clique in selected_cliques:
            # create a new node for the clique
            name = ""
            for node in clique:
                if name != "":
                    name += "+"
                name += str(node)
            new_node = name
            # add the new node to the new graph
            new_graph.add_node(new_node)
            # remove the old nodes from the new graph
            new_graph.remove_nodes_from(clique)
            # add edges between the new clique and the old nodes each clique member was connected to
            for node in clique:
                for neighbor in G.neighbors(node):
                    if neighbor not in clique and neighbor in new_graph.nodes():
                        # TODO also add weight (average)
                        new_graph.add_edge(new_node, neighbor)
                        # print("Added edge between", new_node, "and", neighbor)
        return new_graph

    @staticmethod
    def plot_networkX_graph(G):
        import networkx as nx
        import matplotlib.pyplot as plt

        # Adjust the layout of the graph for better readability
        pos = nx.spring_layout(G, k=0.15)  # You can adjust the value of 'k' for desired spacing

        nx.draw(G, pos, with_labels=True, font_weight='bold')
        plt.show()

    @staticmethod
    def remove_edges(graph, similarity='weight', threshold=0.5):
        G_copy = nx.Graph(graph)
        # --- test if we need to remove edges
        e2js = {frozenset({e1, e2}): G_copy[e1][e2][similarity] for e1, e2 in G_copy.edges}
        # --- remove edges with unmatching cluster solutions
        re = [(e1, e2) for (e1, e2), sim in e2js.items() if sim < threshold]
        G_copy.remove_edges_from(re)
        return G_copy

    @staticmethod
    def remove_edges_with_minor(G, similarity='weight', treshhold=0.3, treshhold_minor=0.8):
        G_copy = nx.Graph(G)
        # --- test if we need to remove edges
        e2js = {frozenset({e1, e2}): G_copy[e1][e2][similarity] for e1, e2 in G_copy.edges}
        # --- remove edges with unmatching cluster solutions
        remove_edges = []
        for (e1, e2), js in e2js.items():
            if js < treshhold:
                js_minor = G_copy[e1][e2][similarity + "_minor"]
                if js_minor < treshhold_minor:
                    remove_edges.append((e1, e2))
        G_copy.remove_edges_from(remove_edges)
        return G_copy

    @staticmethod
    def jaccard_similarity_minor(i, j):
        nb_i = labels_bool_dict2arr[i].sum()
        nb_j = labels_bool_dict2arr[j].sum()
        intersection = np.sum(
            (labels_bool_dict2arr[i].astype(int) + labels_bool_dict2arr[j].astype(int)) == 2)
        return intersection / min(nb_i, nb_j)

    def analyze_cliques_new(G, threshold=0.5, sim='weight'):
        graph = G.copy()
        while True:
            print("iteration")
            cliques = [clique for clique in nx.find_cliques(graph) if
                       all(graph.has_edge(u, v) and graph[u][v][sim] > threshold for u, v in nx.utils.pairwise(clique))]
            if not cliques:
                print("break")
                break

            # Calculate average Jaccardian value for each clique
            avg_jaccard_values = {}
            for clique in cliques:
                total_jaccard_value = sum(graph[u][v][sim] for u, v in nx.utils.pairwise(clique))
                avg_jaccard_value = total_jaccard_value / len(clique)
                avg_jaccard_values[tuple(clique)] = avg_jaccard_value

            # Sort cliques by average Jaccardian value in descending order
            sorted_cliques = sorted(avg_jaccard_values.keys(), key=lambda x: avg_jaccard_values[x], reverse=True)
            # remove all cliques with to low Jaccardian value
            sorted_cliques = [clique for clique in sorted_cliques if avg_jaccard_values[clique] > threshold]
            if not sorted_cliques:
                print("break")
                break
            print("Sorted Cliques:")
            print(sorted_cliques)
            print("Avg Jaccard Values:")
            print(avg_jaccard_values)

            # Keep track of assigned nodes
            assigned_nodes = set()  #
            # Remove all cliques that have nodes that are already assigned
            selected_cliques = []
            for clique in sorted_cliques:
                if not assigned_nodes.intersection(clique):
                    selected_cliques.append(clique)
                    assigned_nodes.update(clique)
            print("Selected Cliques:")
            print(selected_cliques)

            # Create a copy of the graph
            new_graph = graph.copy()

            # Iterate over the selected cliques
            for clique in selected_cliques:
                # Create a new node for the clique
                name = "+".join(str(node) for node in clique)
                new_node = name

                # Add the new node to the new graph
                new_graph.add_node(new_node)

                # Remove the old nodes from the new graph
                new_graph.remove_nodes_from(clique)

                # Add edges between the new clique and the old nodes each clique member was connected to
                for node in clique:
                    for neighbor in graph.neighbors(node):
                        if neighbor not in clique and neighbor in new_graph.nodes():
                            # calculate the average similarity between the new node and the neighbor
                            summe = 0
                            count = 0
                            for n in clique:
                                # check if node is connected to the neighbor
                                if graph.has_edge(n, neighbor):
                                    summe += graph[n][neighbor][sim]
                                    count += 1
                            if count > 0:

                                new_graph.add_edge(new_node, neighbor, **{sim: summe / count})
                            else:

                                new_graph.add_edge(new_node, neighbor, **{sim: 0})

            graph = new_graph
            # check how many nodes in the graph
            if len(graph) == 1:
                break
        return graph


import networkx as nx
import uuid


class Custom_tree_node:
    def __init__(self, name):
        self.name = name
        self.uuid = uuid.uuid4()
        self.children = []
        self.parent = None

    def add_child(self, child) -> 'Custom_tree_node':
        child.parent = self
        self.children.append(child)
        return child

    def get_all_parts(self) -> list[str]:
        parts = []
        # split name by '+' and add each part to the list
        for part in self.name.split("+"):
            parts.append(part)
        return parts

    def create_own_hash(self):
        # create hasx from self.name
        return hash(self.name)

    def get_child_hash(self):
        sum_name = ""
        for child in self.children:
            sum_name += child.name
        return hash(sum_name)

    def create_child_hash(self):
        sum_name = ""
        child_list = []
        child_list.append(self)
        while (len(child_list) > 0):
            for item in child_list:
                for child in item.children:
                    sum_name += str(child.name)

                    if child.children:
                        child_list.append(child)
                child_list.remove(item)
        return hash(sum_name)

    def get_complete_hash(self):
        return hash(self.create_own_hash() + self.create_child_hash())



class Counter:
    def __init__(self):
        self.count = 0
    def increment(self):
        self.count += 1
        return self.count
    def reset(self):
        self.count =0
# some list assistance

from collections import OrderedDict

class ListHelper:
    @staticmethod
    def unique_list(lst):
        # check if list
        if not isinstance(lst, list):
            return lst
        result = []
        for elem in lst:
            # check if empty
            if isinstance(elem, int):
                continue
            if len(elem) == 0:
                continue
            if elem not in result:
                result.append(elem)
        return result
    @staticmethod
    def remvoe_one_size(lst):
        result = []
        for elem in lst:
            if len(elem) == 1:
                continue
            result.append(elem)
        return result



class Custom_Tree:
    """
    A custom tree data structure implementation.

    Attributes:
    - root: The root node of the tree.
    - graph: The graph representation of the tree.

    Methods:
    - add_node(parent_uuid, child_name): Adds a new node with the given child name as a child of the node with the specified parent UUID.
    - add_node_efficent(parent, child_name): Adds a new node with the given child name as a child of the specified parent node.
    - get_size(node): Returns the size of the tree starting from the specified node.
    - get_depth(node): Returns the depth of the tree starting from the specified node.
    - _find_node(node, target_uuid): Finds and returns the node with the specified UUID starting from the specified node.
    - print_tree(node): Prints the tree starting from the specified node.
    - check_subtree_depth_n(current_graph, depth, mode): Checks if there is a subtree in the current graph with a depth of n.
    - improved_problem_handler_create_multiple_trees_on_conflict(current_graph, most_connected_nodes, last_node): Handles conflicts by creating multiple subtrees for each most connected node.
    - problem_handler_create_multiple_trees_on_conflict(current_graph, most_connected_nodes, last_node): Handles conflicts by creating multiple subtrees for each most connected node.
    - find_complete_subgraphs_in_connected_graph(G, current_graph, last_node, problem_solver): Finds complete subgraphs in the connected graph and adds them to the tree.

    """

    def __init__(self):
        self.root = None
        self.graph = nx.Graph()

    def add_node(self, parent_uuid, child_name):
        """
        Adds a new node with the given child name as a child of the node with the specified parent UUID.

        Args:
        - parent_uuid: The UUID of the parent node.
        - child_name: The name of the child node.

        Returns:
        - The newly added node.

        """
        if not self.root:
            self.root = Custom_tree_node(child_name)
            return self.root

        node_return = self.root
        parent_node = self._find_node(self.root, parent_uuid)
        if parent_node:
            node_return = parent_node.add_child(Custom_tree_node(child_name))
        else:
            print("Parent node not found.")
        # return new node
        return node_return

    def add_node_efficent(self, parent, child_name):
        """
        Adds a new node with the given child name as a child of the specified parent node.

        Args:
        - parent: The parent node.
        - child_name: The name of the child node.

        Returns:
        - The newly added node.

        """
        if not self.root:
            self.root = Custom_tree_node(child_name)
            return self.root
        node_return = self.root
        if parent:
            node_return = parent.add_child(Custom_tree_node(child_name))
        else:
            print("Parent node not found.")
        return node_return

    def get_size(self, node=None):
        """
        Returns the size of the tree starting from the specified node.

        Args:
        - node: The starting node. If not specified, the root node is used.

        Returns:
        - The size of the tree.

        """
        if not node:
            node = self.root

        if not node.children:
            return 1

        size = 1
        for child in node.children:
            size += self.get_size(child)

        return size

    def get_depth(self, node=None, depth=0):
        """
        Returns the depth of the tree starting from the specified node.

        Args:
        - node: The starting node. If not specified, the root node is used.
        - depth: The current depth. Defaults to 0.

        Returns:
        - The depth of the tree.

        """
        if not node:
            node = self.root

        if not node.children:
            return depth

        max_depth = depth
        for child in node.children:
            child_depth = self.get_depth(child, depth + 1)
            max_depth = max(max_depth, child_depth)

        return max_depth

    def _find_node(self, node, target_uuid):
        """
        Finds and returns the node with the specified UUID starting from the specified node.

        Args:
        - node: The starting node.
        - target_uuid: The UUID of the node to find.

        Returns:
        - The node with the specified UUID, or None if not found.

        """
        if node.uuid == target_uuid:
            return node
        for child in node.children:
            found = self._find_node(child, target_uuid)
            if found:
                return found
        return None

    def print_tree(self, node=None, depth=0):
        """
        Prints the tree starting from the specified node.

        Args:
        - node: The starting node. If not specified, the root node is used.
        - depth: The current depth. Defaults to 0.

        """
        if not node:
            node = self.root
        # print(f"{node.name}({node.uuid})")
        print(f"{node.name}")
        if node.children:
            for i, child in enumerate(node.children):
                if i < len(node.children) - 1:
                    print("  " * (depth + 1) + "├── ", end="")
                else:
                    print("  " * (depth + 1) + "└── ", end="")
                self.print_tree(child, depth + 2)

    def check_subtree_depth_n(self, current_graph, depth, mode="remove-all-most-connected-nodes"):
        """
        Checks if there is a subtree in the current graph with a depth of n.

        Args:
        - current_graph: The current graph.
        - depth: The desired depth.
        - mode: The mode for removing nodes. Defaults to "remove-all-most-connected-nodes".

        Returns:
        - True if a subtree with the desired depth is found, False otherwise.

        """
        # print("iteration1",current_graph)
        for current_subgraph in nx.connected_components(current_graph):
            # print("iteration2",current_subgraph)
            if NxGraphAssistant.is_complete_graph(self.graph.subgraph(current_subgraph)) or len(current_subgraph) == 1:
                return True
            if depth > 1:
                if mode == "remove-all-most-connected-nodes":
                    most_connected_nodes = NxGraphAssistant.all_most_connected_nodes(self.graph, current_subgraph)
                    # remove all most connected nodes from the set current_subgraph
                    for node in most_connected_nodes:
                        current_subgraph.remove(node)
                    if len(current_subgraph) == 0:
                        return True
                    # print("iteration3",current_subgraph)
                elif mode == "remove-one-most-connected-nodes":
                    most_connected_nodes = NxGraphAssistant.all_most_connected_nodes(self.graph, current_subgraph)
                    # try sorting the most connected nodes for deterministic behavior
                    try:
                        sorted(most_connected_nodes)
                    except:
                        # it can fail in some casses if type is not supported
                        pass
                    current_subgraph.remove(most_connected_nodes[0])
                    # print("iteration3",current_subgraph)

                if self.check_subtree_depth_n(self.graph.subgraph(current_subgraph), depth - 1, mode):
                    return True
            else:
                # print("depth reached",current_subgraph)
                pass
        return False

    def improved_problem_handler_create_multiple_trees_on_conflict(self, current_graph, most_connected_nodes,
                                                                   last_node=None):
        """
        Handles conflicts by creating multiple subtrees for each most connected node and adds them to the tree.

        Args:
        - current_graph: The current graph.
        - most_connected_nodes: The most connected nodes in the current graph.
        - last_node: The last node added to the tree. Defaults to None.

        """
        # print ("Improved Problem Handler")
        saved_node = last_node
        for most_connected_node in most_connected_nodes:
            edited_graph = self.graph.subgraph(current_graph).copy()  # Create a mutable copy of the subgraph
            nodes_to_remove = [node for node in most_connected_nodes if node != most_connected_node]
            edited_graph.remove_nodes_from(nodes_to_remove)

            nodes_to_remove = []
            for node in current_graph:
                if not NxGraphAssistant.connected(edited_graph, most_connected_node, node):
                    nodes_to_remove.append(node)
            edited_graph.remove_nodes_from(nodes_to_remove)

            for current_subgraph in nx.connected_components(edited_graph):
                if self.check_subtree_depth_n(self.graph.subgraph(current_subgraph), depth=2,
                                              mode="remove-all-most-connected-nodes"):
                    # print("Subtree is complete")
                    self.find_complete_subgraphs_in_connected_graph(self.graph, current_subgraph, last_node,
                                                                    problem_solver=Custom_Tree.improved_problem_handler_create_multiple_trees_on_conflict)
                else:
                    pass
                    # print("Subtree is not complete")
                    # print("subtree",current_subgraph)
        # create hashset
        hash_map = {}
        for most_connected_node in saved_node.children:
            print("most connected node", most_connected_node.name)
            hash = most_connected_node.create_child_hash()
            if hash in hash_map:
                hash_map[hash].append(most_connected_node)
            else:
                hash_map[hash] = [most_connected_node]
        print(hash_map)
        # merge all similar nodes together
        # for key in hash_map:
        # if len(hash_map[key]) > 1:
        # for i in range(1,len(hash_map[key])):

    def dynamic_problem_handler_create_multiple_trees_on_conflict(self, current_graph, most_connected_nodes,
                                                                  last_node=None):
        """
        Handles conflicts by creating multiple subtrees for each most connected node and adds them to the tree.

        Args:
        - current_graph: The current graph.
        - most_connected_nodes: The most connected nodes in the current graph.
        - last_node: The last node added to the tree. Defaults to None.

        """
        current_depth = self.get_depth()
        # print ("Dynamic Problem Handler")

        for most_connected_node in most_connected_nodes:
            edited_graph = self.graph.subgraph(current_graph).copy()  # Create a mutable copy of the subgraph
            nodes_to_remove = [node for node in most_connected_nodes if node != most_connected_node]
            edited_graph.remove_nodes_from(nodes_to_remove)

            nodes_to_remove = []
            for node in current_graph:
                if not NxGraphAssistant.connected(edited_graph, most_connected_node, node):
                    nodes_to_remove.append(node)
            edited_graph.remove_nodes_from(nodes_to_remove)

            for current_subgraph in nx.connected_components(edited_graph):
                if self.check_subtree_depth_n(self.graph.subgraph(current_subgraph), depth=10,
                                              mode="remove-all-most-connected-nodes"):
                    # print("Subtree is complete")
                    self.find_complete_subgraphs_in_connected_graph(self.graph, current_subgraph, last_node,
                                                                    problem_solver=Custom_Tree.dynamic_problem_handler_create_multiple_trees_on_conflict)
                else:
                    # print("Subtree is not complete")
                    # print("subtree",current_subgraph)
                    pass

    # hard coded tree size searches if under 30
    def tree_size_problem_handler_create_multiple_trees_on_conflict(self, current_graph, most_connected_nodes,
                                                                    last_node=None):
        size = 30
        for most_connected_node in most_connected_nodes:
            edited_graph = self.graph.subgraph(current_graph).copy()  # Create a mutable copy of the subgraph
            nodes_to_remove = [node for node in most_connected_nodes if node != most_connected_node]
            edited_graph.remove_nodes_from(nodes_to_remove)

            nodes_to_remove = []
            for node in current_graph:
                if not NxGraphAssistant.connected(edited_graph, most_connected_node, node):
                    nodes_to_remove.append(node)
            edited_graph.remove_nodes_from(nodes_to_remove)

            for current_subgraph in nx.connected_components(edited_graph):
                if self.get_size() < size:
                    self.find_complete_subgraphs_in_connected_graph(self.graph, current_subgraph, last_node,
                                                                    problem_solver=Custom_Tree.tree_size_problem_handler_create_multiple_trees_on_conflict)
        if True:
            hash_map = {}
            for most_connected_node in last_node.children:
                hash = most_connected_node.create_child_hash()
                if hash in hash_map:
                    hash_map[hash].append(most_connected_node)
                else:
                    hash_map[hash] = [most_connected_node]
            for key in hash_map:
                print("key", key)
                for node in hash_map[key]:
                    print(node.name)
            for key in hash_map:
                if key != 0 and len(hash_map[key]) > 1:
                    name = ""
                    childs = []
                    for node in hash_map[key]:
                        last_node.children.remove(node)
                        name += str(node.name) + "+"
                        childs = node.children
                    print("merged together duplicate nodes", name[:-1])
                    new_node = Custom_tree_node(name[:-1])
                    last_node.add_child(new_node)
                    new_node.children = childs
        return

    # muss mal schauen ob das funktioniert

    # hard coded tree size searches if under 30
    def alex_optimal(self, current_graph, most_connected_nodes, last_node=None):
        search_depth = 25 - self.get_depth()
        if search_depth < 1:
            search_depth = 2
        for most_connected_node in most_connected_nodes:
            edited_graph = self.graph.subgraph(current_graph).copy()  # Create a mutable copy of the subgraph
            nodes_to_remove = [node for node in most_connected_nodes if node != most_connected_node]
            edited_graph.remove_nodes_from(nodes_to_remove)

            nodes_to_remove = []
            for node in current_graph:
                if not NxGraphAssistant.connected(edited_graph, most_connected_node, node):
                    nodes_to_remove.append(node)
            edited_graph.remove_nodes_from(nodes_to_remove)

            for current_subgraph in nx.connected_components(edited_graph):
                if self.check_subtree_depth_n(self.graph.subgraph(current_subgraph), depth=search_depth,
                                              mode="remove-all-most-connected-nodes"):
                    # print("Subtree is complete")
                    self.find_complete_subgraphs_in_connected_graph(self.graph, current_subgraph, last_node,
                                                                    problem_solver=Custom_Tree.alex_optimal)
                else:
                    # print("Subtree is not complete")
                    # print("subtree",current_subgraph)
                    pass
        if True:
            hash_map = {}
            # Iterate through the children of the last node
            for most_connected_node in last_node.children:
                # Create a hash for the current node
                node_hash = most_connected_node.create_child_hash()

                # If the hash already exists in the hash map, append the node
                if node_hash in hash_map:
                    hash_map[node_hash].append(most_connected_node)
                # Otherwise, create a new entry in the hash map
                else:
                    hash_map[node_hash] = [most_connected_node]

            # Merge the duplicate nodes
            for key, nodes in hash_map.items():
                # Skip the key if it's 0 or there's only one node
                if key != 0 and len(nodes) > 1:
                    # Concatenate the names of the duplicate nodes
                    name = "".join(str([node.name for node in nodes])).join("+")

                    # Remove the duplicate nodes from the last node's children
                    for node in nodes:
                        last_node.children.remove(node)

                    # Create a new node with the concatenated name
                    new_node = Custom_tree_node(name[:-1])
                    last_node.add_child(new_node)

                    # Set the children of the new node
                    new_node.children = [node.children for node in nodes][0]

        if False:
            hash_map = {}
            for most_connected_node in last_node.children:
                hash = most_connected_node.create_child_hash()
                if hash in hash_map:
                    hash_map[hash].append(most_connected_node)
                else:
                    hash_map[hash] = [most_connected_node]
            for key in hash_map:
                print("key", key)
                for node in hash_map[key]:
                    print(node.name)
            for key in hash_map:
                if key != 0 and len(hash_map[key]) > 1:
                    name = ""
                    childs = []
                    for node in hash_map[key]:
                        last_node.children.remove(node)
                        name += str(node.name) + "+"
                        childs = node.children
                    print("merged together duplicate nodes", name[:-1])
                    new_node = Custom_tree_node(name[:-1])
                    last_node.add_child(new_node)
                    new_node.children = childs
        return

    def alex_optimal_b(self, current_graph, most_connected_nodes, last_node=None):
        search_depth = 15
        if search_depth < 1:
            search_depth = 2
        for most_connected_node in most_connected_nodes:
            edited_graph = self.graph.subgraph(current_graph).copy()  # Create a mutable copy of the subgraph
            nodes_to_remove = [node for node in most_connected_nodes if node != most_connected_node]
            edited_graph.remove_nodes_from(nodes_to_remove)

            nodes_to_remove = []
            for node in current_graph:
                if not NxGraphAssistant.connected(edited_graph, most_connected_node, node):
                    nodes_to_remove.append(node)
            edited_graph.remove_nodes_from(nodes_to_remove)

            for current_subgraph in nx.connected_components(edited_graph):
                if self.check_subtree_depth_n(self.graph.subgraph(current_subgraph), depth=search_depth,
                                              mode="remove-all-most-connected-nodes"):
                    # print("Subtree is complete")
                    self.find_complete_subgraphs_in_connected_graph(self.graph, current_subgraph, last_node,
                                                                    problem_solver=Custom_Tree.alex_optimal_b)
                else:
                    # print("Subtree is not complete")
                    # print("subtree",current_subgraph)
                    pass
        if False:
            hash_map = {}
            # Iterate through the children of the last node
            for most_connected_node in last_node.children:
                # Create a hash for the current node
                node_hash = most_connected_node.create_child_hash()

                # If the hash already exists in the hash map, append the node
                if node_hash in hash_map:
                    hash_map[node_hash].append(most_connected_node)
                # Otherwise, create a new entry in the hash map
                else:
                    hash_map[node_hash] = [most_connected_node]

            # Merge the duplicate nodes
            for key, nodes in hash_map.items():
                # Skip the key if it's 0 or there's only one node
                if key != 0 and len(nodes) > 1:
                    # Concatenate the names of the duplicate nodes
                    name = "".join(str([node.name for node in nodes])).join("+")

                    # Remove the duplicate nodes from the last node's children
                    for node in nodes:
                        last_node.children.remove(node)

                    # Create a new node with the concatenated name
                    new_node = Custom_tree_node(name[:-1])
                    last_node.add_child(new_node)

                    # Set the children of the new node
                    new_node.children = [node.children for node in nodes][0]

        if True:
            hash_map = {}
            for most_connected_node in last_node.children:
                hash = most_connected_node.create_child_hash()
                if hash in hash_map:
                    hash_map[hash].append(most_connected_node)
                else:
                    hash_map[hash] = [most_connected_node]
            for key in hash_map:
                if key != 0 and len(hash_map[key]) > 1:
                    name = ""
                    childs = []
                    for node in hash_map[key]:
                        last_node.children.remove(node)
                        name += str(node.name) + "+"
                        childs = node.children
                    print("merged together duplicate nodes", name[:-1])
                    new_node = Custom_tree_node(name[:-1])
                    last_node.add_child(new_node)
                    new_node.children = childs
        return

    # muss mal schauen ob das funktioniert
    def kill_if_no_in_depth_found(self, current_graph, most_connected_nodes, mode="All", last_node=None):
        depth = 2
        for most_connected_node in most_connected_nodes:
            edited_graph = self.graph.subgraph(current_graph).copy()  # Create a mutable copy of the subgraph
            nodes_to_remove = [node for node in most_connected_nodes if node != most_connected_node]
            edited_graph.remove_nodes_from(nodes_to_remove)

            nodes_to_remove = []
            for node in current_graph:
                if not NxGraphAssistant.connected(edited_graph, most_connected_node, node):
                    nodes_to_remove.append(node)
            edited_graph.remove_nodes_from(nodes_to_remove)
            amount = 0
            for current_subgraph in nx.connected_components(edited_graph):
                if self.check_subtree_depth_n(current_subgraph, depth - 1):
                    amount += 1
            if mode == "All":
                if amount == len(nx.connected_components(edited_graph)):
                    for current_subgraph in nx.connected_components(edited_graph):
                        self.find_complete_subgraphs_in_connected_graph(self.graph, current_subgraph, last_node,
                                                                        problem_solver=Custom_Tree.kill_if_no_in_depth_found)
            if mode == "One":
                if amount != 0:
                    for current_subgraph in nx.connected_components(edited_graph):
                        self.find_complete_subgraphs_in_connected_graph(self.graph, current_subgraph, last_node,
                                                                        problem_solver=Custom_Tree.kill_if_no_in_depth_found)

        return

    def problem_handler_create_multiple_trees_on_conflict(self, current_graph, most_connected_nodes, last_node=None):
        """
        Handles conflicts by creating multiple subtrees for each most connected node and adds them to the tree.

        Args:
        - current_graph: The current graph.
        - most_connected_nodes: The most connected nodes in the current graph.
        - last_node: The last node added to the tree. Defaults to None.

        """
        for most_connected_node in most_connected_nodes:
            edited_graph = self.graph.subgraph(current_graph).copy()  # Create a mutable copy of the subgraph
            nodes_to_remove = [node for node in most_connected_nodes if node != most_connected_node]
            edited_graph.remove_nodes_from(nodes_to_remove)

            nodes_to_remove = []
            for node in current_graph:
                if not NxGraphAssistant.connected(edited_graph, most_connected_node, node):
                    nodes_to_remove.append(node)
            edited_graph.remove_nodes_from(nodes_to_remove)

            for current_subgraph in nx.connected_components(edited_graph):
                self.find_complete_subgraphs_in_connected_graph(self.graph, current_subgraph, last_node,
                                                                problem_solver=Custom_Tree.problem_handler_create_multiple_trees_on_conflict)

    def find_complete_subgraphs_in_connected_graph(self, G, current_graph, last_node, problem_solver):
        """
        Finds complete subgraphs in the connected graph and adds them to the tree.

        Args:
        - G: The graph.
        - current_graph: The current graph.
        - last_node: The last node added to the tree.
        - problem_solver: The problem solver function to handle conflicts.

        """
        self.graph = G
        if NxGraphAssistant.is_complete_graph(G.subgraph(current_graph)):
            name = ""
            for node in current_graph:
                if name != "":
                    name += "+"
                name += str(node)
            if last_node is None:
                last_node = self.add_node(0, name)
            else:
                last_node = self.add_node_efficent(last_node, name)


        else:
            most_connected_node = []
            most_connected_nodes = NxGraphAssistant.all_most_connected_nodes(G, current_graph)
            # this ist the "problematic case"
            if len(most_connected_nodes) > 1:
                if last_node is None:
                    last_node = self.add_node(0, "root")

                # print type of problem solver
                problem_solver(self, current_graph, most_connected_nodes, last_node)
            else:

                most_connected_node = most_connected_nodes[0]

                if last_node is None:
                    last_node = self.add_node(0, most_connected_node)
                else:
                    last_node = self.add_node_efficent(last_node, most_connected_node)
                edited_graph = G.subgraph(current_graph).copy()
                edited_graph.remove_node(most_connected_node)
                for current_subgraph in nx.connected_components(edited_graph):
                    self.find_complete_subgraphs_in_connected_graph(G, current_subgraph, last_node, problem_solver)


class ClusteringHandler():
    def __init__(self):
        pass

    def do_all(self, G, solver):
        trees = []
        for g in nx.connected_components(G):
            tree = Custom_Tree()
            graph = G.subgraph(g)
            tree.find_complete_subgraphs_in_connected_graph(graph, graph, None, solver)
            trees.append(tree)
        return trees

    def do_all_and_print(self, G, solver):
        trees = []
        for g in nx.connected_components(G):
            tree = Custom_Tree()
            graph = G.subgraph(g)
            tree.find_complete_subgraphs_in_connected_graph(graph, graph, None, solver)
            trees.append(tree)
        trees.print(trees)
        return trees

    def print(self, trees):
        # Print all trees
        for tree in trees:
            print("Tree number", trees.index(tree) + 1)
            tree.print_tree()
            print("---")
        return trees



clusterMaster = ClusteringHandler()
#graph = GraphCreator.create_random_graph_with_weights(num_nodes=40, edge_probability=0.3)
#graph = GraphCreator.create_random_graph_with_weights(400,0.5)
#graph = GraphCreator().create_GraphX3()
graph = GraphCreator().get_get_orion()
#graph = GraphCreator().create()
graph_base = GraphCreator().get_get_orion()

import time
# Measure time for the first code snippet





graph2 = NxGraphAssistant.remove_edges(graph_base,similarity='weight',threshold=0.2)
graph2 = NxGraphAssistant.analyze_cliques_new(graph2,0.8)

graph = graph2

# print the size of of both graphs
print("Size of graph1:", len(graph))
print("Size of graph2:", len(graph2))
print(graph)
print(graph2)



# Measure time for the second code snippet
start_time1 = time.time()
T = clusterMaster.do_all(graph,Custom_Tree.alex_optimal_b)
end_time1 = time.time()
start_time2 = time.time()
Z = T
end_time2 = time.time()

# Calculate the elapsed time for each code snippet
elapsed_time1 = end_time1 - start_time1
elapsed_time2 = end_time2 - start_time2

# Print the elapsed time for each code snippet
print("Elapsed time for the first code snippet:", elapsed_time1, "seconds")
print("Elapsed time for the second code snippet:", elapsed_time2, "seconds")

clusterMaster.print(T)
clusterMaster.print(Z)
for tree in T:
    tree.print_tree()
    
import pandas as pd
from plotting_tool_3D import plot

# change to where you want to save your plots
output_path = ""

# change to the location of the directory containing the label data
data_path = r"C:\Users\Alexm\Downloads\3D_plotting\Region_dataframes/"

# Orion is split into 5 regions (numbered 0 - 4)
## Region 2 is the largest (22 groups)
## Regions 0 and 4 are the smallest

regions = [f'Region_{float(i)}_sf_200_grouped_solutions.csv' for i in range(5)]
# pick the region you want to work with
r = 0
region = regions[r]

# read in the dataframe
df_region = pd.read_csv(data_path+region)




prefix = 'cluster_label_group'

# Count columns with the specified prefix
count = sum(1 for col in df_region.columns if col.startswith(prefix))

print(f"There are {count} grouped solutions available for Orion-region: {r}.")

for grouped_sol in range(count):
    labels = df_region.loc[:, prefix+f"_{grouped_sol}"].to_numpy()
    fig = plot(labels=labels, df=df_region, filename=f"test_grouped_sol_{grouped_sol}", output_pathname= output_path, hrd = True, icrs = False, return_fig = True)
    fig.show()