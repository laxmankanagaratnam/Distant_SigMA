

import networkx as nx


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
    
    def connected(graph,node1, node2):
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
    def all_most_connected_nodes(entire_graph : nx.Graph ,sub_graph) :
        try:
            most_connected_nodes = [max(sub_graph, key=lambda x: len(entire_graph[x]))]
            for node in sub_graph:
                if len(entire_graph[node]) == len(entire_graph[most_connected_nodes[0]]) and node != most_connected_nodes[0]:
                    most_connected_nodes.append(node)
                    
            return most_connected_nodes
        except:
            return []

    
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
        """
        Removes edges from a graph based on a similarity threshold.

        Args:
            graph (nx.Graph): The graph from which to remove edges.
            similarity (str): The edge attribute key used for comparing edge values. Defaults to 'weight'.
            threshold (float): The threshold below which edges are removed. Defaults to 0.5.

        Returns:
            nx.Graph: A new graph instance with the specified edges removed.
        """
        G_copy = nx.Graph(graph)
        # --- test if we need to remove edges
        e2js = {frozenset({e1, e2}): G_copy[e1][e2][similarity] for e1, e2 in G_copy.edges}
        # --- remove edges with unmatching cluster solutions
        re = [(e1, e2) for (e1, e2), sim in e2js.items() if sim < threshold]
        G_copy.remove_edges_from(re)
        return G_copy
    @staticmethod
    def remove_edges_with_minor(G,similarity='weight',treshhold=0.3,treshhold_minor = 0.8):
        """
        Removes edges from a graph based on a primary and a secondary similarity threshold, which is the minor similarity. only if both thresholds are not met the edge is removed.

        Args:
            G (nx.Graph): The graph from which to remove edges.
            similarity (str): The edge attribute key used for the primary comparison. Defaults to 'weight'.
            threshold (float): The primary threshold below which an edge is considered for removal. Defaults to 0.3.
            threshold_minor (float): The secondary threshold; an edge is removed if it fails both primary and secondary checks. Defaults to 0.8.

        Returns:
            nx.Graph: A new graph instance with the specified edges removed based on both primary and secondary conditions.
        """
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
        """
        Calculates a modified Jaccard similarity focusing on the overlap between two sets.

        Args:
            i, j (int): Indexes for the two label sets to compare.
            labels_bool_dict2arr (dict): A dictionary mapping indices to label arrays.

        Returns:
            float: The calculated Jaccard similarity based on the minimum overlap.
        """
        nb_i = labels_bool_dict2arr[i].sum()
        nb_j = labels_bool_dict2arr[j].sum()
        intersection = np.sum(
            (labels_bool_dict2arr[i].astype(int) + labels_bool_dict2arr[j].astype(int)) == 2)
        return intersection / min(nb_i, nb_j)
    
    def analyze_cliques(G, threshold=0.7, sim='weight'):
        """
        Analyzes and transforms graph cliques based on a similarity threshold, merging nodes into new cliques.
        Goes so long until there are no cliques left that meet the similarity threshold.
        These new cliques are then connected based on their original connections.
        These cliques are then put into a new graph and returned as the result.

        Args:
            G (nx.Graph): The graph to analyze.
            threshold (float): The similarity threshold for clique formation and node merging. Defaults to 0.5.
            sim (str): The attribute key used for similarity comparison. Defaults to 'weight'.

        Returns:
            nx.Graph: A new graph where cliques are merged into single nodes and reconnected based on their original connections.
        """
        graph = G.copy()
        while True:
            #print("iteration")
            cliques = [clique for clique in nx.find_cliques(graph) if all(graph.has_edge(u, v) and graph[u][v][sim] > threshold for u, v in nx.utils.pairwise(clique))]
            if not cliques:
                #print("break")
                break

            # Calculate average Jaccardian value for each clique
            avg_jaccard_values = {}
            for clique in cliques:
                total_jaccard_value = sum(graph[u][v][sim] for u, v in nx.utils.pairwise(clique))
                avg_jaccard_value = total_jaccard_value / len(clique)
                avg_jaccard_values[tuple(clique)] = avg_jaccard_value

            # Sort cliques by average Jaccardian value in descending order
            sorted_cliques = sorted(avg_jaccard_values.keys(), key=lambda x: avg_jaccard_values[x], reverse=True)
            #remove all cliques with to low Jaccardian value
            sorted_cliques = [clique for clique in sorted_cliques if avg_jaccard_values[clique] > threshold]
            if not sorted_cliques:
                #print("break")
                break
           # print("Sorted Cliques:")
            #print(sorted_cliques)
            #print("Avg Jaccard Values:")
            #print(avg_jaccard_values)

            # Keep track of assigned nodes
            assigned_nodes = set()#
			# Remove all cliques that have nodes that are already assigned
            selected_cliques = []
            for clique in sorted_cliques:
                if not assigned_nodes.intersection(clique):
                    selected_cliques.append(clique)
                    assigned_nodes.update(clique)
            #print("Selected Cliques:")
            #print(selected_cliques)

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

