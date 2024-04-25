from ConcensusClustering.consensus import ClusterConsensus

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
    def analyze_cliques(G,treshold=0.9):
        # Find cliques with Jaccardian similarity higher than 0.9

        cliques = [clique for clique in nx.find_cliques(G) if all(G.has_edge(u, v) and G[u][v]['weight'] > treshold for u, v in nx.utils.pairwise(clique))]


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
        print ("Avg Jaccard Values:")
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
                        #TODO also add weight (average)
                        new_graph.add_edge(new_node, neighbor)
                        #print("Added edge between", new_node, "and", neighbor)
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
    def remove_edges_with_minor(G,similarity='weight',treshhold=0.3,treshhold_minor = 0.8):
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

