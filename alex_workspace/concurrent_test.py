#%%

#%%
import networkx as nx
import random
import concurrent.futures

class GraphCreator():
    def __init__(self):
        pass
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

    def create_advanced_graph(self):
        G = nx.Graph()
        G.add_nodes_from(['A1', 'A2', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3', 'C4', 'D1', 'D2', 'D3', 'D4', 'D5', 'E1', 'E2', 'E3', 'E4', 'E5'])
        G.add_edges_from([
            ('A1','B1'),('A1','B2'),('A1','C1'),('A1','C2'),('A1','C3'),('A1','D1'),('A1','D2'),('A1','D3'),('A1','D4'),('A1','E1'),('A1','E2'),('A1','E3'),('A1','E4'),
            ('A2','B3'),('A2','C4'),('A2','D5'),('A2','E5'),
            ('B1','C1'),('B1','D1'),('B1','E1'),
            ('B2','C2'),('B2','D2'),('B2','E2'),('B2','C3'),('B2','D3'),('B2','D4'),('B2','E3'),('B2','E4'),
            ('B3','C4'),('B3','D5'),('B3','E5'),
            ('C1','D1'),('C1','E1'),
            ('C2','D3'),('C2','D4'),('C2','E4'),('C2','E2'),
            ('C3','D2'),('C3','E2'),('C3','E3'),
            ('C4','D5'),('C4','E5'),
            ('D1','E1'),
            ('D2','E2'),('D2','E3'),
            ('D3','E2'),
            ('D4','E4'),
            ('D5','E5'),]
    
        )
        # Add random weights to all edges that simulate a pair of jaccardian similarity and jaccardian similiarity 
        for u, v in G.edges():
            G[u][v]['weight'] = round(random.uniform(0.1, 1.0), 3)
    
        return G

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
    def create_random_graph(num_nodes, edge_probability):
        G = nx.Graph()
        nodes = list(range(1, num_nodes + 1))  # Nodes from 1 to num_nodes
        G.add_nodes_from(nodes)
        
        # Add random edges
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                if random.random() < edge_probability:  # Edge probability as parameter
                    G.add_edge(nodes[i], nodes[j])
        
        # Add weights to all edges
        for u, v in G.edges():
            G[u][v]['weight'] = 0.5
        
        return G

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
#%%
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
    def all_most_connected_nodes(entire_graph : nx.Graph ,sub_graph):
        most_connected_nodes = [max(sub_graph, key=lambda x: len(entire_graph[x]))]
        for node in sub_graph:
            if len(entire_graph[node]) == len(entire_graph[most_connected_nodes[0]]) and node != most_connected_nodes[0]:
                most_connected_nodes.append(node)
                
        return most_connected_nodes
    @staticmethod
    def analyze_cliques(G,treshold=0.9):
        # Find cliques with Jaccardian similarity higher than 0.9
        cliques = [clique for clique in nx.find_cliques(G) if all(G[u][v]['weight'] > treshold for u, v in nx.utils.pairwise(clique))]
        
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
            new_node = ''.join(clique)
            # add the new node to the new graph
            new_graph.add_node(new_node)
            # remove the old nodes from the new graph
            new_graph.remove_nodes_from(clique)
            # add edges between the new clique and the old nodes each clique member was connected to
            for node in clique:
                for neighbor in G.neighbors(node):
                    if neighbor not in clique and neighbor in new_graph.nodes():
                        new_graph.add_edge(new_node, neighbor)
        return new_graph
    
    @staticmethod
    def analyze_cliques_undirected(G):
        pair = NxGraphAssistant.analyze_cliques(G)
        graph = pair[0]
        cliques = pair[1]
        
        # convert directed to undirected graph
        undirected_graph = nx.Graph(graph)
        # add all nodes from G
        undirected_graph.add_nodes_from(G.nodes())
        # add all edges from G
        for u, v in G.edges():
            undirected_graph.add_edge(u, v)
        return undirected_graph, cliques
    @staticmethod
    def plot_networkX_graph(G):
        import networkx as nx
        import matplotlib.pyplot as plt

        # Adjust the layout of the graph for better readability
        pos = nx.spring_layout(G, k=0.15)  # You can adjust the value of 'k' for desired spacing

        nx.draw(G, pos, with_labels=True, font_weight='bold')
        plt.show()

#%%

#%%
import networkx as nx
import uuid

class Custom_tree_node:
    def __init__(self, name):
        self.name = name
        self.uuid = uuid.uuid4()
        self.children = []

    def add_child(self, child):
        self.children.append(child)
        return child 



class Custom_Tree:

    def __init__(self):
        self.root = None
        self.graph = nx.Graph()

    def add_node(self, parent_uuid, child_name):
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
        

    def _find_node(self, node, target_uuid):
        if node.uuid == target_uuid:
            return node
        for child in node.children:
            found = self._find_node(child, target_uuid)
            if found:
                return found
        return None

    def print_tree(self, node=None, depth=0):
        if not node:
            node = self.root
        #print(f"{node.name}({node.uuid})")
        print(f"{node.name}")
        if node.children:
            for i, child in enumerate(node.children):
                if i < len(node.children) - 1:
                    print("  " * (depth + 1) + "├── ", end="")
                else:
                    print("  " * (depth + 1) + "└── ", end="")
                self.print_tree(child, depth + 2)


    def find_complete_subgraphs_in_connected_graph(self, G, current_graph, last_node=None):
        graph = G
        if NxGraphAssistant.is_complete_graph(G.subgraph(current_graph)):
            name = ""
            for node in current_graph:
                if name != "":
                    name += "+"
                name += str(node)
            if last_node is None:
                last_node = self.add_node(0, name)
            else:
                last_node = self.add_node(last_node.uuid, name)
            
            print("complete graph", name)
        else:
            most_connected_nodes = NxGraphAssistant.all_most_connected_nodes(graph, current_graph)
            if len(most_connected_nodes) > 1:
                if last_node is None:
                    last_node = self.add_node(0, "root")
                print("case more than one most connected node")
                print("most connected nodes", most_connected_nodes)
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = []
                    for node in most_connected_nodes:
                        futures.append(executor.submit(self.process_most_connected_node, G, current_graph, node, last_node))
                    for future in concurrent.futures.as_completed(futures):
                        pass
            else:
                print("case one most connected node")
                most_connected_node = most_connected_nodes[0]
                print("most connected node", most_connected_node)
                if last_node is None:
                    last_node = self.add_node(0, most_connected_node)
                else:
                    last_node = self.add_node(last_node.uuid, most_connected_node)
                edited_graph = G.subgraph(current_graph).copy()
                edited_graph.remove_node(most_connected_node)
                for current_subgraph in nx.connected_components(edited_graph):
                    self.find_complete_subgraphs_in_connected_graph(G, current_subgraph, last_node)

    def process_most_connected_node(self, G, current_graph, most_connected_node, last_node):
        print("iteration")
        edited_graph = G.subgraph(current_graph).copy()
        for node in NxGraphAssistant.all_most_connected_nodes(G, current_graph):
            if node != most_connected_node:
                edited_graph.remove_node(node)
        for node in current_graph:
            if not NxGraphAssistant.connected(edited_graph, most_connected_node, node):
                try:
                    edited_graph.remove_node(node)
                except:
                    continue
        print("added:", most_connected_node)
        print("to parent", last_node.name)
        for current_subgraph in nx.connected_components(edited_graph):
            print("-last node", last_node.name)
            print("-most connected node", most_connected_node)
            print("-current subgraph", current_subgraph)
            self.find_complete_subgraphs_in_connected_graph(G, current_subgraph, last_node)



#%%
class ClusteringHandler():
    def __init__(self):
            pass
    def do_all(self,G):
        trees = []
        for g in nx.connected_components(G):
            tree = Custom_Tree()
            graph = G.subgraph(g)
            tree.find_complete_subgraphs_in_connected_graph(graph, graph, None)
            trees.append(tree)

        # Print all trees
        for tree in trees:
            tree.print_tree()
            print("---")
        return trees


    
#%%
clusterMaster = ClusteringHandler()
# G= GraphCreator().create_GraphX0()
#G = NxGraphAssistant.analyze_cliques(G)
#T = clusterMaster.do_all(G)
G = GraphCreator.create_random_graph(50, 0.1)
NxGraphAssistant.plot_networkX_graph(G)
T = clusterMaster.do_all(G)
#%%

#%%
