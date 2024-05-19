import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import sys 
import os
import random


from ConcensusClustering.consensus import ClusterConsensus
import pandas as pd



def get_orion_data(num):
    '''
    Load the Orion dataset and return the graph that represents the data.
    
    Args:
        num (int): The region of the Orion dataset to load. Must be in the range [0, 4].

    Returns:
        nx.Graph: The graph with merged nodes.
        translation (dict): A dictionary that maps the original node labels to the new node labels.
    '''
    # change to the location of the directory containing the label data
    label_path = r'C:\Users\Alexm\OneDrive - Universität Wien\01_WINF\Praktikum1\SigMA_Alex_modifications\alex_workspace\Grouped_solution_labels\Grouped_solution_labels/'

    # Orion is split into 5 regions (numbered 0 - 4)
    ## Region 2 is the largest (22 groups)
    ## Regions 0 and 4 are the smallest

    regions = [f'Region_{i}/' for i in range(5)]

    # pick the region you want to work with
    r =num

    region = regions[r]
    grouped_labels = pd.read_csv(label_path+region+f'grouped_solutions_chunk_{r}.csv', header=None).to_numpy() # load labels
    print(f"There are {grouped_labels.shape[0]} grouped solutions for region {r}.")
    #density = pd.read_csv(label_path+region+f'Density_chunk_{r}.csv', header=None).to_numpy() # load density (for cc.remove_edges_density)
    #rho =density.reshape(len(density),)

    # create graph
    cc = ClusterConsensus(*grouped_labels)
    translation = cc.labels_bool_dict2arr
    #reanme the similarity in every edge to "weight"
    for i, j, d in cc.G.edges(data=True):
        d['weight'] = d['similarity']
        d['weight_minor'] = d['similarity_minor']
        del d['similarity']
        del d['similarity_minor'] 
    return cc.G, translation





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
            for j in range(i+1, len(nodes)):
                if random.random() < edge_probability:  # Edge probability as parameter
                    G.add_edge(nodes[i], nodes[j])
        
        # Add weights to all edges
        for u, v in G.edges():
            G[u][v]['weight'] = 0.5
        
        return G
    #random graph with weights
    def create_random_graph_with_weights(num_nodes, edge_probability):
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
                        0.7,0.001]
        uv = 0
        for u, v in G.edges():
            G[u][v]['weight'] = predefined_w[uv]
            uv += 1
        return G
    def get_get_orion(self,num=1):
        '''
        Load the Orion dataset and return the graph that represents the data.
        
        Args:
            num (int): The region of the Orion dataset to load. Must be in the range [0, 4].

        Returns:
            nx.Graph: The graph with merged nodes.
            translation (dict): A dictionary that maps the original node labels to the new node labels.
        '''
        return get_orion_data(num)
    