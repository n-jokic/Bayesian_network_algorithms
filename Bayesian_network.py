# -*- coding: utf-8 -*-
"""
Created on Mon May 22 20:14:07 2023

@author: Milosh Yokich
"""
import pandas as pd
import numpy as np

import os
script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
rel_path = 'Bayesian_Network.csv'
abs_file_path = os.path.join(script_dir, rel_path)

# Read the CSV file
data = pd.read_csv(abs_file_path)

# Extract the variable names and domains from the CSV file

nodes = [i for i in data['variable']]
parents = [i for i in data['parents']]
domains = [i for i in data['domain']]



cpds = [i for i in data['cpd']]

network_nodes = {}
for node, parent, domain, cpd in zip(nodes, parents, domains, cpds):
    
    if pd.isna(parent):
        par = []
    else:
        par = parent.split()
        
    cpd = cpd.split()
    cpd = [float(i) for i in cpd]
    num_rows = len(cpd) // (len(domain.split())-1)

    # Reshape the list into a matrix
    probability_matrix = np.array(cpd).reshape(num_rows, len(domain.split()) - 1)
    
    
    
    info = {
        'name': node,
        'domain': domain.split(),
        'parents': par,
        'cpd': probability_matrix
    }
    network_nodes[node] = info

def contruct_lookup_table (bn, node, idx):
    if (idx < len(node['parents'])):
        idx = idx + 1;
        for symbol in bn[node['parents'][idx]]['domain']:
            
            contruct_lookup_table (bn,  node, idx )
    else:
        ... #Bolje je definisati domain kao broj
            
          
          

    
for var, node in network_nodes.items():
      print(f"Variable: {var}")
      print(f"Domain size: {node['domain']}")
      print(f"Parents: {node['parents']}")
      print(f"CPD:\n{node['cpd']}")
      print()  
      
      
import networkx as nx
import matplotlib.pyplot as plt
import network2tikz as n2tkz
import scipy

# Create an empty directed graph
graph = nx.DiGraph()

# Add nodes to the graph
for var, node in network_nodes.items():
    graph.add_node(var)

# Add edges to the graph based on parent-child relationships
for var, node in network_nodes.items():
    parents = node['parents']
    for parent in parents:
        graph.add_edge(parent, var)

# Set the layout for visualizing the graph
layout = nx.planar_layout(graph)

# Draw the graph with node labels and edge labels
plt.figure(figsize=(8, 6))
nx.draw_networkx(graph, pos=layout, with_labels=True, node_size=1000, font_size=12, node_color='lightblue', edge_color='gray')
edge_labels = {}
nx.draw_networkx_edge_labels(graph, pos=layout, edge_labels=edge_labels, font_size=10, rotate=False)
plt.title('Bayesian Network Visualization')
plt.show()

visual_style = {}
vertex_label= [i for i in graph];

n2tkz.plot(graph, filename = 'graph.tex', layout = layout, canvas=(10,6), margin=1,vertex_label = vertex_label)



