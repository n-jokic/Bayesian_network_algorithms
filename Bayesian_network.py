# -*- coding: utf-8 -*-
"""
Created on Mon May 22 20:14:07 2023

@author: Milosh Yokich
"""
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import network2tikz as n2tkz
import scipy

import os



class Bayesian_Network(object):
    
    
    def __init__(self, relative_path):
        script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
        abs_file_path = os.path.join(script_dir, rel_path)
        data = pd.read_csv(abs_file_path)
        
        nodes = [i for i in data['variable']]
        parents = [i for i in data['parents']]
        domains = [i for i in data['domain']]
        cpds = [i for i in data['cpd']]
        
        network = self.__form_bayesian_network(nodes, parents, domains, cpds)
        self.network = network
        self.__current_index = 0
        self.__keys = list(self.network.keys())
        
        self.__format_cpds()
        
    def draw_network(self):
        # Create an empty directed graph
        graph = nx.DiGraph()

        # Add nodes to the graph
        for var, node in self.network.items():
            graph.add_node(var)

        # Add edges to the graph based on parent-child relationships
        for var, node in self.network.items():
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
        
        vertex_label= [i for i in graph];
        n2tkz.plot(graph, filename = 'graph.tex', layout = layout, canvas=(10,6), margin=1,vertex_label = vertex_label)
        
        
    def __str__(self):
        
        for var, node in self.network.items():
              print(f"Variable: {var}")
              print(f"Domain: {node['domain']}")
              print(f"Parents: {node['parents']}")
              print(f"CPD:\n{node['cpd']}")
              print()  
        
        return ""
        
    def __iter__(self):
        return self
    
    def __next__(self):
        
        if self.__current_index < len(self.network):
            
            self.__current_index += 1
            return self.network[self.__keys[self.__current_index - 1]]
        else:
            self.__current_index = 0
            raise StopIteration
            
        
    def __form_bayesian_network(self, nodes, parents, domains, cpds):
        
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

        return network_nodes
    
    def __format_cpds(self):
        for node in self:
            self.__contruct_lookup_table(node,-1, [], 0, {})
    
    def __contruct_lookup_table (self, node, idx, iterators, row, lookup_table):
        bn = self.network
        if (idx + 1 < len(node['parents'])):
            idx = idx + 1;
            if idx == 0:
                iterators = [0]*len(node['parents'])
                
            for symbol in bn[node['parents'][idx]]['domain']:
                iterators[idx] = symbol
                lookup_table, row = self.__contruct_lookup_table(node, idx, iterators, row, lookup_table)
                
           
                
            
            if idx == 0:
                self.network[node['name']]['cpd'] = lookup_table
            else:     
                return lookup_table, row
        else:
            if (not len(node['parents']) == 0):
                current_row = node['cpd'][row, :] 
                lookup_table[tuple(iterators)] = {}
            
                for idx, item in enumerate(current_row ):
                    key =node['domain'][idx]
                    lookup_table[tuple(iterators)][key] = item
            
            
                row = row+1;
                
                return lookup_table, row
            else:
                row = 0
                current_row = node['cpd'][row, :]
                lookup_table[tuple([])] = {}
                for idx, item in enumerate(current_row ):
                    key =node['domain'][idx]
                    lookup_table[tuple([])][key] = item
                self.network[node['name']]['cpd'] = lookup_table
        
    def getNetwork(self):
        return self.network
    def setNode(self, node, val):
        self.network[node] = val
    def getNode(self, node):
        return self.network[node]
    
    

    
rel_path = 'Bayesian_Network.csv'

bn = Bayesian_Network(rel_path)
print(bn)
for i in bn:
    print(i)



   
          


