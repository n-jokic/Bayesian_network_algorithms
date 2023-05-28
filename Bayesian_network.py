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
import random

import os
import itertools
from collections import deque

class ContinueI(Exception):
    pass


continue_i = ContinueI()


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
        self.__visited = {}
        for key in self.__keys:
            self.__visited[key] = False
        self.__form_children()
        self.__find_root_vars()
        self.__sample_value = {}
        for key in self.__keys:
            self.__sample_value[key] = -1
        self.__format_cpds()
        
    def __len__(self):
        return len(self.__keys)
    
    def __find_root_vars(self):
        root = []
        for key in self.__keys:
            if len(self.network[key]['parents'])== 0:
                root.append(key)
        self.__root = root
        return
        
        
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
        temp = ""
        for var, node in self.network.items():
                
              temp += f"Variable: {var}\n"
              temp += f"Domain: {node['domain']}\n"
              temp += f"Parents: {node['parents']}\n"
              temp += f"Children: {node['children']}\n"
              temp += f"CPD:\n{node['cpd']}\n"
              temp += "\n"
        return temp[0:-1]
        
    def __iter__(self):
        return self
    
    def __next__(self):
        
        if self.__current_index < len(self.network):
            
            self.__current_index += 1
            return self.network[self.__keys[self.__current_index - 1]]
        else:
            self.__current_index = 0
            raise StopIteration
            
        
    def __form_children(self):
        for _, key1 in enumerate(self.network):
            children = []
            for _,key2 in enumerate(self.network):
                
                if key1 in self.network[key2]['parents']:
                    children.append(key2)
            self.network[key1]['children'] = children
            
            
            
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
                'cpd': probability_matrix,
                'children' : []
            }
            network_nodes[node] = info

        return network_nodes
    
    def __format_cpds(self):
        for node in self:
            self.__contruct_lookup_table(node,-1, [0]*(len(node['parents'])+1), 0, {})
    
    def __contruct_lookup_table (self, node, idx, iterators, row, lookup_table):
        bn = self.network
        if (idx + 1 < len(node['parents'])):
            idx = idx + 1;
            
                
            for symbol in bn[node['parents'][idx]]['domain']:
                iterators[idx] = node['parents'][idx] + symbol
                lookup_table, row = self.__contruct_lookup_table(node, idx, iterators, row, lookup_table)
                
           
                
            
            if idx == 0:
                self.network[node['name']]['cpd'] = lookup_table
            else:     
                return lookup_table, row
        else:
            if (not len(node['parents']) == 0):
                current_row = node['cpd'][row, :] 
                total = 0
                for idx, item in enumerate(current_row ):
                    key = node['domain'][idx]
                    iterators[-1] = node['name'] + key 
                    lookup_table[tuple(iterators)]= round(item, 3)
                    total+=item
                    
                key =node['domain'][idx + 1]
                iterators[-1] = node['name'] + key
                lookup_table[tuple(iterators)]= round(1-total, 3)
                row = row+1;
                
                return lookup_table, row
            else:
                row = 0
                current_row = node['cpd'][row, :]
                total = 0
                for idx, item in enumerate(current_row):
                    key =node['domain'][idx]
                    iterators[-1] = node['name'] + key
                    lookup_table[tuple(iterators)]= round(item, 3)
                    total+=item
                    
                key =node['domain'][idx + 1]
                iterators[-1] = node['name'] + key
                lookup_table[tuple(iterators)]= round(1-total, 3)
                self.network[node['name']]['cpd'] = lookup_table
        
    def getNetwork(self):
        return self.network
    def setNode(self, node, val):
        self.network[node] = val
    def getNode(self, node):
        return self.network[node]
    
    def inference(self, **kwargs):
        algorithm = kwargs['algorithm']
        if algorithm == "elimination":
            return self.__elimination(kwargs['X'], kwargs['e'])
        if algorithm == "rejection_sampling":
            return self.__rejection_sampling(kwargs['X'], kwargs['e'], kwargs['N'])
        if algorithm == "likelihood_weighting":
            return self.__likelihood_weighting(kwargs['X'], kwargs['e'], kwargs['N'])
        if algorithm == "gibbs_sampling":
            return self.__gibbs_sampling(kwargs['X'], kwargs['e'], kwargs['N'])
            
        
    def __factoring_complexity(self, V):
        complexity = len(self.network[V]['domain'])
        
        for i in self.network[V]['parents']:
            complexity*=len(self.network[i]['domain'])
            
        for i in self.network[V]['children']:
            complexity*=len(self.network[i]['domain'])
            
        return complexity
        
        
    def __elimination_ordering(self, variables):
        
        variables.sort(key = lambda x : self.__factoring_complexity(x))
           
        return variables
    
            
    def __update_parent_table(self, V, e):
        parent_table = dict(self.network[V]['cpd'])
        positions = {}
        if self.__visited[V] == True:
            return {}
        
        
        set_parents = set(self.network[V]['parents'])  # Convert list b to a set for faster membership checking
        for i, item in enumerate(e.keys()):
            if item in set_parents:
                positions[item] = self.network[V]['parents'].index(item)
                
        for key in positions:
            for var in tuple(parent_table):
                if e[key] != var[positions[key]]:
                    del parent_table[var]
        
        if V in e:
            for var in tuple(parent_table):
                if e[V] != var[-1]:
                    del parent_table[var]
        temp = list(self.network[V]['parents'])
        if not temp:
            temp = []
        temp.append(V)
        parent_table['vars'] = temp
                    
        return parent_table        
  
    def __update_children_table(self, V, e):
        children_table = []
        
        for i in self.network[V]['children']:
            ch = i
            if self.__visited[ch] == True:
                continue
            
            
            child_table = dict(self.network[i]['cpd'])
            positions = {}
        
            set_parents = set(self.network[i]['parents'])  # Convert list b to a set for faster membership checking
            for i, item in enumerate(e.keys()):
                if item in set_parents:
                    positions[item] = self.network[ch]['parents'].index(item)
            if ch in e:
                if ch in self.network['parents']:
                    positions[ch] = self.network['parents'].index(ch)
                else:
                    positions[ch] = - 1
                
            for key in positions:
                for var in tuple(child_table):
                    if e[key] != var[positions[key]]:
                        del child_table[var]
                        
            temp = list(self.network[ch]['parents'])
            if not temp:
                temp = []
            temp.append(ch)
            child_table['vars'] = temp
            children_table.append(child_table)
            self.__visited[ch] = True
                    
        return children_table 

    
    def __merge_tables(self, parent_table, children_table, e):
        
        merged_table = parent_table
        if not children_table:
            return merged_table
        
        
        for child_table in children_table:
            
            set_merged = set(merged_table['vars']) 
            set_child = set(child_table['vars'])
            
            new_vars = [i for i in set_merged.union(set_child)]
            
            locations_merged = {}
            locations_child = {}
            
            for var in new_vars:
                
                if var in set_merged:
                    locations_merged[var] = merged_table['vars'].index(var)
                if var in set_child:
                    locations_child[var] = child_table['vars'].index(var)
                    
            to_iter = [[var + value for value in self.network[var]['domain']] for var in new_vars]
                
            merged_key = [0]*len(set_merged)
            child_key = [0]*len(set_child)
                
            new_merged = {}
            for new_key in itertools.product(*to_iter):
                try:
                    for field in new_key:
                        if field[0] in set_merged:
                            merged_key[locations_merged[field[0]]] = field
                        if field[0] in set_child:
                            child_key[locations_child[field[0]]] = field
                        if field[0] in e:
                            if field != e[field[0]]:
                                raise continue_i
                            
                    new_merged[new_key] = merged_table[tuple(merged_key)]*child_table[tuple(child_key)]
                except ContinueI:
                    continue

                
            new_merged['vars'] = new_vars
            merged_table = new_merged
            
        
        return merged_table             
                    
    
    def __make_factor(self, V, e):
        
        parent_table = self.__update_parent_table(V, e)
        children_table = self.__update_children_table(V, e)
        
        return self.__merge_tables(parent_table, children_table, e)

    
    def __sum_out(self, factors, V, e):
        
        to_be_merged = []
        others = []
        for factor in factors:
            if V in factor['vars']:
                to_be_merged.append(factor)
            else: 
                others.append(factor)
        if len(to_be_merged) > 1:
            merged = self.__merge_tables(to_be_merged[0], to_be_merged[1:], e)
        else:
            merged = to_be_merged[0]
            
        new_vars = set(merged['vars']) - set(V)
        
        locations = {}
        for var in new_vars:
            locations[var] = merged['vars'].index(var)
        locations[V] = merged['vars'].index(V)
        
        to_iter = [[var + value for value in self.network[var]['domain']] for var in new_vars]
        
        new_merged = {}
        
        key = [0]*(len(new_vars) + 1)
        for value_V in self.network[V]['domain']:
            for new_key in itertools.product(*to_iter):
                try:
                    for field in new_key:
                        key[locations[field[0]]] = field
                        if field[0] in e:
                            if field!= e[field[0]]:
                                raise continue_i
                    key[locations[V]] = V + value_V   
                    
                    
                    
                    if new_key in new_merged:
                        new_merged[new_key] += merged[tuple(key)]
                    else:
                        new_merged[new_key] = merged[tuple(key)]
                except ContinueI:
                    continue
                        
               
            
        new_merged['vars'] = list(new_vars)
        others.append(new_merged)
        return others
    
    def __normal(self, final_table):
        total_prob = 0
        for key in final_table:
            if key == 'vars':
                continue
            total_prob += final_table[key]
            
        for key in final_table:
            if key == 'vars':
                continue
            final_table[key] = round(final_table[key]/total_prob, 5)
            
            
        return final_table
    
    def __pointwise_product(self, factors, e):
        if len(factors) > 1:
            product = self.__merge_tables(factors[0], factors[1:], e)
        else:
            product = factors[0]
            
        return product
    
    def __elimination(self, X, e):
        variables = list(self.network.keys())
        
        hidden_var = set([i for i in variables if i not in e.keys() and i!=X])
        factors = []
        ordering = self.__elimination_ordering(variables)
        ordering = ['G', 'C', 'F', 'D', 'B', 'A']
        
        for V in ordering:
            #print(V)
            factor = self.__make_factor(V, e)
            if factor:
                factors.append(factor)
            if V in hidden_var:
                factors = self.__sum_out(factors, V, e)
            #print(factors)
            self.__visited[V] = True
            #print(self.__visited)
            
                
        #print(variables)
        #self.__visited = {}
        for key in self.__keys:
            self.__visited[key] = False
        #print(factors)
        self.__pointwise_product(factors, e)
        final_table = self.__normal(self.__pointwise_product(factors, e))
        del final_table['vars']
        return final_table
    
    def __rejection_sampling(self, X, e, N):
        C = {}
        to_iter = [[var + value for value in self.network[var]['domain']] for var in self.network]
        
        for Xs in itertools.product(*to_iter):
            C[tuple(Xs)] = 0
        
        key = [*C][0]
        positions_dic = {}
        
        for idx, field in enumerate(key):
            for name in self.__keys:
                if name == field[0]:
                    positions_dic[name] = idx
            
        key = [0]*len(key)
        
        for j in range(N):
            x = self.__sample()
            if self.__is_consistent(x, e):
                
                for var in x:
                    key[positions_dic[var]] = x[var]
                
                C[tuple(key)] += 1
  
    
        return self.__normalize_sample(C, X, e)
    
    def __sample(self):
        to_be_sampled = deque(self.__root)
        
        while to_be_sampled:
            try:
                x = to_be_sampled.popleft()
                
                current_table = self.network[x]['cpd']
                key = [*current_table][0]
                new_key = [0]*len(key)
                
                for idx, field in enumerate(key):
                    if field[0] == x:
                        continue
                    if not self.__visited[field[0]]:
                        to_be_sampled.append(x)
                        raise continue_i
                    else:
                        new_key[idx] = self.__sample_value[field[0]]
                
                prob = np.random.rand()
                prob_dic = {}
                
                
                for value in self.network[x]['domain']:
                    new_key[-1] = x + value
                    prob_dic[tuple(new_key)] = current_table[tuple(new_key)]
                    
                prob_keys = [*prob_dic]
                prob_keys.sort(key = lambda x : prob_dic[x])
                cdf = 0
                for key in prob_keys:
                    cdf+=prob_dic[key]
                    if prob < cdf:
                        self.__sample_value[x] =key[-1]
                        break

                self.__visited[x] = True #Sample generated
                
                for child in self.network[x]['children']:
                    if not self.__visited[child]:
                        to_be_sampled.append(child)
               
            except ContinueI:
                print( self.__visited)
                continue
            
            
        for key in self.__keys:
            self.__visited[key] = False
                
        return self.__sample_value
    def __is_consistent(self, x, e):
        for var in e:
            if x[var] != e[var]:
                return False
        return True
    def __normalize_sample(self, sample, X, e):
        
        key = [*sample][0]
        x_position = 0;
        e_position = [0]*len(e)
        
        for idx, item in enumerate(key):
            if item[0] == X:
                x_position = idx
            for idy, i in enumerate(e):
                if item[0] == i:
                    e_position[idy] = idx
        
        final_dict = {}
        e_val = [e[i] for i in e]
        for value in self.network[X]['domain']:
            
            final_dict[tuple([*e_val, X + value])] = 0
            
        e_keys = [*e]
        new_key = [0]*(len(e_position)+1)
        for key in sample:
            try:
                for idy, idx in enumerate(e_position):
                    new_key[idy] = key[idx]
                    if key[idx] != e[e_keys[idy]]:
                        raise continue_i
                new_key[-1] = key[x_position]
                
                final_dict[tuple(new_key)] += sample[key]
            except ContinueI:
                continue
            
        total_prob = 0
        for key in final_dict:
            total_prob+=final_dict[key]
        
        if total_prob != 0:
            for key in final_dict:
                final_dict[key] = round(final_dict[key]/total_prob, 5)
            
        return final_dict
    
    def __likelihood_weighting(self, X, e, N):
        
        W = {}
        to_iter = [[var + value for value in self.network[var]['domain']] for var in self.network]
        
        for Xs in itertools.product(*to_iter):
            W[tuple(Xs)] = 0
        
        key = [*W][0]
        positions_dic = {}
        
        for idx, field in enumerate(key):
            for name in self.__keys:
                if name == field[0]:
                    positions_dic[name] = idx
            
        key = [0]*len(key)
        
        for j in range(N):
            x, w = self.__weighted_sample(e)
            
            for var in x:
                key[positions_dic[var]] = x[var]
            
            W[tuple(key)] += w
            
        return self.__normalize_sample(W, X, e)
        
    def __weighted_sample(self, e):
        to_be_sampled = deque(self.__root)
        w = 1
        while to_be_sampled:
            try:
                x = to_be_sampled.popleft()
                    
                current_table = self.network[x]['cpd']
                key = [*current_table][0]
                new_key = [0]*len(key)
                
                for idx, field in enumerate(key):
                    if field[0] == x:
                        continue
                    if not self.__visited[field[0]]:
                        to_be_sampled.append(x)
                        raise continue_i
                    else:
                        new_key[idx] = self.__sample_value[field[0]]
                        
                if x in e:
                    new_key[-1] = e[x]
                    self.__visited[x] = True
                    self.__sample_value[x] = e[x]
                    w = w*current_table[tuple(new_key)]
                    raise continue_i
                
                prob = np.random.rand()
                prob_dic = {}
                
                
                for value in self.network[x]['domain']:
                    new_key[-1] = x + value
                    prob_dic[tuple(new_key)] = current_table[tuple(new_key)]
                    
                prob_keys = [*prob_dic]
                prob_keys.sort(key = lambda x : prob_dic[x])
                cdf = 0
                for key in prob_keys:
                    cdf+=prob_dic[key]
                    if prob < cdf:
                        self.__sample_value[x] =key[-1]
                        break

                self.__visited[x] = True #Sample generated
                
                for child in self.network[x]['children']:
                    if not self.__visited[child]:
                        to_be_sampled.append(child)
               
            except ContinueI:
            
                continue
            
        for key in self.__keys:
            self.__visited[key] = False
                
        return self.__sample_value, w
    
    def __gibbs_sampling(self, X, e, N):
        C = {}
        to_iter = [[var + value for value in self.network[var]['domain']] for var in self.network]
        
        for Xs in itertools.product(*to_iter):
            C[tuple(Xs)] = 0
        
        key = [*C][0]
        positions_dic = {}
        
        for idx, field in enumerate(key):
            for name in self.__keys:
                if name == field[0]:
                    positions_dic[name] = idx
            
        key = [0]*len(key)
        
        Z = []
        for var in self.__keys:
            if var not in e:
                idx = int(np.floor(np.random.rand()*len(self.network[var]['domain'])))
                value = var + self.network[var]['domain'][idx]
                self.__sample_value[var] = value
                Z.append(var)
            else:
                self.__sample_value[var] = e[var]
                
                
        for k in range(N):
            z = random.choice(Z)
            self.__sample_value[z] = self.__markov_blanket_sample(z)
            
            for var in self.__sample_value:
                key[positions_dic[var]] = self.__sample_value[var]
            
            C[tuple(key)] += 1
            
            


            

        return self.__normalize_sample(C, X, e)
    
    def __markov_blanket_sample(self, z):
        
        chilren_table = []
        
        current_table = self.network[z]['cpd']
        key = [*current_table][0]
        new_key = [0]*len(key)
        
        for idx,field in enumerate(key):
            new_key[idx] = self.__sample_value[field[0]]
            
        prob_dic = {}
        for value in self.network[z]['domain']:
            new_key[-1] = z + value
            prob_dic[tuple([new_key[-1]])] = current_table[tuple(new_key)]
        
        for child in self.network[z]['children']:
            child_table = self.network[child]['cpd']
            key = [*child_table][0]
            new_key = [0]*len(key)
            
            pos_z = 0
            for idx,field in enumerate(key):
                new_key[idx] = self.__sample_value[field[0]]
                if field[0] == z:
                    pos_z = idx
                    
            child_prob_dic = {}
            for value in self.network[z]['domain']:
                new_key[pos_z] = z + value
                child_prob_dic[tuple([new_key[pos_z]])] = child_table[tuple(new_key)]
            child_prob_dic['vars'] = z
            chilren_table.append(child_prob_dic)
            
        prob_dic['vars'] = z
        merged = self.__merge_tables(prob_dic, chilren_table, {})    
        del merged['vars']
        prob = np.random.rand()
        prob_keys = [*merged]
        prob_keys.sort(key = lambda x : merged[x])
        cdf = 0
        for key in prob_keys:
            cdf+=prob_dic[key]
            if prob < cdf:
                self.__sample_value[z] = key[-1]
                break
        return self.__sample_value[z]
        

    

    
rel_path = 'Bayesian_Network.csv'

bn = Bayesian_Network(rel_path)
print(bn)

print(bn.inference( X = 'E', e = {'F': 'F+'}, algorithm = "gibbs_sampling", N = 1000))



   
          


