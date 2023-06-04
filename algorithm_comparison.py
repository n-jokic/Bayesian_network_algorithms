# -*- coding: utf-8 -*-
"""
Created on Sun May 28 19:29:12 2023

@author: Milosh Yokich
"""
import Bayesian_network
import numpy as np

Nr = 100000


rel_path = 'Bayesian_Network.csv'
bn = Bayesian_network.Bayesian_Network(rel_path)
N = 1000
e = {'F': 'F+'}
X = 'E'
value = '-'

algorithm_type = ["rejection_sampling", "likelihood_weighting", "gibbs_sampling"]

true_value = bn.inference( X = X, e = e, algorithm = "elimination", N = N)
for key in true_value:
    if X + value in key:
        break
true_value = true_value[key]

results = {}
for algorithm in algorithm_type:
    estimations = np.zeros((Nr, ))
    for i in range(Nr):
        if algorithm == "gibbs_sampling":
            k = 1
        else: 
            if algorithm == "rejection_sampling":
                k = 1.8
            else:
                k = 1
        temp = bn.inference( X = X, e = e, algorithm = algorithm, N = int(N*k), burn_in = 20)
        for key in temp:
            if X + value in key:
                break
       
        estimations[i] = temp[key]
        
    
    results[algorithm] = estimations
    

import numpy as np
import matplotlib.pyplot as plt


for algorithm in algorithm_type:
    plt.rcParams['text.usetex'] = True
    data = results[algorithm]
    x_true =true_value 

    # Compute mean and standard deviation
    mean = np.mean(data)
    std = np.std(data)
    
   # Create subplots
    fig, ax = plt.subplots()
    
    # Plot histogram
    ax.hist(data, bins=30, alpha=0.5, color='blue', edgecolor='black', label='Samples')
    ax.axvline(mean, color='red', linestyle='dashed', linewidth=1.5, label='Mean')
    ax.axvline(x_true, color='green', linestyle='dashed', linewidth=1.5, label='True Value')
    
    # Plot bars for 3 standard deviations
    ax.axvspan(mean - 3 * std, mean + 3 * std, color='orange', alpha=0.2, label='3 Std')
    print((np.abs(mean - x_true))/x_true*100)
    # Set labels and title
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    #ax.set_title('Histogram of : ' + algorithm + ' estmation')
    ax.set_xlim([0.58, 0.71])
    # Add legend
    ax.legend()
    plt.rcParams['text.usetex'] = False
    # Display the plot
    ax.set_rasterized(True)
    FULL_PATH = 'C:/Users/milos/OneDrive/VIII semestar/VI/domaci2/izvestaj/slike/zad1/' + algorithm + '.png'
    #plt.savefig(FULL_PATH, format='png', dpi=300)
    plt.show()
    
    

    # Save the figure in EPS format
    
    
    

