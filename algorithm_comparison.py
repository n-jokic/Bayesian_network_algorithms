# -*- coding: utf-8 -*-
"""
Created on Sun May 28 19:29:12 2023

@author: Milosh Yokich
"""
import Bayesian_network
import numpy as np

Nr = 100


rel_path = 'Bayesian_Network.csv'
bn = Bayesian_network.Bayesian_Network(rel_path)
N = 100000
e = {'F': 'F+'}
X = 'E'
value = '-'

algorithm_type = ["rejection_sampling", "likelihood_weighting", "gibbs_sampling"]

true_value = bn.inference( X = X, e = e, algorithm = "elimination", N = N)[tuple(['F+', 'E-'])]

results = {}
for algorithm in algorithm_type:
    estimations = np.zeros((Nr, ))
    for i in range(Nr):
        #print(bn.inference( X = X, e = e, algorithm = algorithm, N = 1000))
        estimations[i] = bn.inference( X = X, e = e, algorithm = algorithm, N = N)[tuple(['F+', 'E-'])]
    
    results[algorithm] = estimations
    

import numpy as np
import matplotlib.pyplot as plt

for algorithm in algorithm_type:
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
    
    # Set labels and title
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram of : ' + algorithm + ' estmation')
    ax.set_xlim([0.55, 0.75])
    # Add legend
    ax.legend()
    plt.rcParams['text.usetex'] = True
    # Display the plot
    plt.show()
    
    

    # Save the figure in EPS format
    
    FULL_PATH = 'C:]Users/milos/OneDrive/VIII semestar/VI/domaci2/izvestaj/slike/zad1/' + algorithm + '.eps'
    plt.savefig(FULL_PATH, format='eps', dpi=300)
    

