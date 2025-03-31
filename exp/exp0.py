import numpy  as np
import pandas as pd
import sys
import os
import time 

# import the clustering algorithms

from rskpp.rskpp import rskmeanspp
from rskpp.rskpp import compute_cost

# import the dataset 
                                                 

data_name = sys.argv[1]
data_file = "data/" + data_name+ ".csv"

df = pd.read_csv(data_file, header=None)
data = np.array(df)

# Set seed for reproducibility

np.random.seed(42)

# values of k for which runtime and solution quality is to be reported

k_vals = [5,10,20,30,40]

# Number of oterations for which the algorithms are to be repeated

iters = 20

# Write the results of the experiments in the exp-data folder 

with open(f"logs/exp0.{data_name}.csv", 'w') as f : 

    f.write(f"k, beta_k\n")

    for k in k_vals : 
        
        rskmeanspp_costs = []


        for _ in range(iters) : 

            centers = rskmeanspp(data,k,10**9)
            rskmeanspp_cost = compute_cost(data,centers)
            rskmeanspp_costs.append(rskmeanspp_cost)

        f.write(f"{k}, {np.var(data) / np.mean(rskmeanspp_costs)}\n")




            




