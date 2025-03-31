import numpy  as np
import pandas as pd
import sys
import os
import time 

# import the clustering algorithms

from rskpp.rskpp import rskmeanspp
from rskpp.rskpp import afkmc2
from rskpp.rskpp import compute_cost

# import the dataset 
                                                 
data_file = "data/synth_data.csv"
data_name = "synth_data"

df = pd.read_csv(data_file, header=None)
data = np.array(df)

# Set seed for reproducibility

np.random.seed(42)

# values of k for which runtime and solution quality is to be reported

k_vals = [5,10,20,50,100]

# Number of oterations for which the algorithms are to be repeated

iters = 20

# Write the results of the experiments in the exp-data folder 

with open(f"logs/exp1.{data_name}.csv", 'w') as f : 

    f.write(f"k, rskmeanspp cost, afkmc2 cost, rskmeanspp std, afkmc2 std, rskmeanspp time, afkmc2 time\n")

    for k in k_vals : 
        
        afkmc2_costs = []
        rskmeanspp_costs = []

        afkmc2_times = []
        rskmeanspp_times = []

        for _ in range(iters) : 

            s1 = time.time()
            centers = afkmc2(data,k,200)
            t1 = time.time()
            afkmc2_time = t1 - s1 
            afkmc2_cost = compute_cost(data,centers)

            s1 = time.time()
            centers = rskmeanspp(data,k,10**9)
            t1 = time.time()
            rskmeanspp_time = t1 - s1 
            rskmeanspp_cost = compute_cost(data,centers)

            afkmc2_costs.append(afkmc2_cost)
            rskmeanspp_costs.append(rskmeanspp_cost)

            afkmc2_times.append(afkmc2_time)
            rskmeanspp_times.append(rskmeanspp_time)

        f.write(f"{k}, {np.mean(rskmeanspp_costs)}, {np.mean(afkmc2_costs)}, {np.std(rskmeanspp_costs)}, {np.std(afkmc2_costs)}, {np.mean(rskmeanspp_times)}, {np.mean(afkmc2_times)}\n")


            




