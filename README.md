
# A New Rejection Sampling Approach to $k$-means++


## Installing the Package

```
pip3 install rskpp
```

## Example Usage

```
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from rskpp.rskpp import rskmeanspp

# Generate synthetic dataset with 3 clusters
n_samples = 500
n_features = 2
n_clusters = 3

data, _ = make_blobs(n_samples=n_samples, centers=n_clusters, n_features=n_features, random_state=42)

# Apply rskpp function
k = n_clusters  # Number of clusters
m = 200  # Upper bound on rejection sampling iterations

centers = rskmeanspp(data, k, m)

# Apply KMeans using rskmeanspp centers as initialization
kmeans = KMeans(n_clusters=k, init=centers, n_init=1, random_state=42)
kmeans.fit(data)

# Plot dataset and final cluster centers
plt.scatter(data[:, 0], data[:, 1], s=10, label="Data points")
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x', s=100, label="Final centers")

plt.legend()
plt.title("K-Means with RS-k-means++ Initialization")
plt.savefig("cluster-plot.png")

```

## Running Tests

1. Download this repository
2. Setup Virtual environment 
  ```
  > /usr/bin/python3 -m venv .venv
  > source .venv/bin/activate
  > pip3 install -r requirements.txt
  ```
3. Place the dataset in the `data` folder in csv format. Name it as `dataname.csv`
4. The experiments can be run by running the corresponding python files : 
  ```
  > python3 exp/exp0.py dataname
  > python3 exp/exp1.py dataname
  > python3 exp/exp1.py dataname
  ```
5. The results of the experiments appear in the logs folder

There is an example dataset called `synth_data` which can be used to run the code.