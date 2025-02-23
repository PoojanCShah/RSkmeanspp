A Rejection Sampling Approach to k-means++
===========================================



This project contains an implementation of the rejection sampling based RS-k-means++ algorithm introduced in [SAJ25]_  for performing the k-means++ seeding [AV07]_ . The implementation is compatible with NumPy and scikit-learn. 

Installation
------------

This package can be directly installed from PyPI using

.. code-block:: python 

   pip install rskpp


Algorithm Description 
----------------------



Algorithm Reference 
---------------------


.. code-block:: python  

   rskmeanspp(data: np.array, k: int, m : int) -> np.array:

**Parameters:**

- **data** (`numpy.array`): Dataset of shape (n, d), where `n` is the number of data points and `d` is the number of features.
- **k** (`int`): The number of clusters.
- **m** (`int`): The upper bound on the number of rejection sampling iterations.

**Returns:**

- (`numpy.array`): A NumPy array of shape (k, d) containing the `k` cluster centers.



Example Usage 
--------------

The ``rskmeanspp`` function performs the seeding, which can used in conjunction with the k-means algorithm (also commonly known as Lloyd's iterations) from scikit-learn. 


.. code-block:: python  

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
   plt.show()




License 
--------

The project is licensed under the MIT License.


Support
--------

Reach me at : https://poojancshah.github.io/

Source Code : https://github.com/PoojanCShah/RSkmeanspp


.. [SAJ25] Poojan Shah, Shashwat Agrawal, and Ragesh Jaiswal. A New Rejection Sampling Approach to k-means++ With Improved Trade-Offs. 2025. arXiv: 2502.02085 [cs.DS]. url: https://arxiv.org/abs/2502.02085.

.. [AV07] David Arthur and Sergei Vassilvitskii. “k-means++: the advantages of careful seeding”.In: Proceedings of the Eighteenth Annual ACM-SIAM Symposium on Discrete Algorithms.SODA ’07. New Orleans, Louisiana: Society for Industrial and Applied Mathematics, 2007,pp. 1027–1035. isbn: 9780898716245. url: https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf.






.. toctree::
   :maxdepth: 2

   description
   reference 
   example
   experiments