#What will the location of the cluster centers be after the k-means algorithm has converged?

import numpy as np

# Data
x = np.array([1.0, 1.2, 1.8, 2.3, 2.6, 3.4, 4.0, 4.1, 4.2, 4.6])

# Initial centroids
centroids = np.array([1.8, 3.3, 3.6])

def k_means(x, centroids, max_iters=10):
    for _ in range(max_iters):
        # Assign clusters
        clusters = np.argmin(np.abs(x[:, np.newaxis] - centroids), axis=1)

        # Calculate new centroids
        new_centroids = np.array([x[clusters == k].mean() for k in range(len(centroids))])

        # Check for convergence (if centroids do not change)
        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids
    
    return centroids, clusters

# Run K-means
final_centroids, clusters = k_means(x, centroids)

print("Final centroids:", final_centroids)
print("Cluster assignment:", clusters)
