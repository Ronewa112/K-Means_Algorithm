
import numpy as np
import sys

# (1) Hardcoded dataset
data_points = np.array([
    [0.22, 0.33],
    [0.45, 0.76],
    [0.73, 0.39],
    [0.25, 0.35],
    [0.51, 0.69],
    [0.69, 0.42],
    [0.41, 0.49],
    [0.15, 0.29],
    [0.81, 0.32],
    [0.50, 0.88],
    [0.23, 0.31],
    [0.77, 0.30],
    [0.56, 0.75],
    [0.11, 0.38],
    [0.81, 0.33],
    [0.59, 0.77],
    [0.10, 0.89],
    [0.55, 0.09],
    [0.75, 0.35],
    [0.44, 0.55]
])

# (2) Read 6 float inputs for initial centers (from stdin)
initial_centers = [float(input()) for _ in range(6)]
cluster_centers = np.array(initial_centers).reshape(3, 2)

# (3) K-Means algorithm loop
while True:
    # Assign points to the closest cluster center
    clusters = [[] for _ in range(len(cluster_centers))]
    for point in data_points:
        distances = [np.sum((point - center) ** 2) for center in cluster_centers]
        cluster_idx = np.argmin(distances)
        clusters[cluster_idx].append(point)

    # Recompute centers, drop empty clusters
    new_centers = []
    new_clusters = []
    for i in range(len(clusters)):
        if clusters[i]:  # Non-empty cluster
            new_center = np.mean(clusters[i], axis=0)
            new_centers.append(new_center)
            new_clusters.append(clusters[i])
    new_centers = np.array(new_centers)

    # Check for convergence: centers unchanged and cluster count the same
    if len(new_centers) == len(cluster_centers) and np.allclose(cluster_centers, new_centers, rtol=1e-6, atol=1e-8):
        break

    # Update centers and clusters
    cluster_centers = new_centers
    clusters = new_clusters

# (4) Compute sum-of-squares error (SSE)
sse = 0.0
for i in range(len(clusters)):
    for point in clusters[i]:
        sse += np.sum((point - cluster_centers[i]) ** 2)

# (5) Output SSE rounded to 4 decimal places
print(f"{sse:.4f}")