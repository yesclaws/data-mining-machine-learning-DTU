# Check for orthogonal between principal compoenet directions

import numpy as np

# Define the matrix V
V = np.array([
    [-0.5939, 0.2906, -0.3413, 0.0621, 0.6652],
    [-0.6521, 0.0759, 0.0004, 0.3813, -0.6508],
    [0.2028, -0.5105, -0.7036, 0.4508, 0.0010],
    [-0.3696, -0.5414, -0.1781, -0.7244, -0.1173],
    [-0.2102, -0.5967, 0.5973, 0.3503, 0.3467]
])

# Calculate the inner product matrix
G = np.dot(V.T, V)

# Display the inner product matrix
print(G)