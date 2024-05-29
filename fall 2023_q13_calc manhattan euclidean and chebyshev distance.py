import numpy as np

# Define the points and their classes
points = np.array([
    [-0.4, -0.8],
    [-0.9, 0.3],
    [0, 0.9],
    [1, -0.1],
    [0.8, -0.7],
    [0.1, 0.8]
])
classes = ['C1', 'C1', 'C1', 'C2', 'C2', 'C2']
test_point = np.array([0, 0])

# Define the distance functions
def manhattan_distance(p1, p2):
    return np.sum(np.abs(p1 - p2))

def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2)**2))

def chebyshev_distance(p1, p2):
    return np.max(np.abs(p1 - p2))

# Calculate the distances
distances_d1 = np.array([manhattan_distance(test_point, point) for point in points])
distances_d2 = np.array([euclidean_distance(test_point, point) for point in points])
distances_dinf = np.array([chebyshev_distance(test_point, point) for point in points])

# Get the indices of the 3 nearest neighbors for each distance metric
nn_indices_d1 = np.argsort(distances_d1)[:3]
nn_indices_d2 = np.argsort(distances_d2)[:3]
nn_indices_dinf = np.argsort(distances_dinf)[:3]

# Determine the classes of the nearest neighbors
nn_classes_d1 = [classes[i] for i in nn_indices_d1]
nn_classes_d2 = [classes[i] for i in nn_indices_d2]
nn_classes_dinf = [classes[i] for i in nn_indices_dinf]

# Print the nearest neighbors and their classes
print("Manhattan Distance Nearest Neighbors:", nn_classes_d1)
print("Euclidean Distance Nearest Neighbors:", nn_classes_d2)
print("Chebyshev Distance Nearest Neighbors:", nn_classes_dinf)

# Determine the predicted class by majority vote
from collections import Counter

def majority_vote(nn_classes):
    vote_count = Counter(nn_classes)
    return vote_count.most_common(1)[0][0]

predicted_class_d1 = majority_vote(nn_classes_d1)
predicted_class_d2 = majority_vote(nn_classes_d2)
predicted_class_dinf = majority_vote(nn_classes_dinf)

print("Predicted Class by Manhattan Distance (d1):", predicted_class_d1)
print("Predicted Class by Euclidean Distance (d2):", predicted_class_d2)
print("Predicted Class by Chebyshev Distance (d_infinity):", predicted_class_dinf)
