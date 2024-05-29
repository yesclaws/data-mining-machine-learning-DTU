# Uses rand index which requires the counting matrix

import numpy as np
from scipy.special import comb

def calculate_rand_index(counting_matrix):
    # Calculate S (number of agreements)
    S = 0
    for i in range(counting_matrix.shape[0]):
        for j in range(counting_matrix.shape[1]):
            S += comb(counting_matrix[i, j], 2)
    
    # Calculate total number of pairs
    total_elements = np.sum(counting_matrix)
    total_pairs = comb(total_elements, 2)
    
    # Calculate sum of binomials for rows (clusters in Z)
    row_sums = np.sum(counting_matrix, axis=1)
    sum_row_binomials = np.sum([comb(n, 2) for n in row_sums])
    
    # Calculate sum of binomials for columns (clusters in Q)
    col_sums = np.sum(counting_matrix, axis=0)
    sum_col_binomials = np.sum([comb(n, 2) for n in col_sums])
    
    # Calculate D (number of disagreements)
    D = total_pairs - (sum_row_binomials + sum_col_binomials - S)
    
    # Calculate Rand index
    rand_index = (S + D) / total_pairs
    return rand_index

# Example counting matrix
# counting_matrix = np.array([
#     [0, 0, 4],
#     [1, 1, 4]
# ])

# counting_matrix = np.array([
#     [2, 0, 0],
#     [2, 0, 1],
#     [2, 1, 2]
# ])

counting_matrix = np.array([
    [114, 0, 32],
    [0, 119, 0],
    [8, 0, 60]
])

rand_index = calculate_rand_index(counting_matrix)
print(f"Rand Index: {rand_index:.3f}")
