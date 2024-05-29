import numpy as np

total_obs = 6
weight = 6

# Calculating alpha_t
epsilon_t = 4/6 # number of obs that was MISclassified 
alpha_t = 0.5 * np.log((1 - epsilon_t) / epsilon_t)

print(alpha_t)

# Initial weights
weights_initial = np.full(total_obs, total_obs/weight) # assume equal weight

# Misclassification - x out of "7" (this number depends on number of total obs) are misclassified (if misclassified put true, if classified correctly put false)
weights_new = np.where([False, False, True, True, True, True], 
                       weights_initial * np.exp(alpha_t), 
                       weights_initial * np.exp(-alpha_t))


print("New weights",weights_new)

# Normalization factor
Z = weights_new.sum()
print(Z)

# Normalized new weights
weights_new_normalized = weights_new / Z

print("NORMALIZED New weights", weights_new_normalized)
