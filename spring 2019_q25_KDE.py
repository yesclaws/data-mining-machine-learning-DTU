import numpy as np
from scipy.stats import norm

# Example dataset
data = np.array([3.918, -6.35, -2.677, -3.003])

# Bandwidth (kernel width sigma)
sigma = 2.0

# Function to calculate KDE with LOO
def kde_loo(data, sigma):
    n = len(data)
    loo_scores = np.zeros(n)
    
    # Calculate the KDE for each point, leaving it out
    for i in range(n):
        # Leave-one-out dataset
        loo_data = np.delete(data, i)
        
        # Calculate the density at the left-out point
        density = np.sum(norm.pdf(data[i], loc=loo_data, scale=sigma)) / (n - 1)
        loo_scores[i] = -np.log(density)  # Negative log-likelihood

    # Calculate the average LOO score
    average_loo_score = np.mean(loo_scores)
    return average_loo_score

# Calculate the LOO error for the given sigma
loo_error = kde_loo(data, sigma)
print(f"Leave-One-Out cross-validation error for sigma = {sigma}: {loo_error}")
