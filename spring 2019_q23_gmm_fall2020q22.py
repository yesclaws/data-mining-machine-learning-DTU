import numpy as np

# Parameters for the Gaussians
mu = np.array([18.347, 14.997, 18.421])
sigma = np.array([1.2193, 0.986, 1.1354])
weights = np.array([0.13, 0.55, 0.32])

# Observation
x0 = 15.38

# Probability density function for a normal distribution
def normal_pdf(x, mean, std):
    return (1 / (np.sqrt(2 * np.pi * std**2))) * np.exp(-((x - mean)**2) / (2 * std**2))

# Calculate densities
densities = np.array([normal_pdf(x0, mu[k], sigma[k]) for k in range(3)])

# Calculate the posterior probabilities using Bayes' theorem
posterior = (densities * weights) / (np.sum(densities * weights))


#Arrays in Python are zero-indexed, meaning posterior[0] corresponds to the first cluster (k=1), posterior[1] to the second cluster (k=2), and posterior[2] to the third cluster (k=3).
print(f"Posterior probability of x0 belonging to cluster 2: {posterior[1]:.3f}")
#print(f"Posterior probability of x0 belonging to cluster 1: {posterior[0]:.3f}")
#print(f"Posterior probability of x0 belonging to cluster 3: {posterior[2]:.3f}")