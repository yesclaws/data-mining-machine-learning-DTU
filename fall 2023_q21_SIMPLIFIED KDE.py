#Simplified KDE in this context refers to directly using the KDE formula to calculate the density at specific test points without integrating over the entire distribution.

import numpy as np

def kde_density(x, data, bandwidth):
    n = len(data)
    density = (1 / (n * np.sqrt(2 * np.pi * bandwidth**2))) * np.sum(
        np.exp(-((x - data)**2) / (2 * bandwidth**2))
    )
    return density
# Bandwidth: The bandwidth 𝜎 determines the smoothness of the density estimate. A smaller 𝜎 results in a more sensitive estimate, while a larger 𝜎 smooths out the density.
# More sensitive = refers to how closely the estimated density function follows the individual data points
# Given data points
data = np.array([-0.7,1.1,1.8]) #CHANGE THESE VALUES
bandwidth_squared = 0.5 #CHANGE THESE VALUES
bandwidth = 1

# Test points
test_points = [-0.9,-0.3,0.6,1.2] # CHANGE VALUES

# Anomaly threshold
threshold = 0.015 # CHANGE VALUES


# Calculate densities at test points
for x in test_points:
    density = kde_density(x, data, bandwidth)
    print(f"Density at x = {x}: {density:.6f}")


# Check if test points are anomalies
for x in test_points:
    density = kde_density(x, data, bandwidth)
    print(density)
    if density < threshold:
        print(f"Point {x} is an anomaly.")
    else:
        print(f"Point {x} is not an anomaly.")
