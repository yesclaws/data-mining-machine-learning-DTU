import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

# Sample data
data = np.array([-3, -1, 5, 6]).reshape(-1, 1)

# Test points
x_plot = np.linspace(-6, 10, 1000).reshape(-1, 1) 

# KDE with small bandwidth
kde_small = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(data)
log_dens_small = kde_small.score_samples(x_plot)

# KDE with large bandwidth
kde_large = KernelDensity(kernel='gaussian', bandwidth=2.0).fit(data)
log_dens_large = kde_large.score_samples(x_plot)

# Plotting
plt.figure(figsize=(10, 6))
plt.fill_between(x_plot[:, 0], np.exp(log_dens_small), alpha=0.5, label='Bandwidth = 0.2')
plt.fill_between(x_plot[:, 0], np.exp(log_dens_large), alpha=0.5, label='Bandwidth = 2.0')
plt.plot(data[:, 0], np.full_like(data[:, 0], -0.01), '|k', markeredgewidth=1)
plt.legend()
plt.title('KDE with Different Bandwidths')
plt.show()
