from scipy.stats import binom

# Define n1 and n2
n1 = 28
n2 = 35
N = n1 + n2
m = min(n1, n2)

# Calculate p-value using the CDF of the binomial distribution
p_value = 2 * binom.cdf(m, N, 0.5)  # Two-tailed test
print(f"The p-value from the binomial distribution is {p_value:.3f}")
