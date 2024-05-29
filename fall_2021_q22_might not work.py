import numpy as np

# Weight vector w
w = np.array([-np.sqrt(3)/np.sqrt(20), 15])

# Calculate norm squared of w
w_norm_squared = np.dot(w, w.T)

# Given cost function value and assumed RSS
E_lambda = 8
RSS = 6
lambda_value = (E_lambda - RSS) / w_norm_squared

print("Lambda =", lambda_value)
