import numpy as np

# Data
x = np.array([1, 2, 3, 4])
y = np.array([6, 2, 3, 4])

# Transformation
X_transformed = np.column_stack((np.cos(np.pi * x / 2), np.sin(np.pi * x / 2)))

# Linear Regression to find w*
XtX = np.dot(X_transformed.T, X_transformed)
XtX_inv = np.linalg.inv(XtX)
Xty = np.dot(X_transformed.T, y)
w_star = np.dot(XtX_inv, Xty)

# Output the second element of w_star which corresponds to w2*
print("w2* =", w_star[1])
