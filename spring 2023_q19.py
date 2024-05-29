import numpy as np
"""
    Applies a given transformation to the input data, standardizes it,
    predicts outcomes, and calculates the regularized loss.

    Parameters:
    - x: array of input values.
    - y: array of output values.
    - w: weight array [w1, w2, ...].
    - lambda_reg: regularization parameter lambda.
    - transformation: a function that takes x and returns the transformed x~.

    Returns:
    - Regularized loss.
    """
def evaluate_transformation(x, y, w, lambda_reg, transformation):
    # Transform and standardize data
    x_transformed = transformation(x)
    print("Transformed x:", x_transformed)
    
    x_transformed = (x_transformed - np.mean(x_transformed, axis=0)) / np.std(x_transformed, axis=0, ddof=1)
    print("Standardized x:", x_transformed)

    # Calculate predictions
    predictions = x_transformed.dot(w)
    print("Predictions:", predictions)

    # Compute squared errors
    squared_errors = (y - predictions) ** 2
    loss = np.sum(squared_errors)
    print("Squared Errors:", squared_errors)

    # Add regularization term
    regularization_term = lambda_reg * (np.linalg.norm(w)**2)
    total_loss = loss + regularization_term
    print("Regularization Term:", regularization_term)

    return total_loss

# Adjust transformation functions if necessary and re-run the function with debug outputs


# Example transformations
def transformation_A(x):
    return np.vstack([x, x**3]).T

def transformation_B(x):
    return np.vstack([x, np.sin(x)]).T

def transformation_D(x):
    return np.vstack([x, x**2]).T

# Usage
x = np.array([-0.5, 0.39, 1.19, -1.08])
y = np.array([-0.86, -0.61, 1.37, 0.1])
w = np.array([0.39, 0.77])  # Example weights
lambda_reg = 0.25  # Example lambda

# Evaluate each transformation
loss_A = evaluate_transformation(x, y, w, lambda_reg, transformation_A)
loss_B = evaluate_transformation(x, y, w, lambda_reg, transformation_B)
loss_D = evaluate_transformation(x, y, w, lambda_reg, transformation_D)

print("Loss for Transformation A:", loss_A)
print("Loss for Transformation B:", loss_B)
print("Loss for Transformation D:", loss_D)
