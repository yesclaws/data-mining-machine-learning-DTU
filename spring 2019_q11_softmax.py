import numpy as np

# Define the softmax function
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# Store weight matrices for classes in a dictionary for better readability and accessibility
weights = {
    'A': np.array([[-0.77, 0.26], [-5.54, -2.09], [0.01, -0.03]]),  # Weights for option A
    'B': np.array([[0.51, 0.1], [1.65, 3.8], [0.01, 0.04]]),        # Weights for option B
    'C': np.array([[-0.9, -0.09], [4.39, 2.45], [0.0, -0.04]]),     # Weights for option C
    'D': np.array([[-1.22, 0.28], [-9.88, 2.9], [0.0, -0.01]])      # Weights for option D
}

# Example input (b1, b2)
b = np.array([0, -1])  # Example input where b1 = 0, b2 = -1

# Iterate over each set of weights and compute predictions
for label, w in weights.items():
    y_hat = np.dot(w, b)
    probabilities = softmax(y_hat)
    predicted_class = np.argmax(probabilities) + 1  # +1 because class indices are 0-based

    # Print the probabilities of belonging to each class and the predicted class
    print(f"Probabilities for option {label}:", probabilities)
    print(f"Predicted class for option {label}:", predicted_class)
