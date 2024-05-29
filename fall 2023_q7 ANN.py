import numpy as np

# Activation function: tanh
def tanh(x):
    return np.tanh(x)

# Correct the weights to ensure proper formatting
# w1 should be a 2x3 matrix for compatibility with the input vector which includes the bias term
w1 = np.array([
    [2.2, 0.7, -0.3],  # Corrected weight matrix
    [-0.2, 0.8, 0.4]
])

w2 = np.array([-0.7, 0.5])  # Weights for the output layer (only 2 neurons hence length is 2)
w0_second = 2.2  # Bias for the output layer

# Test input vector, including the bias term at the start
x_test = np.array([1, -2.0, -1.88])  # Assuming the first element as bias if needed

# Compute hidden layer activations
hidden_inputs = np.dot(w1, x_test)  # Matrix multiplication of weights and inputs
hidden_activations = tanh(hidden_inputs)  # Apply tanh activation function

# Compute output
final_input = w0_second + np.dot(w2, hidden_activations)  # Combine the activations with the second layer weights
output = tanh(final_input)  # Apply tanh to compute final output

print("Output of the neural network:", output)
