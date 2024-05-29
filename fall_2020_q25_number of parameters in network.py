input_features = 4
hidden_units = 6
output_classes = 3

# Weights and biases from input to hidden layer
weights_input_hidden = input_features * hidden_units
biases_hidden = hidden_units

# Weights and biases from hidden to output layer
weights_hidden_output = hidden_units * output_classes
biases_output = output_classes

total_parameters = weights_input_hidden + biases_hidden + weights_hidden_output + biases_output

print(f"Total parameters in the network: {total_parameters}")
