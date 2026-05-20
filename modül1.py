import numpy as np

# Training data
inputs = np.array([
    [-5],
    [-4],
    [-3],
    [-2],
    [-1],
    [0],
    [1],
    [2],
    [3],
    [4],
    [5]
])

# Correct answers
# Negative numbers -> 0
# Positive numbers -> 1
outputs = np.array([
    [0],
    [0],
    [0],
    [0],
    [0],
    [0],
    [1],
    [1],
    [1],
    [1],
    [1]
])

# Random starting values
weight = np.random.randn()
bias = np.random.randn()

# Learning rate
lr = 0.1

# Activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative
def sigmoid_derivative(x):
    return x * (1 - x)

# Training
for epoch in range(500000):

    # Convert matrix into simple array
    x = inputs.flatten()

    # Neuron formula
    weighted_sum = x * weight + bias

    # Predictions
    predictions = sigmoid(weighted_sum)

    # Compare predictions to correct answers
    error = outputs.flatten() - predictions

    # Calculate adjustments
    adjustments = error * sigmoid_derivative(predictions)

    # Learn
    weight += np.mean(adjustments * x) * lr
    bias += np.mean(adjustments) * lr

    # Print every 100 epochs
    if epoch % 10000 == 0:

        print("\nEpoch:", epoch)

        print("Weight:", round(weight, 4))
        print("Bias:", round(bias, 4))

        print("Predictions:")

        for i in range(len(x)):
            print(
                "Input:",
                x[i],
                "Prediction:",
                round(predictions[i], 4),
                "Expected:",
                outputs[i][0]
            )

# Final test
print("\nFINAL TESTS")

tests = [-10, -1, -7, -3, 0, 1, 10]

for t in tests:
    result = sigmoid(t * weight + bias)

    print(
        "Input:",
        t,
        "Prediction:",
        round(result, 4)
    )