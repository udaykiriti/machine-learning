import numpy as np

class HebbianLearning:
    def __init__(self, input_size):
        # Initialize weights randomly
        self.weights = np.random.rand(input_size)

    
    def train(self, X, y, epochs):
        for _ in range(epochs):
            for i in range(len(X)):
                # Update weights based on Hebbian rule
                prediction = self.predict(X[i])
                self.weights += X[i] * (y[i] - prediction)

    def predict(self, x):
        # Activation function (threshold for AND gate)
        activation = np.dot(x, self.weights)
        return 1 if activation > 0.5 else 0

# Define the AND gate inputs and outputs
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([0, 0, 0, 1])

# Initialize and train the Hebbian learning model
hebbian_model = HebbianLearning(input_size=2)
hebbian_model.train(X, y, epochs=10)

# Display the final weight matrix
print("Final Weight Matrix after training:", hebbian_model.weights)

# Test the model
print("AND Gate Predictions:")
for x in X:
    print(f"Input: {x}, Predicted Output: {hebbian_model.predict(x)}")
