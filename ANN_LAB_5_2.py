import numpy as np

class DifferentialHebbianLearning:
    def __init__(self, input_size, learning_rate=0.1):
        # Initialize weights randomly
        self.weights = np.random.rand(input_size)
        self.learning_rate = learning_rate

    def train(self, X, y, epochs):
        for _ in range(epochs):
            for i in range(len(X)):
                # Predict output
                prediction = self.predict(X[i])
                # Update weights based on the differential Hebbian rule
                self.weights += self.learning_rate * X[i] * (y[i] - prediction)

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

# Initialize and train the differential Hebbian learning model
dhebbian_model = DifferentialHebbianLearning(input_size=2)
dhebbian_model.train(X, y, epochs=10)

# Display the final weight matrix
print("Final Weight Matrix after training:", dhebbian_model.weights)

# Test the model
print("AND Gate Predictions:")
for x in X:
    print(f"Input: {x}, Predicted Output: {dhebbian_model.predict(x)}")
