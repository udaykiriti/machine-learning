import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from ann_lab3 import X_train, X_test, y_train

iris = load_iris()
X = iris.data
y = iris.target

encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y.reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)


class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.weights_input_hidden = np.random.rand(input_size, hidden_size) * 0.1
        self.weights_hidden_output = np.random.rand(hidden_size, output_size) * 0.1

def sigmoid(self,z):
    return 1/(1+np.exp(-z))

def sigmoid_derivative(self,z):
    return z*(1-z)


def forward(self, X):
    self.hidden_input = np.dot(X, self.weights_input_hidden)
    self.hidden_output = self.sigmoid(self.hidden_input)
    self.final_input = np.dot(self.hidden_output, self.weights_hidden_output)
    self.final_output = self.sigmoid(self.final_input)
    return self.final_output


def backward(self, X, y):
    output_error = y - self.final_output
    output_delta = output_error * self.sigmoid_derivative(self.final_output)
    hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
    hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)

    # Update weights
    self.weights_hidden_output += np.dot(self.hidden_output.T, output_delta) * self.learning_rate
    self.weights_input_hidden += np.dot(X.T, hidden_delta) * self.learning_rate


def train(self, X, y, epochs=1000):
    for _ in range(epochs):
        self.forward(X)
        self.backward(X, y)


def predict(self, X):
    return self.forward(X)
# Initialize MLP
input_size = X_train.shape[1]
hidden_size = 5  # You can adjust this
output_size = y_onehot.shape[1]

mlp = MLP(input_size, hidden_size, output_size, learning_rate=0.01)

# Train the MLP
mlp.train(X_train, y_train, epochs=1000)
# Make predictions
predictions = mlp.predict(X_test)

# Convert predictions from probabilities to class labels
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)

# Calculate accuracy
accuracy = np.mean(predicted_classes == true_classes)
print(f'Accuracy: {accuracy * 100:.2f}%')



