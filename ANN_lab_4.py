import numpy as np
import matplotlib.pyplot as plt

# Define activation functions
def linear(x):
    return x

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def bipolar_sigmoid(x):
    return (2 / (1 + np.exp(-x))) - 1

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, x * alpha)

def softplus(x):
    return np.log(1 + np.exp(x))

# Generate values for x
x = np.linspace(-10, 10, 100)

# Plot the functions
plt.figure(figsize=(15,15))

plt.subplot(4, 2, 1)
plt.plot(x, linear(x), label='Linear')
plt.title('Linear')

plt.subplot(4, 2, 2)
plt.plot(x, sigmoid(x), label='Sigmoid')
plt.title('Sigmoid')

plt.subplot(4, 2, 3)
plt.plot(x, bipolar_sigmoid(x), label='Bipolar Sigmoid')
plt.title('Bipolar Sigmoid')

plt.subplot(4, 2, 4)
plt.plot(x, tanh(x), label='Tanh')
plt.title('Tanh')

plt.subplot(4, 2, 5)
plt.plot(x, relu(x), label='ReLU')
plt.title('ReLU')

plt.subplot(4, 2, 6)
plt.plot(x, leaky_relu(x), label='Leaky ReLU')
plt.title('Leaky ReLU')

plt.subplot(4, 2, 7)
plt.plot(x, softplus(x), label='Softplus')
plt.title('Softplus')

plt.tight_layout()
plt.show()
