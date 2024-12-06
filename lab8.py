import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from minisom import MiniSom

iris = datasets.load_iris()
data = iris.data
target = iris.target

som_size = 20
som = MiniSom(som_size, som_size, data.shape[1], sigma=1.0, learning_rate=0.5)

som.train(data, 10000, verbose=True)

plt.figure(figsize=(10, 8))
plt.title("Self-Organizing Map - Iris Dataset")
for i, (x, t) in enumerate(zip(data, target)):
    win = som.winner(x)
    plt.text(win[0], win[1], str(t), color=plt.cm.jet(t / 3.),
             fontweight='bold', ha='center', va='center',
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

plt.imshow(som.distance_map().T, cmap='bone')
plt.colorbar()
plt.show()
