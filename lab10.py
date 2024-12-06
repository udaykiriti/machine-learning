import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

epochs = np.arange(1, 11)

accuracy = np.random.rand(10) * 0.9
precision = np.random.rand(10) * 0.8
recall = np.random.rand(10) * 0.7
f1_score_metric = np.random.rand(10) * 0.75
loss = np.random.rand(10) * 0.5
roc_auc = np.random.rand(10) * 0.85

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(epochs, precision, label="Precision", color="blue")
plt.title("Precision vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Precision")
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(epochs, recall, label="Recall", color="orange")
plt.title("Recall vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Recall")
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(epochs, f1_score_metric, label="F1-Score", color="green")
plt.title("F1-Score vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("F1-Score")
plt.legend()

plt.tight_layout()
plt.show()
