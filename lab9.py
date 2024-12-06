import numpy as np

X = np.array([[0, 0, 1, 1],
              [1, 0, 1, 0],
              [1, 1, 0, 0],
              [0, 1, 1, 0]], dtype=np.float64)

y = np.array([1, 2, 2, 1])
w1 = np.array([0, 0, 1, 1], dtype=np.float64)
w2 = np.array([1, 0, 1, 0], dtype=np.float64)

learning_rate = 1.0
epochs = 100
for epoch in range(epochs):
    for i, x in enumerate(X):
        dist_w1 = np.linalg.norm(x - w1)
        dist_w2 = np.linalg.norm(x - w2)

        if dist_w1 < dist_w2:
            if y[i] == 1:
                w1 += learning_rate * (x - w1)
            else:
                w1 -= learning_rate * (x - w1)
        else:
            if y[i] == 2:
                w2 += learning_rate * (x - w2)
            else:
                w2 -= learning_rate * (x - w2)

test_points = np.array([[0, 0, 1, 0],
                        [1, 0, 0, 1]], dtype=np.float64)

for test in test_points:
    dist_w1 = np.linalg.norm(test - w1)
    dist_w2 = np.linalg.norm(test - w2)

    if dist_w1 < dist_w2:
        print(f"Test point {test} is classified as Class 1")
    else:
        print(f"Test point {test} is classified as Class 2")

print("\nFinal prototypes:")
print("Class 1 prototype (w1):", w1)
print("Class 2 prototype (w2):", w2)
