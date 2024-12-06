import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
iris_data = pd.read_csv(url, header=None, names=columns)

# Display initial data information
print("First few rows of the dataset:")
print(iris_data.head())
print("\nSummary statistics:")
print(iris_data.describe())
print("\nData types:")
print(iris_data.dtypes)

# Prepare features and target
X = iris_data.drop('class', axis=1)
y = iris_data['class']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Decision Tree Classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
print("\nConfusion Matrix:")
confusion = confusion_matrix(y_test, y_pred)
print(confusion)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualize the confusion matrix
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=clf.classes_, yticklabels=clf.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Plot the decision tree
plt.figure(figsize=(12, 8))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=clf.classes_)
plt.title('Decision Tree Visualization')
plt.show()