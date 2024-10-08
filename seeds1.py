# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn import tree

# Loading the dataset
data = pd.read_csv('seeds.csv')

# Displaying the dataset and its characteristics
print(data)
print(data.describe())
print(data.info())

# Splitting the dataset into features and target variable
X = data.drop('Type', axis=1)
y = data['Type']

# Splitting the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing and fitting the Decision Tree Classifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, Y_train)

# Making predictions on the test set
y_pred = classifier.predict(X_test)

# Evaluating the model
print("Accuracy:", accuracy_score(Y_test, y_pred))
print("Classification Report:\n", classification_report(Y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(Y_test, y_pred))

# Plotting the decision tree
plt.figure(figsize=(20,10))
tree.plot_tree(classifier, filled=True, feature_names=X.columns, class_names=['1', '2', '3'])
plt.show()