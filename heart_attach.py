import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import tree


data = pd.read_csv('heart.csv')

print(data.head())
print(data.describe())
print(data.info())
X = data.drop('target', axis=1)
y = data['target']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("Training set shape:", X_train.shape, "Testing set shape:", X_test.shape)


clf = DecisionTreeClassifier(criterion='gini', random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

print("Classification Report:")
classification_report_dict = classification_report(y_test, y_pred, output_dict=True)
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

precision = precision_score(y_test, y_pred, average=None)
recall = recall_score(y_test, y_pred, average=None)
f1 = f1_score(y_test, y_pred, average=None)


for i, cls in enumerate(clf.classes_):
    print(f'Class {cls}:')
    print(f'  Precision: {precision[i]}')
    print(f'  Recall: {recall[i]}')
    print(f'  F1 Score: {f1[i]}')

plt.figure(figsize=(30, 15))
tree.plot_tree(clf, filled=True, feature_names=X.columns, class_names=[str(cls) for cls in clf.classes_])
plt.show()