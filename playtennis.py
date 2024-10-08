import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Step 1: Load data from playtennis.csv
df = pd.read_csv('playtennis.csv')

# Step 2: Encode categorical variables
enc = OneHotEncoder(drop='first')
encoded_data = enc.fit_transform(df[['Outlook', 'Temperature', 'Humidity', 'Wind']]).toarray()

# Prepare features (X) and target (y)
X = pd.DataFrame(encoded_data, columns=enc.get_feature_names_out())
y = df['PlayTennis'].apply(lambda x: 1 if x == "Yes" else 0)

# Step 3: Perform logistic regression
logistic_model = LogisticRegression()
logistic_model.fit(X, y)

# Predict and calculate accuracy for Logistic Regression
y_pred_logistic = logistic_model.predict(X)
accuracy_logistic = accuracy_score(y, y_pred_logistic)
print(f"Logistic Regression Accuracy: {accuracy_logistic * 100:.2f}%")

# Confusion matrix for Logistic Regression
conf_matrix_logistic = confusion_matrix(y, y_pred_logistic)
print("Logistic Regression Confusion Matrix:")
print(conf_matrix_logistic)

# Step 4: Perform decision tree classification
decision_tree_model = DecisionTreeClassifier()
decision_tree_model.fit(X, y)

# Predict and calculate accuracy for Decision Tree
y_pred_tree = decision_tree_model.predict(X)
accuracy_tree = accuracy_score(y, y_pred_tree)
print(f"Decision Tree Accuracy: {accuracy_tree * 100:.2f}%")

# Confusion matrix for Decision Tree
conf_matrix_tree = confusion_matrix(y, y_pred_tree)
print("Decision Tree Confusion Matrix:")
print(conf_matrix_tree)

# Step 5: Generate and plot correlation matrix heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(X.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Features")
plt.show()

# Step 6: Plot regression coefficients
coefficients = pd.Series(logistic_model.coef_[0], index=X.columns)
plt.figure(figsize=(10, 6))
coefficients.plot(kind='bar')
plt.title("Logistic Regression Coefficients")
plt.ylabel("Coefficient Value")
plt.show()

# Step 7: Plot Decision Tree (optional)
from sklearn.tree import plot_tree

plt.figure(figsize=(20, 10))
plot_tree(decision_tree_model, feature_names=X.columns, class_names=['No', 'Yes'], filled=True)
plt.title("Decision Tree Visualization")
plt.show()
