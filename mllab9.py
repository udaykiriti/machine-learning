import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA  # Added PCA import

# Load the dataset
data = pd.read_csv('emails.csv')  # Replace 'emails.csv' with your dataset file
print(data.head())

# Drop irrelevant columns like 'Email' if present
data = data.drop(['Email No.', 'other_irrelevant_column'], axis=1, errors='ignore')

# Separate features and target
X = data.drop('Prediction', axis=1)  # Features (replace 'Prediction' with your target column name)
y = data['Prediction']               # Target (replace 'Prediction' with your target column name)

print(X.head())

# Handle missing values (if any)
X.fillna(X.mean(), inplace=True)

# Convert categorical variables to numerical form
X = pd.get_dummies(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply PCA before fitting the XGBoost model
pca = PCA(n_components=0.70)  # Retain 70% of the variance
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Create and train the XGBoost model
xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False)
xgb_model.fit(X_train_pca, y_train)

# Predict on the test set
y_pred = xgb_model.predict(X_test_pca)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy * 100:.2f}%')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{class_report}')

