import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("winequality-red.csv", sep=';')
features = data.drop("quality", axis=1)
labels = data["quality"]

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Model trained. Accuracy: {accuracy:.2f}")

print("\nEnter the following values to predict wine quality:")
user_input = []
for col in features.columns:
    val = float(input(f"{col}: "))
    user_input.append(val)

prediction = model.predict([user_input])
print(f"\nPredicted wine quality: {prediction[0]}")
