import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Sample Data: DataFrame representing movie data (replace this with your actual data)
data = {
    'director': ['Dir1', 'Dir2', 'Dir3', 'Dir1', 'Dir3', 'Dir2', 'Dir1'],
    'hero': ['Hero1', 'Hero2', 'Hero1', 'Hero3', 'Hero2', 'Hero1', 'Hero3'],
    'heroine': ['Her1', 'Her2', 'Her1', 'Her3', 'Her2', 'Her3', 'Her1'],
    'genre': ['Action', 'Drama', 'Action', 'Comedy', 'Drama', 'Action', 'Comedy'],
    'budget': [50, 60, 45, 80, 70, 55, 40],  # in millions
    'box_office_success': [1, 0, 1, 1, 0, 1, 0]  # 1 = success, 0 = failure
}

# Create a DataFrame
df = pd.DataFrame(data)

# Step 1: Encode Categorical Data (Label Encoding)
label_encoder = LabelEncoder()

df['director_encoded'] = label_encoder.fit_transform(df['director'])
df['hero_encoded'] = label_encoder.fit_transform(df['hero'])
df['heroine_encoded'] = label_encoder.fit_transform(df['heroine'])
df['genre_encoded'] = label_encoder.fit_transform(df['genre'])

# Step 2: Prepare features (X) and target (y)
X = df[['director_encoded', 'hero_encoded', 'heroine_encoded', 'genre_encoded', 'budget']]
y = df['box_office_success']

# Step 3: Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Step 7: Take input from the console
director_input = input("Enter director name: ")
hero_input = input("Enter hero name: ")
heroine_input = input("Enter heroine name: ")
genre_input = input("Enter genre: ")
budget_input = float(input("Enter budget (in millions): "))

# Encode the inputs using the same LabelEncoder
director_encoded = label_encoder.fit(df['director']).transform([director_input])[0]
hero_encoded = label_encoder.fit(df['hero']).transform([hero_input])[0]
heroine_encoded = label_encoder.fit(df['heroine']).transform([heroine_input])[0]
genre_encoded = label_encoder.fit(df['genre']).transform([genre_input])[0]

# Step 8: Predict for the new movie based on console inputs
new_movie = [[director_encoded, hero_encoded, heroine_encoded, genre_encoded, budget_input]]
predicted_success = model.predict(new_movie)
print(f'Predicted Success: {"Success" if predicted_success[0] == 1 else "Failure"}')
