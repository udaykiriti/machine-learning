import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import tree
from IPython.display import Image
import pydotplus

# Load the data
data = pd.read_csv("Movie_regression.csv")

# Display the first few rows of the data
print(data.head(10))

# Display data information
print(data.info())

# Fill missing values in 'Time_taken' with the mean value
data["Time_taken"].fillna(value=data["Time_taken"].mean(), inplace=True)

# Confirm no missing values in 'Time_taken'
print(data.info())

# Convert 'Budget' and 'Collection' to crores (by dividing by 100)
data["Budget"] = data["Budget"] / 100
data["Collection"] = data["Collection"] / 100

# Display updated data
print(data.head())

# Plot distributions of categorical variables
sns.countplot(x="Genre", data=data)
plt.show()
sns.countplot(x="3D_available", data=data)
plt.show()

# Plot relationships between numerical variables and 'Collection'
sns.scatterplot(x="Lead_ Actor_Rating", y="Collection", data=data)
plt.title("Collection Vs Actor Rating")
plt.show()

sns.scatterplot(x="Num_multiplex", y="Collection", data=data)
plt.title("Multiplexes vs Collection")
plt.show()

sns.scatterplot(x="Budget", y="Collection", data=data)
plt.title("Budget spent vs Collection")
plt.show()

sns.scatterplot(x="Critic_rating", y="Collection", data=data)
plt.title("Critic Rating vs Collection")
plt.show()

# Convert categorical columns to dummy variables
# Ensure to drop first column for avoiding multicollinearity
data = pd.get_dummies(data, columns=["3D_available", "Genre"], drop_first=True)

# Drop unnecessary columns
data = data.drop(["Multiplex coverage", "Time_taken", "Num_multiplex"], axis=1)

# Check data types to confirm there are no non-numeric columns
print(data.dtypes)

# Ensure all columns are numeric for correlation computation
# Convert any non-numeric values to NaN
data = data.apply(pd.to_numeric, errors='coerce')

# Drop rows with any NaN values
data.dropna(inplace=True)

# Display the updated dataframe and its correlation
print("Correlations after processing:")


# Define the features (X) and target (y)
x = data.loc[:, data.columns != "Collection"]
y = data["Collection"]

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Build the decision tree regression model
model = DecisionTreeRegressor(max_depth=3)
model.fit(x_train, y_train)

# Predict on the test set
pred = model.predict(x_test)

# Evaluate the model
mse = mean_squared_error(y_test, pred)
r2 = r2_score(y_test, pred)

# Display results
print(f"Predictions: {pred}")
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Calculate adjusted R^2
n = x_train.shape[0]
p = x_train.shape[1]
adj_r2 = 1 - ((1 - r2) * (n - 1)) / (n - p - 1)
print(f"Adjusted R^2: {adj_r2}")

# Visualize the decision tree
dot_data = tree.export_graphviz(model, out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())


# Prediction function for user input
def prediction_calculator(n):
    for i in range(n):
        print(f"\nENTER THE INPUTS FOR MOVIE {i + 1}: ")
        ME = float(input("Marketing expense (in crores): "))
        PE = float(input("Production expense (in crores): "))
        Budget = float(input("Budget (in crores): "))
        ML = float(input("Movie Length (in mins): "))
        Act_rate = float(input("Lead Actor rating (1-10 IMDb scale): "))
        Actr_rate = float(input("Lead Actress rating (1-10 IMDb scale): "))
        DR = float(input("Director rating (1-10 IMDb scale): "))
        PR = float(input("Producer rating (1-10 IMDb scale): "))
        CR = float(input("Critic rating (1-10 IMDb critic rating scale): "))
        TV = float(input("Trailer views (in lakhs): "))
        TH = int(input("Twitter hashtags: "))
        Avg_act = float(input("Average actors: "))
        three_d = int(input("3D (1-yes/0-no): "))
        genre = input("Genre (action/comedy/drama/thriller): ").lower()

        # One-hot encoding for the genre
        ga = gc = gt = gd = 0
        if genre == "action":
            ga = 1
        elif genre == "comedy":
            gc = 1
        elif genre == "drama":
            gd = 1
        elif genre == "thriller":
            gt = 1

        # Create an array with input values
        c = np.array([ME, PE, Budget, ML, Act_rate, Actr_rate, DR, PR, CR, TV, TH, Avg_act, three_d, ga, gc, gd, gt])
        c_rs = c.reshape(1, -1)

        # Predict using the trained model
        pred = model.predict(c_rs)
        print(f"Predicted Collection (in crores): {pred[0]}")


# Get number of use cases for prediction
use_case = int(input("ENTER NUMBER OF USE CASES: "))
prediction_calculator(use_case)
