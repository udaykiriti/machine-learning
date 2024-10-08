import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the dataset
Dst = pd.read_csv("kc_house_data.csv")

# Drop unnecessary columns 'date' and 'id'
Dst = Dst.drop(['date', 'id'], axis=1)

# Select features 'bedrooms' and 'bathrooms'
X = Dst[['bedrooms', 'bathrooms']]

# Select the target variable 'price'
Y = Dst['price']

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Create a Linear Regression model
Linear_Regression = LinearRegression()

# Train the model
Linear_Regression.fit(X_train, Y_train)

# Predict the target variable for the test set
Y_predict = Linear_Regression.predict(X_test)

# Print the predicted values
print("Predicted")
print(Y_predict)

# Print the coefficients of the model
print("Coefficients : ")
print(Linear_Regression.coef_)