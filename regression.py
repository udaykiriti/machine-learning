import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

Dst = pd.read_csv("Salary_Data.csv")

X = Dst["YearsExperience"].values.reshape(-1, 1)
Y = Dst["Salary"].values

plt.scatter(X, Y)
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Salary vs Years of Experience")
plt.show()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

model = LinearRegression()
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

plt.scatter(X_train, Y_train, color='red')
plt.plot(X_train, model.predict(X_train), color='red')
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Salary vs Years of Experience (Training set)")
plt.show()

print("Coefficient:", model.coef_)
print("Intercept:", model.intercept_)
