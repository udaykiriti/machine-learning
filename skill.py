import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


data = pd.read_csv('solar_plant_data.csv')



data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

data['hour'] = data.index.hour
data['day_of_year'] = data.index.dayofyear


data.fillna(method='ffill', inplace=True)

X = data[['temperature', 'humidity', 'cloud_cover', 'hour', 'day_of_year']]
y = data['solar_output']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

plt.figure(figsize=(10, 5))
plt.plot(y_test.index, y_test, label='Actual Output', color='blue')
plt.plot(y_test.index, y_pred, label='Predicted Output', color='orange')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Solar Output')
plt.title('Actual vs Predicted Solar Output')
plt.show()
