import pandas as pd
import matplotlib.pyplot as plt

tr_data = pd.read_csv("test.csv")

print(tr_data.head())

print(tr_data.dtypes)

column_names = tr_data.columns
for column in column_names:
    print(f"{column} - {tr_data[column].isnull().sum()}")

print(tr_data['Pclass'].value_counts())

ax = tr_data['Pclass'].value_counts().plot(kind='bar')
ax.set_xlabel('Passenger Class')
ax.set_ylabel('Count')
ax.set_title('Passenger Class Distribution')
plt.show()
