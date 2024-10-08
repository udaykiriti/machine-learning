import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('heart.csv')

print(data.head())
print(data.describe())
print(data.info())

age_bins = [29, 39, 49, 59, 69, 79, 89]
age_labels = ['20-30', '31-40', '41-50', '51-60', '61-70', '71-89']
data['age_group'] = pd.cut(data['age'], bins=age_bins, labels=age_labels)
age_group_counts = data['age_group'].value_counts()

plt.figure(figsize=(8, 8))
age_group_counts.plot.pie(autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired(range(len(age_group_counts))))
plt.title('Age Distribution')
plt.ylabel('')
plt.show()

age_trestbps_mean = data.groupby('age_group')['trestbps'].mean().reindex(age_labels)

plt.figure(figsize=(10, 6))
age_trestbps_mean.plot(kind='bar', color='black')
plt.title(' Blood Pressure by Age Group')
plt.xlabel('Age ')
plt.ylabel('trestbps')
plt.xticks(rotation=45)
plt.show()
