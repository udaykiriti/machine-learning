import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

PATH = '../input/'
train_data = pd.read_csv(PATH + 'train.csv')
test_data = pd.read_csv(PATH + 'test.csv')
gender_submission = pd.read_csv(PATH + 'gender_submission.csv')

train_data.head()
test_data.head()
train_data.describe()
train_data.columns
train_data.dtypes

column_names = train_data.columns
for column in column_names:
    print(column + ' - ' + str(train_data[column].isnull().sum()))

train_data.Survived.value_counts()
plt = train_data.Survived.value_counts().plot('bar')
plt.set_xlabel('Survived or not')
plt.set_ylabel('Passenger Count')
plt.show()

plt = train_data.Pclass.value_counts().sort_index().plot('bar', title='')
plt.set_xlabel('Pclass')
plt.set_ylabel('Survival Probability')
plt.show()

train_data[['Pclass', 'Survived']].groupby('Pclass').count()
train_data[['Pclass', 'Survived']].groupby('Pclass').sum()

plt = train_data[['Pclass', 'Survived']].groupby('Pclass').mean().Survived.plot('bar')
plt.set_xlabel('Pclass')
plt.set_ylabel('Survival Probability')
plt.show()

plt = train_data.Sex.value_counts().sort_index().plot('bar')
plt.set_xlabel('Sex')
plt.set_ylabel('Passenger count')
plt.show()

plt = train_data[['Sex', 'Survived']].groupby('Sex').mean().Survived.plot('bar')
plt.set_xlabel('Sex')
plt.set_ylabel('Survival Probability')
plt.show()

plt = train_data.Embarked.value_counts().sort_index().plot('bar')
plt.set_xlabel('Embarked')
plt.set_ylabel('Passenger count')
plt.show()

plt = train_data[['Embarked', 'Survived']].groupby('Embarked').mean().Survived.plot('bar')
plt.set_xlabel('Embarked')
plt.set_ylabel('Survival Probability')
plt.show()

plt = train_data.SibSp.value_counts().sort_index().plot('bar')
plt.set_xlabel('SibSp')
plt.set_ylabel('Passenger count')
plt.show()

plt = train_data[['SibSp', 'Survived']].groupby('SibSp').mean().Survived.plot('bar')
plt.set_xlabel('SibSp')
plt.set_ylabel('Survival Probability')
plt.show()

plt = train_data.Parch.value_counts().sort_index().plot('bar')
plt.set_xlabel('Parch')
plt.set_ylabel('Passenger count')
plt.show()

plt = train_data[['Parch', 'Survived']].groupby('Parch').mean().Survived.plot('bar')
plt.set_xlabel('Parch')
plt.set_ylabel('Survival Probability')
plt.show()

sns.factorplot('Pclass', col = 'Embarked', data = train_data, kind = 'count')
sns.factorplot('Sex', col = 'Pclass', data = train_data, kind = 'count')
sns.factorplot('Sex', col = 'Embarked', data = train_data, kind = 'count')

train_data.head()
train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1
train_data.head()
train_data = train_data.drop(columns=['Ticket', 'PassengerId', 'Cabin'])
train_data.head()

train_data['Sex'] = train_data['Sex'].map({'male':0, 'female':1})
train_data['Embarked'] = train_data['Embarked'].map({'C':0, 'Q':1, 'S':2})
train_data.head()

train_data['Title'] = train_data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
train_data = train_data.drop(columns='Name')
train_data.Title.value_counts().plot('bar')

train_data['Title'] = train_data['Title'].replace(['Dr', 'Rev', 'Col', 'Major', 'Countess', 'Sir', 'Jonkheer', 'Lady', 'Capt', 'Don'], 'Others')
train_data['Title'] = train_data['Title'].replace('Ms', 'Miss')
train_data['Title'] = train_data['Title'].replace('Mme', 'Mrs')
train_data['Title'] = train_data['Title'].replace('Mlle', 'Miss')

plt = train_data.Title.value_counts().sort_index().plot('bar')
plt.set_xlabel('Title')
plt.set_ylabel('Passenger count')

plt = train_data[['Title', 'Survived']].groupby('Title').mean().Survived.plot('bar')
plt.set_xlabel('Title')
plt.set_ylabel('Survival Probability')

train_data['Title'] = train_data['Title'].map({'Master':0, 'Miss':1, 'Mr':2, 'Mrs':3, 'Others':4})
train_data.head()
corr_matrix = train_data.corr()
plt.figure(figsize=(9, 8))
sns.heatmap(data = corr_matrix,cmap='BrBG', annot=True, linewidths=0.2)