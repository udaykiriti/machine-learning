import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

import matplotlib.pyplot as plt
from sklearn import tree

data=pd.read_csv('seeds.csv')
print(data)
print(data.describe())
print(data.info())

x=data.drop('Type',axis=1)
y=data['Type']

X_train,X_test,Y_train,