import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,precision_score,recall_score
from sklearn.model_selection import train_test_split
import matplotlib as plt
from sklearn import tree

print(data.head())
print(data.shape())
print(data.info())

data.replace( to_replace '?',np,nan,inplace=True)
print(data.isNull)

