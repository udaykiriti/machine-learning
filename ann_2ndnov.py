import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from ann_inlab import iris_data

iris=load_iris()
X=iris_data
y=iris.target

