import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

from Models.tree_algorithms import RandomForest

iris = datasets.load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)

X_train, X_test, y_train, y_test = train_test_split(df[iris.feature_names], iris.target, test_size=0.5, stratify=iris.target, random_state=123456)

rf = RandomForest(X_train, y_train, n_trees=5, min_leaf=3)
predictions = rf.predict(X_train.values)