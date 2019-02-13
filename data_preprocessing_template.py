# Import libraries
import numpy as np
import matplot.pyplot as plt
import pandas as pd

# Import dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Splitting dataset into the training set and test set
from sklear.cross_validation import train_test_split
X_train, X_test, y_train, t_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling 
""" from sklearn.preprocessing import StandardScaler
sc_X = sc_X.fit_transform(X_train)
X_train = sc_X.transform(X_text)"""
