# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 14:28:39 2021

@author: Nwosu Obioma Bertrand

Supervised Learning Classification

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

# Get complete Data set
iris = load_iris()

# Load Data from iris
Data_iris = iris.data

# convert data from array to Dataframe
Data_iris = pd.DataFrame(Data_iris, columns=iris.feature_names)

# adding labels to the dataframe
Data_iris['label'] = iris.target

# Visualize the data
plt.scatter(Data_iris.iloc[:, 2], Data_iris.iloc[:, 3], c=iris.target)
plt.xlabel('Petal Length (cm)')
plt.ylabel('petal width (cm)')
plt.show()

# seperate label features from dependent variables
x = Data_iris.iloc[:, 0:4]
y = Data_iris.iloc[:, 4]

# Continue at video 3 supervised learning knn development
