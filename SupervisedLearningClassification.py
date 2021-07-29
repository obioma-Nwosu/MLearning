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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


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

"""
K-NN CLassifier

1st parameter = K number of neighbors
2nd parameter = shows metric and p=1 is manhattan and not Euclidean
"""
kNN = KNeighborsClassifier(n_neighbors=6, metric='minkowski', p=1)

# fit data with our model, train model with x and y defined above
kNN.fit(x, y)

# predict with the model given new samples
x_New = np.array([[5.6, 3.4, 1.4, 0.1]])
kNN.predict(x_New)  # The 0 we get means setosa in target names

x_New2 = np.array([[7.5, 4, 5.5, 2]])
kNN.predict(x_New2)  # the 2 means virginica


"""
Split and test our dataset
"""

# the names are variables but the order whic they appear is important
# 1st parameter = dataset (x,y)
# 2nd parameter = testsize in percentage
# 3rd parameter = trainsize in percentage
# 4th parameter = random state important for the next time you come back and it
# uses the same random state
# 5th parameter = shuffle through our dataste
# 6th parameter = makes sure we test all our labels and not just one label
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                    train_size=0.8,
                                                    random_state=88,
                                                    shuffle=True, stratify=y)
# now we train our model set with x_train and y_train
kNN.fit(X_train, y_train)

# now test it with the test data
# we use only X_test because we want to compare it wit result stored in y_train
predicted_types = kNN.predict(X_test)

# check accuracy
accuracy_score(y_test, predicted_types)

"""
Decision Tree

similar to Knn we just did
"""
decisionTree = DecisionTreeClassifier()

decisionTree.fit(X_train, y_train)
predicted_types_dt = decisionTree.predict(X_test)

accuracy_score(y_test, predicted_types_dt)

"""
Cross validation

"""

# implement cross validation on the decision tree
# model, features and label, cv which is number of k or cross valiadation
Scores_Dt = cross_val_score(decisionTree, x, y, cv=10)


"""
Naive Bayes Classifier
"""
NB = GaussianNB()

NB.fit(X_train, y_train)
predicted_types_NB = NB.predict(X_test)

accuracy_score(y_test, predicted_types_NB)

scores_NB = cross_val_score(NB, x, y, cv=10)


"""
Logistic Regression
"""

Data_Cancer = load_breast_cancer()

# put data into x(samples) and y(target)
x_cancer = Data_Cancer.data
y_cancer = Data_Cancer.target

# train test and split
X_train_cancer, X_test_cancer, y_train_cancer, y_test_cancer = train_test_split(
    x_cancer, y_cancer, test_size=0.3, train_size=0.7, random_state=88)

LR = LogisticRegression()

LR.fit(X_train_cancer, y_train_cancer)

predicted_types_LR = LR.predict(X_test_cancer)

"""
Evaluation Metrics

"""
# confusion matrix and tp,tn,fp,fn table
matrix_variable = confusion_matrix(y_test_cancer, predicted_types_LR)
representation_var = classification_report(y_test_cancer, predicted_types_LR)

# Roc Curve
y_probability = LR.predict_proba(X_test_cancer)

# since we need only second column from y_probability
y_probability = y_probability[:, 1]

# to draw graph we need fpr, tpr, threshold
FPR, TPR, Thresholds = roc_curve(y_test_cancer, y_probability)
plt.plot(FPR, TPR)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()

# calculate roc area under curve score
roc_auc_score(y_test_cancer, y_probability)
