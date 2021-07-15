# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 09:49:00 2021

@author: Obioma Nwosu

Data pre-processing in Machine learning
"""

import pandas as pd
import seaborn as sn
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import minmax_scale, normalize

# The reason the full path of the data is not given is because we
# set the console as the working directory using the hamburger menu
# and clicking set console working directory
Data_Set1 = pd.read_csv('Data_Set.csv')

# start from the third row, skip first two rows
Data_Set2 = pd.read_csv('Data_Set.csv', header=2)

# Rename columns
Data_Set3 = Data_Set2.rename(columns={'Temperature': 'Temp'})

# Remove a column (0 for rows, 1 for columns)
# You can use Data_Set3.drop('No. Occupants', axis=1), inplace=True) without
# Creating a new Variable Data_Set4
Data_Set4 = Data_Set3.drop('No. Occupants', axis=1)

# Delete a row(3rd Row) inbetween our data
Data_Set5 = Data_Set4.drop(2, axis=0)

# Reset index
Data_Set6 = Data_Set5.reset_index(drop=True)

# shows the statistics of the dataset in the console ###########
Data_Set6.describe()

# Locate a particular item in the column of a data set min_val in this case
Min_item = Data_Set6['E_Heat'].min()

# get the particular value, this will print row number
Data_Set6['E_Heat'][Data_Set6['E_Heat'] == Min_item]

# Replace this number
Data_Set6['E_Heat'].replace(-4, 21, inplace=True)

# ############# COVARIANCE ##########################

# show covariance matrix
Data_Set6.cov()

# using seaborn heatmap to draw heatmap of the corelation of a data-set
sn.heatmap(Data_Set6.corr())

# missing values and all it's solution (nan/null)

# check the info of the data-set
Data_Set6.info()

# replace string or unwanted characters with something else
Data_Set7 = Data_Set6.replace('!', np.NaN)
Data_Set7.info()

# Change all colums to numerical columns
Data_Set7 = Data_Set7.apply(pd.to_numeric)

# find current location of all null values in your table
Data_Set7.isnull()

# remove the null values in the column by deleting the specific row
Data_Set7.drop(13, axis=0, inplace=True)

# removing all nan values at the same time by deleting all rows affected
Data_Set7.dropna(axis=0, inplace=True)

# ##### Better way is to replace the cells with NAN values ###
# ffill will use data before nan to fill NAN
# bfill will use the data after nan to fill NAN
Data_Set8 = Data_Set7.fillna(method='ffill')

# using ScikitLearn to fill data of nan values with specific numbers
M_var = SimpleImputer(missing_values=np.nan, strategy='mean')
M_var.fit(Data_Set7)

# Transform this new DataSet
Data_Set9 = M_var.transform(Data_Set7)

"""""
OUTLIER DETECTION
"""""

# show box plot to see outlier
Data_Set8.boxplot()

# The mathematics involved in outliers as explained in the note
Data_Set8['E_Plug'].quantile(0.25)
Data_Set8['E_Plug'].quantile(0.75)

"""

# the calculations in comment
# q1 = 19.75 from python
# q3 = 32.25 from python
# IQR = 32.25 - 19.75 = 12.5

Mild outlier

lower Bound = q1 - 1.5 * IQR = 19.75 - 1.5 * 12.5 = 1
Upper Bound = q3 + 1.5 * IQR = 32.25 + 1.5 * 12.5 = 51

EXtreme Outlier

Upper Bound = Q3 + 3*IQR = 32.25 + 3*12.5 = 69.75
following the result from the box plot we can see that we have a result 120
which is way above the xtreme outlier hence could be an outlier

"""
# Replacing the Outlier
Data_Set8['E_Plug'].replace(120, 42, inplace=True)

"""
CONCATENATION...

attaching datasets from rows and columns

for column-wise concatenation you use axis=1

for row-wise concatenation you use axis=0

"""
new_col = pd.read_csv('Data_New.csv')

# actual concatenation
Data_Set10 = pd.concat([Data_Set8, new_col], axis=1)

"""
Dummy Variable/Coding

used to give meaning to the machine e.g strings to pattern of numbers
"""
# using pandas to get dummy varaible
Data_Set11 = pd.get_dummies(Data_Set10)


"""
NORMALIZATION
"""
# using the minmaxscale method
Data_Set12 = minmax_scale(Data_Set11, feature_range=(0, 1))

# Using the normalize method
# axis = 0 for normalizing features/axis 1 for nomarlizing each sample
# norm = 'l2' seems to be the default value, l1 is the alternative

Data_Set13 = normalize(Data_Set11, norm='l2', axis=0)

# change back to Dtaframe as they change to array after normalization
Data_Set13 = pd.DataFrame(Data_Set13, columns=['Time', 'E_Plug', 'E_Heat',
                                               'Price', 'Temp',
                                               'Offpeak', 'Peak'])
