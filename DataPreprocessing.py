# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 09:49:00 2021

@author: Obioma Nwosu

Data pre-processing in Machine learning
"""

import pandas as pd
import seaborn as sn
import numpy as np

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
