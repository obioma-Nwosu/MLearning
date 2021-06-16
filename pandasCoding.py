# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 11:18:27 2021

@author: Obioma Nwosu
"""

"""
Pandas(Panel data) designed for working with
DataFrames and Series
"""

import pandas as pd

# make our first SERIES
Age = pd.Series([10,20,30,40], index=['age1','age2','age3','age4'])

#To Access value of age3
Age.age3

#filter some of the data
filtered_age = Age[Age>20]

#get values in Age
Age.values

#get the index of the Age series
Age.index

#changing the values of the index of the Age series
Age.index = ['A1','A2','A3','A4']

#Create our DATA FRAME
import numpy as np

DF = np.array([[20,10,8],[25,8,10],[27,5,3],[30,9,7]]) 

#Changing the Above array DF to a datframe using pandas
Data_set = pd.DataFrame(DF)

#Changing the indexes and columns
#index
Data_set = pd.DataFrame(DF, index=['Student1','Student2','Student3','Student4'])

#colums
Data_set = pd.DataFrame(DF, index=['Student1','Student2','Student3','Student4'], columns=['Age','Maths','English'])

#add columns
Data_set['Physics'] = [9,6,7,10]

#loc And iloc
#loc (location)  used to select sepcial group of rows or columns by labels or boolean array
Data_set.loc['Student2']

#iloc extract first row, first column (first sample, first feature)
Data_set.iloc[1,3]

#specifying all the rows and specific column
Data_set.iloc[:,0]

#specifying the range of all rows and more columns(column 1 and 2 only)
new_data = Data_set.iloc[:,1:3]

#Deleting or dropping columns (AXis=1 is for colums, AXis=0 is for rows)
Data_set.drop('English',axis=1)

#Replacing values in our dataset
Data_set = Data_set.replace(10,12)

#Replacing Multiple values at once using Dictionary
Data_set = Data_set.replace({12:10, 9:30})

#Checking your dataset using heads and tails
#heads (Check 3 first rows)
Data_set.head(3)

#tails (Check 3 last rows)
Data_set.tail(3)

#sort values using column name
Data_set.sort_values('Age',ascending=False)

#sort index using (AXis=1 is for colums, AXis=0 is for rows)
Data_set.sort_index(axis=0, ascending=True)

#How to open a dataframe from csv format
#first set console as a working directory by clicking the three lines on this current panel and choosing the option

#open datafile
data = pd.read_csv('Data_Set.csv')
