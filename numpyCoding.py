# -*- coding: utf-8 -*-
"""
Created on Wed May 12 10:25:06

@author: Obioma Nwosu

numpy coding

use ctrl + I to check documentation for package
"""

import numpy as np

#numpy arrays
number_array = np.array([[1,2,3], [4,5,6]])

#array multiplication
np1 = np.array([[1,3],[4,5]])
np2 = np.array([[3,4],[5,7]])

#using the @ multiplies the normal matrix way
multiplied_array = np1@np2

#using the dot class to multiply
multiplied_dot = np.dot(np1, np2)

#using the * symbol multiplies the coreesponding elem in same position
multiplied_direct = np1 * np2

#using the np multiply
multiplied_multi = np.multiply(np1, np2)


#ADDITION of arrays
sumed_array = np1 + np2

sumed_array1 = np.sum(np1)

#SUBTRACTION of arrays
subtracted_array = np1-np2

subtracted_array1 = np.subtract(np1, np2)

#broadcasting: python changes a number to an array size to be able to add
broadcast_num = np1 + 3
#this can also be donne with arrays of different sizes


#DIVISION of arrays in numpy
div_array = np.divide(np1, 5)

#to return integer number in place of float
div_array = np.floor_divide(np1, 5)


#Generating NORMAL DISTRIBUTION WITH numpy
ND = np.random.standard_normal((3,4))


#Generating Uniform Distribution
UD = np.random.uniform(1,12,(3,4))

#Generating random numbers for float and int respectively
randomFloat_arr = np.random.rand(2, 5,)

randomInt_arr = np.random.randint(1,50,(2,5))

#Generating zero arrays
zero_array = np.zeros((3,5))

#Generating ones arrays
ones_array = np.ones((1,3))

#filtering elements in an array
filtered_array = np.logical_and(randomInt_arr>12, randomInt_arr<30)
filtered_values = randomInt_arr[filtered_array]


#Statistics with numpy

Statistics_Data = np.array([1,3,5,4,7,9])

#getting average(Mean)
Mean_SD = np.mean(Statistics_Data)

#getting the Median
Meadian_SD = np.median(Statistics_Data)

#getting the variance
Variance_SD = np.var(Statistics_Data)

#getting the standard deviation
SD_SD = np.std(Statistics_Data)

#Statistics for multi dimensional array (axis=1 goes through rows)
multi_variance = np.var(number_array,axis=1)

#array (axis=0 goes through columns)
multi_varianceC = np.var(number_array, axis=0)